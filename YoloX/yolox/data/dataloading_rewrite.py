# -*- coding: utf-8 -*-

import os
import os.path
import cv2
import uuid
import copy
import torch
import pickle
import random
import itertools
import numpy as np
import torch.distributed as dist
import xml.etree.ElementTree as ET

from loguru import logger
from typing import Optional
from functools import wraps
from torch.utils.data.sampler import BatchSampler as torchBatchSampler
from torch.utils.data.sampler import Sampler
from torch.utils.data.dataloader import DataLoader as torchDataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.dataset import Dataset as torchDataset

from yolox.evaluators.voc_evaluator import voc_eval
from yolox.data.data_augment import random_affine
from yolox.utils import adjust_box_anns, get_local_rank # yolox/utils/allreduce_norm.py, yolox/utils/dist.py

# from datapool.datapool_syncontrol_v4 import synthesis_control


"""Cover yolox/data/sampler.py, yolox/data/dataloading.py, yolox/data/data_prefetcher.py,
         yolox/data/datasets/voc.py, yolox/data/datasets/mosaicdetection.py"""

# VOC_CLASSES = ("safe hat", "no safe hat")

VOC_CLASSES = ("aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", 
               "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor")

class AnnotationTransform(object):

    """Transforms a VOC annotation into a Tensor of bbox coords and label index
    Initilized with a dictionary lookup of classnames to indexes

    Arguments:
        class_to_ind (dict, optional): dictionary lookup of classnames -> indexes
            (default: alphabetic indexing of VOC's 20 classes)
        keep_difficult (bool, optional): keep difficult instances or not
            (default: False)
        height (int): height
        width (int): width
    """

    def __init__(self, class_to_ind=None, keep_difficult=True):
        self.class_to_ind = class_to_ind or dict(zip(VOC_CLASSES, range(len(VOC_CLASSES))))
        self.keep_difficult = keep_difficult

    def __call__(self, target):
        """
        Arguments:
            target (annotation) : the target annotation to be made usable
                will be an ET.Element
        Returns:
            a list containing lists of bounding boxes  [bbox coords, class name]
        """
        res = np.empty((0, 5))
        for obj in target.iter("object"):
            difficult = obj.find("difficult")
            if difficult is not None:
                difficult = int(difficult.text) == 1
            else:
                difficult = False
            if not self.keep_difficult and difficult:
                continue
            name = obj.find("name").text.strip()
            bbox = obj.find("bndbox")

            pts = ["xmin", "ymin", "xmax", "ymax"]
            bndbox = []
            for i, pt in enumerate(pts):
                cur_pt = int(float(bbox.find(pt).text)) - 1
                # scale height or width
                # cur_pt = cur_pt / width if i % 2 == 0 else cur_pt / height
                bndbox.append(cur_pt)
            label_idx = self.class_to_ind[name]
            bndbox.append(label_idx)
            res = np.vstack((res, bndbox))  # [xmin, ymin, xmax, ymax, label_ind]
            # img_id = target.find('filename').text[:-4]

        width = int(target.find("size").find("width").text)
        height = int(target.find("size").find("height").text)
        img_info = (height, width)

        return res, img_info


class Dataset(torchDataset): # 原yolox/data/data_prefetcher.py
    """ This class is a subclass of the base :class:`torch.utils.data.Dataset`,
    that enables on the fly resizing of the ``input_dim``.

    Args:
        input_dimension (tuple): (width,height) tuple with default dimensions of the network
    """

    def __init__(self, input_dimension, mosaic=True):
        super().__init__()
        self.__input_dim = input_dimension[:2]
        self.enable_mosaic = mosaic

    @property
    def input_dim(self):
        """
        Dimension that can be used by transforms to set the correct image size, etc.
        This allows transforms to have a single source of truth
        for the input dimension of the network.

        Return:
            list: Tuple containing the current width,height
        """
        if hasattr(self, "_input_dim"):
            return self._input_dim
        return self.__input_dim

    @staticmethod
    def mosaic_getitem(getitem_fn):
        """
        Decorator method that needs to be used around the ``__getitem__`` method. |br|
        This decorator enables the closing mosaic

        Example:
            >>> class CustomSet(ln.data.Dataset):
            ...     def __len__(self):
            ...         return 10
            ...     @ln.data.Dataset.mosaic_getitem
            ...     def __getitem__(self, index):
            ...         return self.enable_mosaic
        """

        @wraps(getitem_fn)
        def wrapper(self, index):
            if not isinstance(index, int):
                self.enable_mosaic = index[0]
                index = index[1]

            ret_val = getitem_fn(self, index)

            return ret_val

        return wrapper


class VOCDetection(Dataset):
    """
    VOC Detection Dataset Object

    input is image, target is annotation

    Args:
        root (string): filepath to VOCdevkit folder.
        image_set (string): imageset to use (eg. 'train', 'val', 'test')
        transform (callable, optional): transformation to perform on the
            input image
        target_transform (callable, optional): transformation to perform on the
            target `annotation`
            (eg: take in caption string, return tensor of word indices)
        dataset_name (string, optional): which dataset to load
            (default: 'VOC2007')
    """

    def __init__(self,data_dir, image_sets=[("2007", "trainval"), ("2012", "trainval")], img_size=(416, 416),
                 preproc=None, target_transform=AnnotationTransform(), dataset_name="VOC0712", cache=False):
        super().__init__(img_size)
        self.root = data_dir
        self.image_set = image_sets
        self.img_size = img_size
        self.preproc = preproc
        self.target_transform = target_transform
        self.name = dataset_name
        self._annopath = os.path.join("%s", "Annotations", "%s.xml")
        self._imgpath = os.path.join("%s", "JPEGImages", "%s.jpg")
        self._classes = VOC_CLASSES
        self.ids = list()
        for (year, name) in image_sets:
            self._year = year
            rootpath = os.path.join(self.root, "VOC" + year)
            for line in open(os.path.join(rootpath, "ImageSets", "Main", name + ".txt")):
                self.ids.append((rootpath, line.strip())) # 读取训练集中每张图片的id ('F:/VOCtrainval/VOCdevkit\\VOC2012', '2011_003275')

        self.annotations = self._load_coco_annotations()
        self.imgs = None
        if cache: # 这个参数由parser模块控制,默认为False
            self._cache_images()

    def __len__(self):
        return len(self.ids)

    def _load_coco_annotations(self):
        return [self.load_anno_from_ids(_ids) for _ids in range(len(self.ids))]

    def _cache_images(self):
        logger.warning("\n********************************************************************************\n"
                       "You are using cached images in RAM to accelerate training.\n"
                       "This requires large system RAM.\n"
                       "Make sure you have 60G+ RAM and 19G available disk space for training VOC.\n"
                       "********************************************************************************\n")
        max_h = self.img_size[0]
        max_w = self.img_size[1]
        cache_file = os.path.join(self.root, f"img_resized_cache_{self.name}.array")
        if not os.path.exists(cache_file):
            logger.info("Caching images for the first time. This might take about 3 minutes for VOC")
            self.imgs = np.memmap(cache_file,shape=(len(self.ids), max_h, max_w, 3),dtype=np.uint8,mode="w+")
            from tqdm import tqdm
            from multiprocessing.pool import ThreadPool

            NUM_THREADs = min(8, os.cpu_count())
            loaded_images = ThreadPool(NUM_THREADs).imap(lambda x: self.load_resized_img(x),range(len(self.annotations)))
            pbar = tqdm(enumerate(loaded_images), total=len(self.annotations))
            for k, out in pbar:
                self.imgs[k][: out.shape[0], : out.shape[1], :] = out.copy()
            self.imgs.flush()
            pbar.close()
        else:
            logger.warning("You are using cached imgs! Make sure your dataset is not changed!!\n"
                           "Everytime the self.input_size is changed in your exp file, you need to delete\n"
                           "the cached data and re-generate them.\n")

        logger.info("Loading cached imgs...")
        self.imgs = np.memmap(cache_file,shape=(len(self.ids), max_h, max_w, 3),dtype=np.uint8,mode="r+")

    def load_anno_from_ids(self, index):
        img_id = self.ids[index]
        target = ET.parse(self._annopath % img_id).getroot()

        assert self.target_transform is not None
        res, img_info = self.target_transform(target)
        height, width = img_info

        r = min(self.img_size[0] / height, self.img_size[1] / width)
        res[:, :4] *= r
        resized_info = (int(height * r), int(width * r))

        return (res, img_info, resized_info)

    def load_anno(self, index):
        return self.annotations[index][0]

    def load_resized_img(self, index):
        img = self.load_image(index)
        r = min(self.img_size[0] / img.shape[0], self.img_size[1] / img.shape[1])
        resized_img = cv2.resize(img,(int(img.shape[1] * r), int(img.shape[0] * r)),interpolation=cv2.INTER_LINEAR).astype(np.uint8)

        return resized_img

    def load_image(self, index):
        img_id = self.ids[index]
        img = cv2.imread(self._imgpath % img_id, cv2.IMREAD_COLOR)
        assert img is not None

        return img

    def pull_item(self, index):
        """Returns the original image and target at an index for mixup

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            img, target
        """
        if self.imgs is not None:
            target, img_info, resized_info = self.annotations[index]
            pad_img = self.imgs[index]
            img = pad_img[: resized_info[0], : resized_info[1], :].copy()
        else:
            img = self.load_resized_img(index)
            target, img_info, _ = self.annotations[index]

        return img, target, img_info, index

    @Dataset.mosaic_getitem
    def __getitem__(self, index):
        img, target, img_info, img_id = self.pull_item(index)

        if self.preproc is not None:
            img, target = self.preproc(img, target, self.input_dim)

        return img, target, img_info, img_id

    def evaluate_detections(self, all_boxes, output_dir=None):
        """
        all_boxes is a list of length number-of-classes.
        Each list element is a list of length number-of-images.
        Each of those list elements is either an empty list []
        or a numpy array of detection.

        all_boxes[class][image] = [] or np.array of shape #dets x 5
        """
        self._write_voc_results_file(all_boxes)
        IouTh = np.linspace(0.5, 0.95, int(np.round((0.95 - 0.5) / 0.05)) + 1, endpoint=True)
        mAPs = []
        for iou in IouTh:
            mAP = self._do_python_eval(output_dir, iou)
            mAPs.append(mAP)

        print("--------------------------------------------------------------")
        print("map_5095:", np.mean(mAPs))
        print("map_50:", mAPs[0])
        print("--------------------------------------------------------------")
        return np.mean(mAPs), mAPs[0]

    def _get_voc_results_file_template(self):
        filename = "comp4_det_test" + "_{:s}.txt"
        filedir = os.path.join(self.root, "results", "VOC" + self._year, "Main")
        if not os.path.exists(filedir):
            os.makedirs(filedir)
        path = os.path.join(filedir, filename)
        return path

    def _write_voc_results_file(self, all_boxes):
        for cls_ind, cls in enumerate(VOC_CLASSES):
            cls_ind = cls_ind
            if cls == "__background__":
                continue
            print("Writing {} VOC results file".format(cls))
            filename = self._get_voc_results_file_template().format(cls)
            with open(filename, "wt") as f:
                for im_ind, index in enumerate(self.ids):
                    index = index[1]
                    dets = all_boxes[cls_ind][im_ind]
                    """
                    for i in range(len(all_boxes)):
                        print(i)
                        print(all_boxes[i])
                        print("-----")
                    print("----------")
                    """
                    if dets == []:
                        continue
                    for k in range(dets.shape[0]):
                        f.write("{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n".format(index,
                                                                                   dets[k, -1],
                                                                                   dets[k, 0] + 1,
                                                                                   dets[k, 1] + 1,
                                                                                   dets[k, 2] + 1,
                                                                                   dets[k, 3] + 1))

    def _do_python_eval(self, output_dir="output", iou=0.5):
        rootpath = os.path.join(self.root, "VOC" + self._year)
        name = self.image_set[0][1]
        annopath = os.path.join(rootpath, "Annotations", "{:s}.xml")
        imagesetfile = os.path.join(rootpath, "ImageSets", "Main", name + ".txt")
        cachedir = os.path.join(self.root, "annotations_cache", "VOC" + self._year, name)
        if not os.path.exists(cachedir):
            os.makedirs(cachedir)
        aps = []
        # The PASCAL VOC metric changed in 2010
        use_07_metric = True if int(self._year) < 2010 else False
        print("Eval IoU : {:.2f}".format(iou))
        if output_dir is not None and not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        for i, cls in enumerate(VOC_CLASSES):

            if cls == "__background__":
                continue

            filename = self._get_voc_results_file_template().format(cls)
            rec, prec, ap = voc_eval(filename,annopath,imagesetfile,cls,cachedir,ovthresh=iou,use_07_metric=use_07_metric)
            aps += [ap]
            if iou == 0.5:
                print("AP for {} = {:.4f}".format(cls, ap))
            if output_dir is not None:
                with open(os.path.join(output_dir, cls + "_pr.pkl"), "wb") as f:
                    pickle.dump({"rec": rec, "prec": prec, "ap": ap}, f)
        if iou == 0.5:
            print("Mean AP = {:.4f}".format(np.mean(aps)))
            print("~~~~~~~~")
            print("Results:")
            for ap in aps:
                print("{:.3f}".format(ap))
            print("{:.3f}".format(np.mean(aps)))
            print("~~~~~~~~")
            print("")
            print("--------------------------------------------------------------")
            print("Results computed with the **unofficial** Python eval code.")
            print("Results should be very close to the official MATLAB eval code.")
            print("Recompute with `./tools/reval.py --matlab ...` for your paper.")
            print("-- Thanks, The Management")
            print("--------------------------------------------------------------")

        return np.mean(aps)
    

def get_mosaic_coordinate(mosaic_image, mosaic_index, xc, yc, w, h, input_h, input_w):
    # TODO update doc
    # index0 to top left part of image
    if mosaic_index == 0:
        x1, y1, x2, y2 = max(xc - w, 0), max(yc - h, 0), xc, yc
        small_coord = w - (x2 - x1), h - (y2 - y1), w, h
    # index1 to top right part of image
    elif mosaic_index == 1:
        x1, y1, x2, y2 = xc, max(yc - h, 0), min(xc + w, input_w * 2), yc
        small_coord = 0, h - (y2 - y1), min(w, x2 - x1), h
    # index2 to bottom left part of image
    elif mosaic_index == 2:
        x1, y1, x2, y2 = max(xc - w, 0), yc, xc, min(input_h * 2, yc + h)
        small_coord = w - (x2 - x1), 0, w, min(y2 - y1, h)
    # index2 to bottom right part of image
    elif mosaic_index == 3:
        x1, y1, x2, y2 = xc, yc, min(xc + w, input_w * 2), min(input_h * 2, yc + h)  # noqa
        small_coord = 0, 0, min(w, x2 - x1), min(y2 - y1, h)
    return (x1, y1, x2, y2), small_coord


class MosaicDetection(Dataset):
    """Detection dataset wrapper that performs mixup for normal dataset."""

    def __init__(self, dataset, img_size, mosaic=True, preproc=None, degrees=10.0, translate=0.1, mosaic_scale=(0.5, 1.5),
                 mixup_scale=(0.5, 1.5), shear=2.0, enable_mixup=True, mosaic_prob=1.0, mixup_prob=1.0, *args):
        """
        Args:
            dataset(Dataset) : Pytorch dataset object.
            img_size (tuple):
            mosaic (bool): enable mosaic augmentation or not.
            preproc (func):
            degrees (float):
            translate (float):
            mosaic_scale (tuple):
            mixup_scale (tuple):
            shear (float):
            enable_mixup (bool):
            *args(tuple) : Additional arguments for mixup random sampler.
        """
        super().__init__(img_size, mosaic=mosaic)
        self._dataset = dataset
        self.preproc = preproc
        self.degrees = degrees
        self.translate = translate
        self.scale = mosaic_scale
        self.shear = shear
        self.mixup_scale = mixup_scale
        self.enable_mosaic = mosaic # 这个参数实际由build.py的no_aug_epochs控制,也就是从0开始使用mosaic增强的轮次
        self.enable_mixup = enable_mixup
        self.mosaic_prob = mosaic_prob
        self.mixup_prob = mixup_prob
        self.local_rank = get_local_rank()

    def __len__(self):
        return len(self._dataset)

    @Dataset.mosaic_getitem
    def __getitem__(self, idx):
        if self.enable_mosaic and random.random() < self.mosaic_prob:
            mosaic_labels = []
            input_dim = self._dataset.input_dim
            input_h, input_w = input_dim[0], input_dim[1]

            # yc, xc = s, s  # mosaic center x, y
            yc = int(random.uniform(0.5 * input_h, 1.5 * input_h))
            xc = int(random.uniform(0.5 * input_w, 1.5 * input_w))

            # 3 additional image indices
            indices = [idx] + [random.randint(0, len(self._dataset) - 1) for _ in range(3)]

            for i_mosaic, index in enumerate(indices):
                img, _labels, _, img_id = self._dataset.pull_item(index)
                h0, w0 = img.shape[:2]  # orig hw
                scale = min(1. * input_h / h0, 1. * input_w / w0)
                img = cv2.resize(img, (int(w0 * scale), int(h0 * scale)), interpolation=cv2.INTER_LINEAR)
                
                # generate output mosaic image
                (h, w, c) = img.shape[:3]
                if i_mosaic == 0:
                    mosaic_img = np.full((input_h * 2, input_w * 2, c), 114, dtype=np.uint8)

                # suffix l means large image, while s means small image in mosaic aug.
                (l_x1, l_y1, l_x2, l_y2), (s_x1, s_y1, s_x2, s_y2) = get_mosaic_coordinate(mosaic_img, i_mosaic, xc, yc, w, h, input_h, input_w)

                mosaic_img[l_y1:l_y2, l_x1:l_x2] = img[s_y1:s_y2, s_x1:s_x2]
                padw, padh = l_x1 - s_x1, l_y1 - s_y1

                labels = _labels.copy()
                
                # Normalized xywh to pixel xyxy format
                if _labels.size > 0:
                    labels[:, 0] = scale * _labels[:, 0] + padw
                    labels[:, 1] = scale * _labels[:, 1] + padh
                    labels[:, 2] = scale * _labels[:, 2] + padw
                    labels[:, 3] = scale * _labels[:, 3] + padh
                mosaic_labels.append(labels)

            if len(mosaic_labels):
                mosaic_labels = np.concatenate(mosaic_labels, 0)
                np.clip(mosaic_labels[:, 0], 0, 2 * input_w, out=mosaic_labels[:, 0])
                np.clip(mosaic_labels[:, 1], 0, 2 * input_h, out=mosaic_labels[:, 1])
                np.clip(mosaic_labels[:, 2], 0, 2 * input_w, out=mosaic_labels[:, 2])
                np.clip(mosaic_labels[:, 3], 0, 2 * input_h, out=mosaic_labels[:, 3])

            mosaic_img, mosaic_labels = random_affine(mosaic_img, mosaic_labels, target_size=(input_w, input_h), degrees=self.degrees,
                                                      translate=self.translate, scales=self.scale, shear=self.shear)

            # CopyPaste: https://arxiv.org/abs/2012.07177
            if (self.enable_mixup and not len(mosaic_labels) == 0 and random.random() < self.mixup_prob):
                mosaic_img, mosaic_labels = self.mixup(mosaic_img, mosaic_labels, self.input_dim)
                
            mix_img, padded_labels = self.preproc(mosaic_img, mosaic_labels, self.input_dim)
            img_info = (mix_img.shape[1], mix_img.shape[0])

            # img_info and img_id are not used for training.
            # They are also hard to be specified on a mosaic image.
            return mix_img, padded_labels, img_info, img_id

        else:
            self._dataset._input_dim = self.input_dim
            img, label, img_info, img_id = self._dataset.pull_item(idx)
            img, label = self.preproc(img, label, self.input_dim)
            return img, label, img_info, img_id

    def mixup(self, origin_img, origin_labels, input_dim):
        jit_factor = random.uniform(*self.mixup_scale)
        FLIP = random.uniform(0, 1) > 0.5
        cp_labels = []
        while len(cp_labels) == 0:
            cp_index = random.randint(0, self.__len__() - 1)
            cp_labels = self._dataset.load_anno(cp_index)
        img, cp_labels, _, _ = self._dataset.pull_item(cp_index)
        
        if len(img.shape) == 3:
            cp_img = np.ones((input_dim[0], input_dim[1], 3), dtype=np.uint8) * 114
        else:
            cp_img = np.ones(input_dim, dtype=np.uint8) * 114

        cp_scale_ratio = min(input_dim[0] / img.shape[0], input_dim[1] / img.shape[1])
        resized_img = cv2.resize(img, (int(img.shape[1] * cp_scale_ratio), int(img.shape[0] * cp_scale_ratio)), interpolation=cv2.INTER_LINEAR)

        cp_img[: int(img.shape[0] * cp_scale_ratio), : int(img.shape[1] * cp_scale_ratio)] = resized_img

        cp_img = cv2.resize(cp_img, (int(cp_img.shape[1] * jit_factor), int(cp_img.shape[0] * jit_factor)))
        
        cp_scale_ratio *= jit_factor

        if FLIP:
            cp_img = cp_img[:, ::-1, :]

        origin_h, origin_w = cp_img.shape[:2]
        target_h, target_w = origin_img.shape[:2]
        padded_img = np.zeros((max(origin_h, target_h), max(origin_w, target_w), 3), dtype=np.uint8)
        padded_img[:origin_h, :origin_w] = cp_img

        x_offset, y_offset = 0, 0
        if padded_img.shape[0] > target_h:
            y_offset = random.randint(0, padded_img.shape[0] - target_h - 1)
        if padded_img.shape[1] > target_w:
            x_offset = random.randint(0, padded_img.shape[1] - target_w - 1)
        padded_cropped_img = padded_img[y_offset: y_offset + target_h, x_offset: x_offset + target_w]

        cp_bboxes_origin_np = adjust_box_anns(cp_labels[:, :4].copy(), cp_scale_ratio, 0, 0, origin_w, origin_h)
        if FLIP:
            cp_bboxes_origin_np[:, 0::2] = (origin_w - cp_bboxes_origin_np[:, 0::2][:, ::-1])
        cp_bboxes_transformed_np = cp_bboxes_origin_np.copy()
        cp_bboxes_transformed_np[:, 0::2] = np.clip(cp_bboxes_transformed_np[:, 0::2] - x_offset, 0, target_w)
        cp_bboxes_transformed_np[:, 1::2] = np.clip(cp_bboxes_transformed_np[:, 1::2] - y_offset, 0, target_h)

        cls_labels = cp_labels[:, 4:5].copy()
        box_labels = cp_bboxes_transformed_np
        labels = np.hstack((box_labels, cls_labels))
        origin_labels = np.vstack((origin_labels, labels))
        origin_img = origin_img.astype(np.float32)
        origin_img = 0.5 * origin_img + 0.5 * padded_cropped_img.astype(np.float32)

        return origin_img.astype(np.uint8), origin_labels


class MosaicDetection_Stitcher(Dataset):
    """Detection dataset wrapper that performs mixup for normal dataset."""

    def __init__(self, dataset, img_size, preproc=None, *args):
        super().__init__(img_size)
        self._dataset = dataset
        self.preproc = preproc

    def __len__(self):
        return len(self._dataset)

    @Dataset.mosaic_getitem
    def __getitem__(self, idx): # 真正被执行的是这个函数
        self._dataset._input_dim = self.input_dim
        img, label, img_info, img_id = self._dataset.pull_item(idx) # 此处是resize_img, height width channel, label.shape=[2, 5]
        
        # print("resize img: ", img.shape) # [h, w, c]
        # print("label before", label) # xyxy
        
        
        # 缩放后填充前, xyxy2xywh+填充标签
        img = np.transpose(img, (2, 0, 1))
        boxes = label[:, :4].copy()
        classes = label[:, 4].copy()
        for i in range(boxes.shape[0]):
            scaled_xcen = (boxes[i][2] + boxes[i][0]) / 2
            scaled_ycen = (boxes[i][3] + boxes[i][1]) / 2
            scaled_w = boxes[i][2] - boxes[i][0]
            scaled_h = boxes[i][3] - boxes[i][1]
            boxes[i][0] = scaled_xcen
            boxes[i][1] = scaled_ycen
            boxes[i][2] = scaled_w
            boxes[i][3] = scaled_h
        classes = np.expand_dims(classes, 1)
        targets_t = np.hstack((classes, boxes))
        label = np.zeros((120, 5))
        label[range(len(targets_t))[: 120]] = targets_t[: 120]
        label = np.ascontiguousarray(label, dtype=np.float32)
        
        
        # img, label = self.preproc(img, label, self.input_dim) # 此处是pad_img channel, height width, label.shape=[120, 5]
        
        # print("padding img: ", img.shape) # [c, h, w] this is correct
        # print("label after", label) # xywh
        
        return img, label, img_info, img_id


class MosaicDetection_DataPool(Dataset):
    """Detection dataset wrapper that performs mixup for normal dataset."""

    def __init__(self, dataset, batch_size, csv_path, img_size, preproc=None, *args):
        super().__init__(img_size)
        self._dataset = dataset
        self.preproc = preproc
        self.batchsize = batch_size
        self.csv_path = csv_path

    def __len__(self):
        return len(self._dataset)

    @Dataset.mosaic_getitem
    def __getitem__(self, idx): # 真正被执行的是这个函数
        # self._dataset._input_dim = self.input_dim
        # img, label, img_info, img_id = self._dataset.pull_item(idx) # 此处是resize_img, height width channel, label.shape=[2, 5]
        
        # print("resize img: ", img.shape) # [h, w, c]
        # print("label before", label) # xyxy
        
        img, label = synthesis_control(self.csv_path)
        label = label.numpy()
        
        # 缩放后填充前, xyxy2xywh+填充标签
        img = np.transpose(img, (2, 0, 1))
        boxes = label[:, :4].copy()
        classes = label[:, 4].copy()
        for i in range(boxes.shape[0]):
            scaled_xcen = (boxes[i][2] + boxes[i][0]) / 2
            scaled_ycen = (boxes[i][3] + boxes[i][1]) / 2
            scaled_w = boxes[i][2] - boxes[i][0]
            scaled_h = boxes[i][3] - boxes[i][1]
            boxes[i][0] = scaled_xcen
            boxes[i][1] = scaled_ycen
            boxes[i][2] = scaled_w
            boxes[i][3] = scaled_h
        classes = np.expand_dims(classes, 1)
        targets_t = np.hstack((classes, boxes))
        label = np.zeros((120, 5))
        label[range(len(targets_t))[: 120]] = targets_t[: 120]
        label = np.ascontiguousarray(label, dtype=np.float32)
        
        # img, label = self.preproc(img, label, self.input_dim) # 此处是pad_img channel, height width, label.shape=[120, 5]
        
        # print("padding img: ", img.shape) # [c, h, w], but cv2.imwrite require [h, w, c]
        # print("label after", label[0]) # [cls, xcen, ycen, width, height]
        
        """# visualize code, verfiy the correctness of labels
        _COLORS = np.array([0.000, 0.447, 0.741]).astype(np.float32).reshape(-1,3)
        temp_img = copy.deepcopy(img)
        temp_img = np.transpose(temp_img, (1, 2, 0))
        for i in range(label.shape[0]):
            xmin = int(label[i][1] - label[i][3] / 2)
            ymin = int(label[i][2] - label[i][4] / 2)
            xmax = int(label[i][1] + label[i][3] / 2)
            ymax = int(label[i][2] + label[i][4] / 2)
            color = (_COLORS[0]).astype(np.uint8).tolist()
            cv2.rectangle(temp_img, (xmin, ymin), (xmax, ymax), color, 2)
        cv2.imwrite('/home/fyp1/ChengYuxuan/YoloXDSTRe/test/' + str(random.randint(0, 500)) + '.jpg', temp_img)"""
        
        return img, label, 0, 0 #  img_info, img_id # img_info和img_id实际上在train_rewrite.py的DataPrefetcher()的preload()中并没有被利用


class YoloBatchSampler(torchBatchSampler):
    """
    This batch sampler will generate mini-batches of (mosaic, index) tuples from another sampler.
    It works just like the :class:`torch.utils.data.sampler.BatchSampler`,
    but it will turn on/off the mosaic aug.
    """

    def __init__(self, *args, mosaic=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.mosaic = mosaic

    def __iter__(self):
        for batch in super().__iter__():
            yield [(self.mosaic, idx) for idx in batch]


class InfiniteSampler(Sampler):
    """
    In training, we only care about the "infinite stream" of training data.
    So this sampler produces an infinite stream of indices and
    all workers cooperate to correctly shuffle the indices and sample different indices.
    The samplers in each worker effectively produces `indices[worker_id::num_workers]`
    where `indices` is an infinite stream of indices consisting of
    `shuffle(range(size)) + shuffle(range(size)) + ...` (if shuffle is True)
    or `range(size) + range(size) + ...` (if shuffle is False)
    """

    def __init__(self, size: int, shuffle: bool = True, seed: Optional[int] = 0, rank=0, world_size=1):
        """
        Args:
            size (int): the total number of data of the underlying dataset to sample from
            shuffle (bool): whether to shuffle the indices or not
            seed (int): the initial seed of the shuffle. Must be the same
                across all workers. If None, will use a random seed shared
                among workers (require synchronization among all workers).
        """
        self._size = size
        assert size > 0
        self._shuffle = shuffle
        self._seed = int(seed)

        if dist.is_available() and dist.is_initialized():
            self._rank = dist.get_rank()
            self._world_size = dist.get_world_size()
        else:
            self._rank = rank
            self._world_size = world_size

    def __iter__(self):
        start = self._rank
        yield from itertools.islice(self._infinite_indices(), start, None, self._world_size)

    def _infinite_indices(self):
        g = torch.Generator()
        g.manual_seed(self._seed)
        while True:
            if self._shuffle:
                yield from torch.randperm(self._size, generator=g)
            else:
                yield from torch.arange(self._size)

    def __len__(self):
        return self._size // self._world_size


def get_yolox_datadir():
    """
    get dataset dir of YOLOX. If environment variable named `YOLOX_DATADIR` is set,
    this function will return value of the environment variable. Otherwise, use data
    """
    yolox_datadir = os.getenv("YOLOX_DATADIR", None)
    if yolox_datadir is None:
        import yolox
        yolox_path = os.path.dirname(os.path.dirname(yolox.__file__))
        yolox_datadir = os.path.join(yolox_path, "datasets")
    return yolox_datadir


class DataLoader(torchDataLoader):
    """
    Lightnet dataloader that enables on the fly resizing of the images.
    See :class:`torch.utils.data.DataLoader` for more information on the arguments.
    Check more on the following website:
    https://gitlab.com/EAVISE/lightnet/-/blob/master/lightnet/data/_dataloading.py
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__initialized = False
        shuffle = False
        batch_sampler = None
        if len(args) > 5: # 经过测试len(args)=1
            shuffle = args[2]
            sampler = args[3]
            batch_sampler = args[4]
        elif len(args) > 4:
            shuffle = args[2]
            sampler = args[3]
            if "batch_sampler" in kwargs:
                batch_sampler = kwargs["batch_sampler"]
        elif len(args) > 3:
            shuffle = args[2]
            if "sampler" in kwargs:
                sampler = kwargs["sampler"]
            if "batch_sampler" in kwargs:
                batch_sampler = kwargs["batch_sampler"]
        else:
            if "shuffle" in kwargs:
                shuffle = kwargs["shuffle"]
            if "sampler" in kwargs:
                sampler = kwargs["sampler"]
            if "batch_sampler" in kwargs:
                batch_sampler = kwargs["batch_sampler"]

        # Use custom BatchSampler
        if batch_sampler is None:
            if sampler is None:
                if shuffle:
                    sampler = torch.utils.data.sampler.RandomSampler(self.dataset)
                    # sampler = torch.utils.data.DistributedSampler(self.dataset)
                else:
                    sampler = torch.utils.data.sampler.SequentialSampler(self.dataset)
            batch_sampler = YoloBatchSampler(sampler, self.batch_size, self.drop_last, input_dimension=self.dataset.input_dim)
            # batch_sampler = IterationBasedBatchSampler(batch_sampler, num_iterations =
        
        self.batch_sampler = batch_sampler
        
        self.__initialized = True
        
    def close_mosaic(self):
        self.batch_sampler.mosaic = False


class DataLoader_Stitcher(torchDataLoader):
    """
    Lightnet dataloader that enables on the fly resizing of the images.
    See :class:`torch.utils.data.DataLoader` for more information on the arguments.
    Check more on the following website:
    https://gitlab.com/EAVISE/lightnet/-/blob/master/lightnet/data/_dataloading.py
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__initialized = False
        shuffle = False
        batch_sampler_ir = None
        collate_fn = None
        if len(args) > 5: # 经过测试len(args)=1
            shuffle = args[2]
            sampler = args[3]
            batch_sampler_ir = args[4]
        elif len(args) > 4:
            shuffle = args[2]
            sampler = args[3]
            if "batch_sampler" in kwargs:
                batch_sampler_ir = kwargs["batch_sampler"]
        elif len(args) > 3:
            shuffle = args[2]
            if "sampler" in kwargs:
                sampler = kwargs["sampler"]
            if "batch_sampler" in kwargs:
                batch_sampler_ir = kwargs["batch_sampler"]
        else:
            if "shuffle" in kwargs:
                shuffle = kwargs["shuffle"]
            if "sampler" in kwargs:
                sampler = kwargs["sampler"]
            if "batch_sampler" in kwargs:
                batch_sampler_ir = kwargs["batch_sampler"]
            if "collate_fn" in kwargs:
                collate_fn = kwargs["collate_fn"]

        # Use custom BatchSampler
        if batch_sampler_ir is None:
            if sampler is None:
                if shuffle:
                    sampler = torch.utils.data.sampler.RandomSampler(self.dataset)
                    # sampler = torch.utils.data.DistributedSampler(self.dataset)
                else:
                    sampler = torch.utils.data.sampler.SequentialSampler(self.dataset)
            batch_sampler_ir = YoloBatchSampler(sampler, self.batch_size, self.drop_last, input_dimension=self.dataset.input_dim)
            # batch_sampler = IterationBasedBatchSampler(batch_sampler, num_iterations =
        
        self.__initialized = False
        self.batch_sampler = batch_sampler_ir
        self.collate_fn = collate_fn
        self.__initialized = True

    def close_mosaic(self):
        self.batch_sampler.mosaic = False


def list_collate(batch):
    """
    Function that collates lists or tuples together into one list (of lists/tuples).
    Use this as the collate function in a Dataloader, if you want to have a list of
    items as an output, as opposed to tensors (eg. Brambox.boxes).
    """
    print("list_collate")
    items = list(zip(*batch))

    for i in range(len(items)):
        if isinstance(items[i][0], (list, tuple)):
            items[i] = list(items[i])
        else:
            items[i] = default_collate(items[i])

    return items


def worker_init_reset_seed(worker_id):
    seed = uuid.uuid4().int % 2**32
    random.seed(seed)
    torch.set_rng_state(torch.manual_seed(seed).get_state())
    np.random.seed(seed)
