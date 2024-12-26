# Copyright (c) Facebook, Inc. and its affiliates.
import os
import cv2
import copy
import torch
import pickle
import random
import logging
import itertools
import contextlib
import numpy as np

from typing import Callable, Union

import torch.utils.data as data
from torch.utils.data.sampler import Sampler

from detectron2.structures import Boxes, Instances
from detectron2.utils.serialize import PicklableWrapper

from detectron2.data.data_augment import random_affine, TrainTransform

# from datapool.datapool_analysis_control import analysis_main_coco # DataPool Analysis Main Function
# from datapool.datapool_syncontrol_v4 import synthesis_preproc # DataPool Analysis Secondary Function
# from datapool.datapool_syncontrol_v4 import synthesis_control # DataPool Synthesis Primary Function

__all__ = ["MapDataset", "DatasetFromList", "AspectRatioGroupedDataset", "ToIterableDataset"]

logger = logging.getLogger(__name__)


# copied from: https://docs.python.org/3/library/itertools.html#recipes
def _roundrobin(*iterables):
    "roundrobin('ABC', 'D', 'EF') --> A D E B F C"
    # Recipe credited to George Sakkis
    num_active = len(iterables)
    nexts = itertools.cycle(iter(it).__next__ for it in iterables)
    while num_active:
        try:
            for next in nexts:
                yield next()
        except StopIteration:
            # Remove the iterator we just exhausted from the cycle.
            num_active -= 1
            nexts = itertools.cycle(itertools.islice(nexts, num_active))


def _shard_iterator_dataloader_worker(iterable, chunk_size=1):
    # Shard the iterable if we're currently inside pytorch dataloader worker.
    worker_info = data.get_worker_info()
    if worker_info is None or worker_info.num_workers == 1:
        # do nothing
        yield from iterable
    else:
        # worker0: 0, 1, ..., chunk_size-1, num_workers*chunk_size, num_workers*chunk_size+1, ...
        # worker1: chunk_size, chunk_size+1, ...
        # worker2: 2*chunk_size, 2*chunk_size+1, ...
        # ...
        yield from _roundrobin(*[itertools.islice(iterable, worker_info.id * chunk_size + chunk_i, None, worker_info.num_workers * chunk_size) for chunk_i in range(chunk_size)])


class _MapIterableDataset(data.IterableDataset):
    """Map a function over elements in an IterableDataset.

    Similar to pytorch's MapIterDataPipe, but support filtering when map_func
    returns None.

    This class is not public-facing. Will be called by `MapDataset`."""
    
    def __init__(self, dataset, map_func):
        self._dataset = dataset
        self._map_func = PicklableWrapper(map_func)  # wrap so that a lambda will work

    def __len__(self):
        return len(self._dataset)

    def __iter__(self):
        for x in map(self._map_func, self._dataset):
            if x is not None:
                yield x


class MapDataset(data.Dataset):
    """Map a function over the elements in a dataset."""
    def __init__(self, dataset, map_func):
        
        """Args:
            dataset: a dataset where map function is applied. Can be either
                map-style or iterable dataset. When given an iterable dataset,
                the returned object will also be an iterable dataset.
            map_func: a callable which maps the element in dataset. map_func can
                return None to skip the data (e.g. in case of errors).
                How None is handled depends on the style of `dataset`.
                If `dataset` is map-style, it randomly tries other elements.
                If `dataset` is iterable, it skips the data and tries the next."""
        
        self._dataset = dataset
        self._map_func = PicklableWrapper(map_func)  # wrap so that a lambda will work

        self._rng = random.Random(42)
        self._fallback_candidates = set(range(len(dataset)))

    def __new__(cls, dataset, map_func):
        is_iterable = isinstance(dataset, data.IterableDataset)
        if is_iterable:
            return _MapIterableDataset(dataset, map_func)
        else:
            return super().__new__(cls)

    def __getnewargs__(self):
        return self._dataset, self._map_func

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, idx):
        retry_count = 0
        cur_idx = int(idx)

        while True:
            data = self._map_func(self._dataset[cur_idx])
            if data is not None:
                self._fallback_candidates.add(cur_idx)
                return data # this is the point providing data to model

            # _map_func fails for this idx, use a random new index from the pool
            retry_count += 1
            self._fallback_candidates.discard(cur_idx)
            cur_idx = self._rng.sample(self._fallback_candidates, k=1)[0]

            if retry_count >= 3:
                logger = logging.getLogger(__name__)
                logger.warning("Failed to apply `_map_func` for idx: {}, retry count: {}".format(idx, retry_count))


class _TorchSerializedList:
    """A list-like object whose items are serialized and stored in a torch tensor. When
    launching a process that uses TorchSerializedList with "fork" start method,
    the subprocess can read the same buffer without triggering copy-on-access. When
    launching a process that uses TorchSerializedList with "spawn/forkserver" start
    method, the list will be pickled by a special ForkingPickler registered by PyTorch
    that moves data to shared memory. In both cases, this allows parent and child
    processes to share RAM for the list data, hence avoids the issue in
    https://github.com/pytorch/pytorch/issues/13246.

    See also https://ppwwyyxx.com/blog/2022/Demystify-RAM-Usage-in-Multiprocess-DataLoader/
    on how it works."""

    def __init__(self, lst: list):
        self._lst = lst # lst is annots, total 117266 images, since been divided as train & val

        def _serialize(data):
            buffer = pickle.dumps(data, protocol=-1)
            return np.frombuffer(buffer, dtype=np.uint8)

        logger.info("Serializing {} elements to byte tensors and concatenating them all ...".format(len(self._lst)))
        self._lst = [_serialize(x) for x in self._lst]
        self._addr = np.asarray([len(x) for x in self._lst], dtype=np.int64)
        self._addr = torch.from_numpy(np.cumsum(self._addr))
        self._lst = torch.from_numpy(np.concatenate(self._lst))
        logger.info("Serialized dataset takes {:.2f} MiB".format(len(self._lst) / 1024**2))

    def __len__(self):
        return len(self._addr)

    def __getitem__(self, idx):
        start_addr = 0 if idx == 0 else self._addr[idx - 1].item()
        end_addr = self._addr[idx].item()
        bytes = memoryview(self._lst[start_addr:end_addr].numpy())

        # @lint-ignore PYTHONPICKLEISBAD
        return pickle.loads(bytes)


_DEFAULT_DATASET_FROM_LIST_SERIALIZE_METHOD = _TorchSerializedList


@contextlib.contextmanager
def set_default_dataset_from_list_serialize_method(new):
    """Context manager for using custom serialize function when creating DatasetFromList"""
    global _DEFAULT_DATASET_FROM_LIST_SERIALIZE_METHOD
    orig = _DEFAULT_DATASET_FROM_LIST_SERIALIZE_METHOD
    _DEFAULT_DATASET_FROM_LIST_SERIALIZE_METHOD = new
    yield
    _DEFAULT_DATASET_FROM_LIST_SERIALIZE_METHOD = orig


class DatasetFromList(data.Dataset):
    """Wrap a list to a torch Dataset. It produces elements of the list as data."""
    def __init__(self, lst: list, copy: bool = True, serialize: Union[bool, Callable] = True):
        """
        Args:
            lst (list): a list which contains elements to produce.
            copy (bool): whether to deepcopy the element when producing it,
                so that the result can be modified in place without affecting the
                source in the list.
            serialize (bool or callable): whether to serialize the stroage to other
                backend. If `True`, the default serialize method will be used, if given
                a callable, the callable will be used as serialize method.
        """
        self._lst = lst
        self._copy = copy
        
        if not isinstance(serialize, (bool, Callable)):
            raise TypeError(f"Unsupported type for argument `serailzie`: {serialize}")
        
        self._serialize = serialize is not False

        if self._serialize:
            serialize_method = (serialize if isinstance(serialize, Callable) else _DEFAULT_DATASET_FROM_LIST_SERIALIZE_METHOD)
            logger.info(f"Serializing the dataset using: {serialize_method}")
            self._lst = serialize_method(self._lst)

    def __len__(self):
        return len(self._lst)

    def __getitem__(self, idx):
        if self._copy and not self._serialize:
            return copy.deepcopy(self._lst[idx])
        else:
            return self._lst[idx]


class ToIterableDataset(data.IterableDataset):
    """Convert an old indices-based (also called map-style) dataset to an iterable-style dataset."""
    def __init__(self, dataset: data.Dataset, sampler: Sampler, shard_sampler: bool = True, shard_chunk_size: int = 1,):
        """Args:
            dataset: an old-style dataset with ``__getitem__``
            sampler: a cheap iterable that produces indices to be applied on ``dataset``.
            shard_sampler: whether to shard the sampler based on the current pytorch data loader
                worker id. When an IterableDataset is forked by pytorch's DataLoader into multiple
                workers, it is responsible for sharding its data based on worker id so that workers
                don't produce identical data.

                Most samplers (like our TrainingSampler) do not shard based on dataloader worker id
                and this argument should be set to True. But certain samplers may be already
                sharded, in that case this argument should be set to False.
            shard_chunk_size: when sharding the sampler, each worker will"""
        
        assert not isinstance(dataset, data.IterableDataset), dataset
        assert isinstance(sampler, Sampler), sampler
        self.dataset = dataset
        self.sampler = sampler
        self.shard_sampler = shard_sampler
        self.shard_chunk_size = shard_chunk_size

    def __iter__(self):
        if not self.shard_sampler:
            sampler = self.sampler
        else:
            # With map-style dataset, `DataLoader(dataset, sampler)` runs the
            # sampler in main process only. But `DataLoader(ToIterableDataset(dataset, sampler))`
            # will run sampler in every of the N worker. So we should only keep 1/N of the ids on
            # each worker. The assumption is that sampler is cheap to iterate so it's fine to
            # discard ids in workers.
            sampler = _shard_iterator_dataloader_worker(self.sampler, self.shard_chunk_size)
        for idx in sampler:
            yield self.dataset[idx]

    def __len__(self):
        return len(self.sampler)


# api: inps must be tensor [bs, c, h, w], targets must be numpy [bs, N, R][xyxycls]
def visualize_inputs_st(inps, targets):
    _COLORS = np.array([0.000, 0.447, 0.741,
                        0.850, 0.325, 0.098,
                        0.929, 0.694, 0.125,
                        0.494, 0.184, 0.556,
                        0.466, 0.674, 0.188,
                        0.301, 0.745, 0.933,
                        0.635, 0.078, 0.184,
                        0.300, 0.300, 0.300,
                        0.600, 0.600, 0.600,
                        1.000, 0.000, 0.000,
                        1.000, 0.500, 0.000,
                        0.749, 0.749, 0.000,
                        0.000, 1.000, 0.000,
                        0.000, 0.000, 1.000,
                        0.667, 0.000, 1.000,
                        0.333, 0.333, 0.000,
                        0.333, 0.667, 0.000,
                        0.333, 1.000, 0.000,]).astype(np.float32).reshape(-1, 3)
    
    for i in range(inps.shape[0]):
        # 处理图片
        temp_img = copy.deepcopy(inps[i])
        temp_img = np.ascontiguousarray(temp_img.permute(1, 2, 0).numpy())

        # 处理标签
        label_tensor = targets[i]

        for i in range(len(label_tensor)):
            box = label_tensor[i]
            x0 = int(box[0])
            y0 = int(box[1])
            x1 = int(box[2])
            y1 = int(box[3])
            color = (_COLORS[random.randint(0, 17)] * 255).astype(np.uint8).tolist()
            cv2.rectangle(temp_img, (x0, y0), (x1, y1), color, 2) # cv2 takes numpy
        
        # temp_img = cv2.cvtColor(temp_img, cv2.COLOR_BGR2RGB) # if readin by PIL.Image, then comment out this line
        cv2.imwrite('/home/mail/2023t3/t330201601/Detectron3/tbtest/mosaic/' + str(random.randint(1, 500)) + '_test.jpg', temp_img) # cv2.imwrite reqire [h, w, c]


# api: inps must be tensor [bs, c, h, w], targets must be numpy [bs, N, R]
def visualize_inputs_dp(inps, targets):
    _COLORS = np.array([0.000, 0.447, 0.741,
                        0.850, 0.325, 0.098,
                        0.929, 0.694, 0.125,
                        0.494, 0.184, 0.556,
                        0.466, 0.674, 0.188,
                        0.301, 0.745, 0.933,
                        0.635, 0.078, 0.184,
                        0.300, 0.300, 0.300,
                        0.600, 0.600, 0.600,
                        1.000, 0.000, 0.000,
                        1.000, 0.500, 0.000,
                        0.749, 0.749, 0.000,
                        0.000, 1.000, 0.000,
                        0.000, 0.000, 1.000,
                        0.667, 0.000, 1.000,
                        0.333, 0.333, 0.000,
                        0.333, 0.667, 0.000,
                        0.333, 1.000, 0.000,]).astype(np.float32).reshape(-1, 3)
    
    for i in range(inps.shape[0]):
        # 处理图片
        temp_img = copy.deepcopy(inps[i])
        temp_img = np.ascontiguousarray(temp_img.permute(1, 2, 0).numpy())

        # 处理标签
        label_tensor = targets[i]

        for i in range(len(label_tensor)):
            box = label_tensor[i]
            x0 = int(box[0])
            y0 = int(box[1])
            x1 = int(box[2])
            y1 = int(box[3])
            color = (_COLORS[random.randint(0, 17)] * 255).astype(np.uint8).tolist()
            cv2.rectangle(temp_img, (x0, y0), (x1, y1), color, 2) # cv2 takes numpy
        
        # temp_img = cv2.cvtColor(temp_img, cv2.COLOR_BGR2RGB) # if readin by PIL.Image, then comment out this line
        cv2.imwrite('D:/AICV-Detectron5/tbtest/datapool/' + str(random.randint(1, 500)) + '_test.jpg', temp_img) # cv2.imwrite reqire [h, w, c]


class AspectRatioGroupedDataset(data.IterableDataset):
    """Batch data that have similar aspect ratio together.
    In this implementation, images whose aspect ratio < (or >) 1 will
    be batched together.
    This improves training speed because the images then need less padding
    to form a batch.

    It assumes the underlying dataset produces dicts with "width" and "height" keys.
    It will then produce a list of original dicts with length = batch_size,
    all with similar aspect ratios."""
    

    def __init__(self, dataset, batch_size, data_aug):
        """
        Args:
            dataset: an iterable. Each element must be a dict with keys
                "width" and "height", which will be used to batch data.
            batch_size (int):
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self._buckets = [[] for _ in range(2)]
        # Hard-coded two aspect ratio groups: w > h and w < h.
        # Can add support for more aspect ratio groups, but doesn't seem useful
        
        # addition code: data aug selection
        self.data_aug = data_aug
        
        if self.data_aug == 'original':
            self.batch_size = self.batch_size * 1
            
        if self.data_aug == 'mosaic':
            self.batch_size = self.batch_size * 4
            self.preproc = TrainTransform(max_labels=1, flip_prob=0.0, hsv_prob=0.0)
            
        if self.data_aug == 'stitcher':
            self.batch_size = self.batch_size * 4
    
    def __iter__(self):
        for d in self.dataset:
            w, h = d["width"], d["height"]
            bucket_id = 0 if w > h else 1
            bucket = self._buckets[bucket_id]
            bucket.append(d)
            if len(bucket) == self.batch_size:
                # data is list, len(data) is bs, every element is dict
                # Be advice: images in data are not same size, [c, h, w]
                # images in data are torch.tensor, stitcher require numpy
                # Be advice: height & width in data not match img.shape
                data = bucket[:]
            
                if self.data_aug == 'stitcher':
                    # delete file_name & image_id
                    for i in range(len(data)):
                        del data[i]["file_name"]
                        del data[i]["image_id"]
                        del data[i]["height"]
                        del data[i]["width"]
            
                    # Stitcher input: img & lab are numpy inside list, img=[c, h, w], lab=[N, 5] [cls, xcen, ycen, w, h]
                    # Stitcher output: img & lab are tensor, img=[c, h, w], lab=[bs, 120, 5] [cls, xcen, ycen, w, h]
                    # stitcher_process input & output: data is list, len(data)=bs, data[i]=dict, dict.key()=image, instances
                    data = stitcher_process(data)
                
                if self.data_aug == 'mosaic':
                    # delete file_name & image_id
                    for i in range(len(data)):
                        del data[i]["file_name"]
                        del data[i]["image_id"]
                        del data[i]["height"]
                        del data[i]["width"]
                    
                    # Mosaic Augmentation from Yolov4
                    data = mosaic_process(data, self.preproc)
                    
                # visualize img & lab match or not
                # for i in range(len(data)):
                #     visualize_inputs_st(data[i]['image'].unsqueeze(0), data[i]['instances'].gt_boxes.tensor.unsqueeze(0).numpy())
            
                # Clear bucket first, because code after yield is not guaranteed to execute
                del bucket[:]
                yield data


def mosaic_process(data, preproc):
    # Preprocess data dict
    images_list, labels_list = [], []
    
    for i in range(len(data)):
        images_list.append(data[i]['image'].numpy()) # [c, h, w]
        gt_classes = data[i]['instances'].gt_classes
        gt_boxes = data[i]['instances'].gt_boxes.tensor
        labels_list.append(torch.cat((gt_boxes, gt_classes.unsqueeze(1)), dim=1).numpy())
    
    mosaic_img, mosaic_lab, data_mosaic = [], [], []
        
    # Put images & labels into Stitcher
    for i in range(int(len(images_list)/4)):
        mosaic_img.append([images_list[i*4], images_list[i*4+1], images_list[i*4+2], images_list[i*4+3]])
        mosaic_lab.append([labels_list[i*4], labels_list[i*4+1], labels_list[i*4+2], labels_list[i*4+3]])
    
    for i in range(len(mosaic_img)):
        imgs, annots = mosaic_augment(mosaic_img[i], mosaic_lab[i], preproc)
        imgs = torch.from_numpy(imgs) # .transpose(2, 0, 1)
        annots = xywh_to_xyxy(torch.from_numpy(annots))
        
        gt_classes = annots[..., 0]
        gt_boxes = annots[..., 1:5]
        
        # build up Boxes & instances with boxes & classes
        # have to be this way when building instances
        img_size = (int(imgs.shape[1]), int(imgs.shape[2]))
        instances = Instances(img_size)
        instances.gt_boxes = Boxes(gt_boxes)
        instances.gt_classes = gt_classes

        data_mosaic.append({'image':imgs, 'instances': instances})
    
    return data_mosaic


def mosaic_augment(image_list, label_list, preproc):
    # get maximum size, [c h w]=[3, 320, 498]
    max_size = tuple(max(s) for s in zip(*[img.shape for img in image_list]))
    input_h, input_w, mosaic_labels = max_size[1], max_size[2], []
    
    # yc, xc = s, s  # mosaic center x, y
    yc = int(random.uniform(0.5 * input_h, 1.5 * input_h))
    xc = int(random.uniform(0.5 * input_w, 1.5 * input_w))
    
    for i_mosaic, img in enumerate(image_list):
        img, _labels = img.transpose(1, 2, 0), label_list[i_mosaic] # img.shape=[h, w, c], label=[xyxycls]
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

    mosaic_img, mosaic_labels = random_affine(mosaic_img, mosaic_labels, target_size=(input_w, input_h), degrees=0.0, translate=0.0, scales=(0.1, 2), shear=0.0)
    # print(mosaic_img.shape, mosaic_labels) # img.shape=[h, w, c], lab.shape=[xyxycls]
    mosaic_img, mosaic_labels = preproc(mosaic_img, mosaic_labels, (input_h, input_w))
    # print(mosaic_img.shape, mosaic_labels) # img.shape=[c, h, w], lab.shape=[cls, xc, yc, w, h]
    
    # filter invalide labels
    mosaic_index = []
    for i in range(mosaic_labels.shape[0]):
        if mosaic_labels[i][3] != 0 and mosaic_labels[i][4] != 0:
            mosaic_index.append(i)
    if len(mosaic_index) != 0:
        mosaic_labels = mosaic_labels[mosaic_index]
    else:
        mosaic_labels = np.array([[80, 0.0, 0.0, 0.0, 0.0]])
    
    return mosaic_img, mosaic_labels


def get_mosaic_coordinate(mosaic_image, mosaic_index, xc, yc, w, h, input_h, input_w):
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


def stitcher_process(data):
    # Preprocess data dict
    images_list, labels_list = [], []
    
    for i in range(len(data)):
        images_list.append(data[i]['image'].numpy())
        
        gt_classes = data[i]['instances'].gt_classes
        gt_boxes = data[i]['instances'].gt_boxes.tensor
        annots = torch.cat((gt_classes.unsqueeze(1), gt_boxes), dim=1).numpy()
        
        # xyxy -> xcenycenwh
        for j in range(annots.shape[0]):
            scaled_xcen = (annots[j][1] + annots[j][3]) / 2
            scaled_ycen = (annots[j][2] + annots[j][4]) / 2
            scaled_w = annots[j][3] - annots[j][1]
            scaled_h = annots[j][4] - annots[j][2]
            annots[j][1] = scaled_xcen
            annots[j][2] = scaled_ycen
            annots[j][3] = scaled_w
            annots[j][4] = scaled_h
        
        labels_list.append(annots)
    
    # Put images & labels into Stitcher
    imgs, annots = to_image_list_synthesize_4([images_list, labels_list])
    
    # Postprocess data dict bs -> list
    # data_stitcher, annots = [], [i for i in annots]
    data_stitcher = []
    for i in range(imgs.shape[0]):
        # [cls, xcen, ycen, w, h] -> [cls, x1, y1, x2, y2]
        for j in range(annots[i].shape[0]):
            x1 = annots[i][j][1] - annots[i][j][3] / 2
            y1 = annots[i][j][2] - annots[i][j][4] / 2
            x2 = annots[i][j][1] + annots[i][j][3] / 2
            y2 = annots[i][j][2] + annots[i][j][4] / 2
            annots[i][j][1] = x1
            annots[i][j][2] = y1
            annots[i][j][3] = x2
            annots[i][j][4] = y2
        
        # divide annots into gt_classes & gt_boxes
        # gt_classes, gt_boxes = [], torch.randn(annots[i].shape[0], 4)
        # for j in range(annots[i].shape[0]):
        #     gt_classes.append(annots[i][j][0].numpy())
        # gt_classes = torch.from_numpy(np.stack(gt_classes))
        gt_classes = annots[i][..., 0]
        gt_boxes = annots[i][...,1:5]
        
        # build up Boxes & instances with boxes & classes
        # have to be this way when building instances
        img_size = (int(imgs[i].shape[1]), int(imgs[i].shape[2]))
        instances = Instances(img_size)
        instances.gt_boxes = Boxes(gt_boxes)
        instances.gt_classes = gt_classes
        
        data_stitcher.append({'image':imgs[i], 'instances': instances})
    
    return data_stitcher


def to_image_list_synthesize_4(transposed_info):
    tensors = transposed_info[0] # batch经过list(zip(*x))处理后的transposed_info是1个列表,其中有1个元组
    if isinstance(tensors, (tuple, list)): # 判断tensors是否属于tuple或list, tensors[i].shape=(3, 640, 640)
        targets = transposed_info[1] # targets[i].shape=(120, 5)
        # img_ids = transposed_info[2] # ((321, 500), (333, 500), (320, 499), (220, 331), (334, 499), (220, 331), (321, 500), (333, 500))

        # synthesize data:
        assert len(tensors) % 4 == 0, 'len(tensor) % 4 != 0, could not be synthesized ! uneven'
        max_size = tuple(max(s) for s in zip(*[img.shape for img in tensors])) # 取得最大图片的尺寸,(3, 320, 498) c h w
        
        batch_shape = (len(tensors)//4,) + max_size # 将两个元组拼接成1个元组
        syn_batched_imgs = torch.from_numpy(tensors[0]).new(*batch_shape).zero_() # syn_batched_imgs.shape = [1, 3, 320, 512]
        # 创造1个batch_shape类型的空张量,并且每处都赋予0值,tensors[0].new()创建1个无值的张量,啥张量后跟new都行
        # 但new()需要输入参数,并且输入的参数不能是列表,因此需要用*batch_shape方式来去除列表
        
        syn_imgs, syn_targets = [], []
        with torch.no_grad():
            for idx, pad_img in enumerate(syn_batched_imgs): # idx = 0, pad_img.shape = [3, 320, 512], 因为len(tensors)//4==1
                # currently suppose first w then h
                new_h, new_w = max_size[1]//2, max_size[2]//2 # 缩放后尺寸是最大宽高的一半

                # NOTE: interpolate api require first h then w ! interpolate()作用为缩放图片, [c, h, w]
                mode = 'nearest' # squeeze(0)去除第1维
                topLeftImg = torch.nn.functional.interpolate(torch.from_numpy(tensors[idx*4]).unsqueeze(0),size=(new_h, new_w),mode=mode).squeeze(0)
                topRightImg = torch.nn.functional.interpolate(torch.from_numpy(tensors[idx*4+1]).unsqueeze(0),size=(new_h, new_w),mode=mode).squeeze(0)
                bottomLeftImg = torch.nn.functional.interpolate(torch.from_numpy(tensors[idx*4+2]).unsqueeze(0),size=(new_h, new_w),mode=mode).squeeze(0)
                bottomRightImg = torch.nn.functional.interpolate(torch.from_numpy(tensors[idx*4+3]).unsqueeze(0),size=(new_h, new_w),mode=mode).squeeze(0)
                
                """print("pad_img: ", pad_img.shape)
                print("topLeftImg: ", tensors[idx*4].shape, topLeftImg.shape)
                print("topRightImg: ", tensors[idx*4+1].shape, topRightImg.shape)
                print("bottomLeftImg: ", tensors[idx*4+2].shape, bottomLeftImg.shape)
                print("bottomRightImg: ", tensors[idx*4+3].shape, bottomRightImg.shape)"""
                
                c = topLeftImg.shape[0] # 取得缩放后图片的通道数
                assert c == topRightImg.shape[0] and c == bottomLeftImg.shape[0] and c == bottomRightImg.shape[0] # 确定缩放后子图片通道数相等
                
                # 当pad_img的宽高不是new_w×2&new_h×2时拼贴会报错
                if topRightImg.shape[1] * 2 != pad_img.shape[1] or topRightImg.shape[2] * 2 != pad_img.shape[2]:
                    pad_img = torch.nn.functional.interpolate(pad_img.unsqueeze(0), size=(new_h * 2, new_w * 2), mode=mode).squeeze(0)
                
                # 将四张缩放后的子图片拼接成一张图片,画个图就很清晰了
                pad_img[:c, :topLeftImg.shape[1], :topLeftImg.shape[2]].copy_(topLeftImg)
                pad_img[:c, :topRightImg.shape[1], topLeftImg.shape[2]:].copy_(topRightImg)
                pad_img[:c, topLeftImg.shape[1]:, :bottomLeftImg.shape[2]].copy_(bottomLeftImg)
                pad_img[:c, topRightImg.shape[1]:, topLeftImg.shape[2]:].copy_(bottomRightImg)
                
                # cv2.imwrite("D:/AICV-YoloXReGPU/abc.jpg", np.transpose(pad_img.numpy(), (1, 2, 0)))
                
                # resize each of four sub-imgs into (new_h, new_w) scale
                # resize api require first w then h ! (120, 5) 120个[cls, x, y, w, h]
                topLeftBL = resize(torch.from_numpy(targets[idx*4]), (tensors[idx*4].shape[2], tensors[idx*4].shape[1]), (new_w, new_h))
                topRightBL = resize(torch.from_numpy(targets[idx*4+1]), (tensors[idx*4+1].shape[2], tensors[idx*4+1].shape[1]), (new_w, new_h))
                bottomLeftBL = resize(torch.from_numpy(targets[idx*4+2]), (tensors[idx*4+2].shape[2], tensors[idx*4+2].shape[1]), (new_w, new_h))
                bottomRightBL = resize(torch.from_numpy(targets[idx*4+3]), (tensors[idx*4+3].shape[2], tensors[idx*4+3].shape[1]), (new_w, new_h))
                
                # 计算四张图片上得到新目标所需的偏移值
                offsets = [torch.Tensor([0.0,0.0,0.0,0.0]), torch.Tensor([new_w,0.0,new_w,0.0]), torch.Tensor([0.0,new_h,0.0,new_h]), torch.Tensor([new_w,new_h,new_w,new_h])]
                
                # append offsets to box coordinates except for topLeftBL 调整GT框坐标到新位置
                topLeftBL = compute_tensor(topLeftBL, offsets[0])
                topRightBL = compute_tensor(topRightBL, offsets[1])
                bottomLeftBL = compute_tensor(bottomLeftBL, offsets[2])
                bottomRightBL = compute_tensor(bottomRightBL, offsets[3])
                
                # 从xyxy还原成xywh格式
                topLeftBL = xyxy_to_xywh(topLeftBL)
                topRightBL = xyxy_to_xywh(topRightBL)
                bottomLeftBL = xyxy_to_xywh(bottomLeftBL)
                bottomRightBL = xyxy_to_xywh(bottomRightBL)
                
                # 填充图片至new_w & new_h
                if pad_img.shape[1] < new_h: # input_size=[height, width], pad_img=[c, h, w]
                    dh = new_h - pad_img.shape[1]
                    dh /= 2
                    pad_top, pad_bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
                else:
                    pad_top, pad_bottom = 0, 0
                
                if pad_img.shape[2] < new_w: # input_size=[height, width], pad_img=[c, h, w]
                    dw = new_w - pad_img.shape[2]
                    dw /= 2
                    pad_left, pad_right = int(round(dw - 0.1)), int(round(dw + 0.1))
                else:
                    pad_left, pad_right = 0, 0
                
                # print(pad_img.shape) # [w, c, h] which is wrong, right one is [c, h, w]
                pad_img = cv2.copyMakeBorder(np.transpose(pad_img.numpy(), (1, 2, 0)), pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
                pad_img = torch.from_numpy(np.transpose(pad_img, (2, 0, 1)))
                
                # 保存图片到列表中,最后拼接成批量
                syn_imgs.append(pad_img.unsqueeze(0))
                
                # 根据填充过程相应地移动边框坐标
                topLeftBL = add_tensor(topLeftBL, pad_left, pad_top)
                topRightBL = add_tensor(topRightBL, pad_left, pad_top)
                bottomLeftBL = add_tensor(bottomLeftBL, pad_left, pad_top)
                bottomRightBL = add_tensor(bottomRightBL, pad_left, pad_top)
                
                
                """topLeft = xywh_to_xyxy(copy.deepcopy(topLeftBL))
                topRight = xywh_to_xyxy(copy.deepcopy(topRightBL))
                bottomLeft = xywh_to_xyxy(copy.deepcopy(bottomLeftBL))
                bottomRight = xywh_to_xyxy(copy.deepcopy(bottomRightBL))
                
                # 可视化拼贴图片的标签是否与目标匹配
                temp_img = np.transpose(copy.deepcopy(pad_img).numpy(), (1, 2, 0)).copy()
                # cv2.imwrite('/mnt/yoloxredstorig/synthesis/syn_img_' + str(random.randint(0, 100000)) + '.jpg', temp_img)
                label_tensor = torch.cat((topLeft,  topRight, bottomLeft, bottomRight), 0)
                _COLORS = np.array([0.000, 0.447, 0.741]).astype(np.float32).reshape(-1, 3)
                for i in range(len(label_tensor)):
                    box = label_tensor[i]
                    x0 = int(box[1])
                    y0 = int(box[2])
                    x1 = int(box[3])
                    y1 = int(box[4])
                    color = (_COLORS[0] * 255).astype(np.uint8).tolist()
                    cv2.rectangle(temp_img, (x0, y0), (x1, y1), color, 2)
                cv2.imwrite('/mnt/yoloxredstorig/synthesis/syn_img_' + str(random.randint(0, 100000)) + '.jpg', temp_img) # cv2.imwrite reqire [h, w, c]
                # print("already done!")
                
                topLeft = xyxy_to_xywh(copy.deepcopy(topLeftBL))
                topRight = xyxy_to_xywh(copy.deepcopy(topRightBL))
                bottomLeft = xyxy_to_xywh(copy.deepcopy(bottomLeftBL))
                bottomRight = xyxy_to_xywh(copy.deepcopy(bottomRightBL))"""
                
                # 添加0值行变为shape=(120, 5)
                syn_bbox = torch.cat((topLeftBL, topRightBL, bottomLeftBL, bottomRightBL), dim=0)
                # zero = torch.tensor([[0., 0., 0., 0., 0.]])
                # for i in range(120 - syn_bbox.shape[0]):
                #     syn_bbox = torch.cat((syn_bbox, zero),dim=0)
                # del zero
                # syn_targets.append(syn_bbox.unsqueeze(0))
                syn_targets.append(syn_bbox)
                
        
        # 检查ID数量是否也为4的倍数
        # assert len(img_ids)%4 == 0
        
        # 拼接合成目标与合成标签为batch张量
        syn_imgs = torch.cat(syn_imgs, dim=0)
        # syn_targets = torch.cat(syn_targets, dim=0)
        
        return syn_imgs, syn_targets
    else:
        raise TypeError("Unsupported type for to_image_list: {}".format(type(tensors)))


def resize(targets, sizeori, sizenew): # 输入xywh返回xyxy
    # 去掉标签张量中的0值行
    temp = []
    for i in range(targets.shape[0]):
        if targets[i][3]!=0 and targets[i][4]!=0:
            temp.append(targets[i])
    targets = torch.stack(temp, dim=0)
    del temp

    # 取得新宽高与旧宽高的比例元组
    ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(sizenew, sizeori))

    # 当宽高新旧比例不等时需要分别操作
    ratio_width, ratio_height = ratios
    for i in range(targets.shape[0]):
        xmin = targets[i][1] - targets[i][3]/2
        ymin = targets[i][2] - targets[i][4]/2
        xmax = targets[i][1] + targets[i][3]/2
        ymax = targets[i][2] + targets[i][4]/2
        scaled_xmin = xmin * ratio_width
        scaled_xmax = xmax * ratio_width
        scaled_ymin = ymin * ratio_height
        scaled_ymax = ymax * ratio_height
        targets[i][1] = scaled_xmin
        targets[i][2] = scaled_ymin
        targets[i][3] = scaled_xmax
        targets[i][4] = scaled_ymax
    
    return targets


def compute_tensor(tensor1, tensor2):
    for i in range(tensor1.shape[0]):
        tensor1[i][1:] = tensor1[i][1:] + tensor2
    
    return tensor1


def add_tensor(tensor1, pad_left, pad_top): # 专门用于为xcen&ycen添加填充量
    for i in range(tensor1.shape[0]):
        tensor1[i][1] = tensor1[i][1] + pad_left
        tensor1[i][2] = tensor1[i][2] + pad_top

    return tensor1


def xyxy_to_xywh(tensor): # 输入xyxy返回xywh
    for i in range(tensor.shape[0]):
        scaled_xcen = (tensor[i][3] + tensor[i][1]) / 2
        scaled_ycen = (tensor[i][4] + tensor[i][2]) / 2
        scaled_w = tensor[i][3] - tensor[i][1]
        scaled_h = tensor[i][4] - tensor[i][2]
        tensor[i][1] = scaled_xcen
        tensor[i][2] = scaled_ycen
        tensor[i][3] = scaled_w
        tensor[i][4] = scaled_h
    
    return tensor


def xywh_to_xyxy(tensor): # [cls, x1, y1, x2, y2]
    for i in range(tensor.shape[0]):
        x1 = tensor[i][1] - tensor[i][3] / 2
        y1 = tensor[i][2] - tensor[i][4] / 2
        x2 = tensor[i][1] + tensor[i][3] / 2
        y2 = tensor[i][2] + tensor[i][4] / 2
        tensor[i][1] = x1
        tensor[i][2] = y1
        tensor[i][3] = x2
        tensor[i][4] = y2
        
    return tensor
