#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.
import os
import sys
import time
import torch
import pickle
import tempfile
import numpy as np
import xml.etree.ElementTree as ET

from tqdm import tqdm
from loguru import logger
from collections import ChainMap
from yolox.utils import gather, is_main_process, postprocess, synchronize, time_synchronized


class VOCEvaluator:
    """
    VOC AP Evaluation class.
    """
    def __init__(self, dataloader, img_size, confthre, nmsthre, num_classes):
        """
        Args:
            dataloader (Dataloader): evaluate dataloader.
            img_size (int): image size after preprocess. images are resized
                to squares whose shape is (img_size, img_size).
            confthre (float): confidence threshold ranging from 0 to 1, which
                is defined in the config file.
            nmsthre (float): IoU threshold of non-max supression ranging from 0 to 1.
        """
        self.dataloader = dataloader
        self.img_size = img_size
        self.confthre = confthre
        self.nmsthre = nmsthre
        self.num_classes = num_classes
        self.num_images = len(dataloader.dataset)

    def evaluate(self, model, distributed=False, half=False, trt_file=None, decoder=None, test_size=None,):
        """
        VOC average precision (AP) Evaluation. Iterate inference on the test dataset
        and the results are evaluated by COCO API.

        NOTE: This function will change training mode to False, please save states if needed.

        Args:
            model : model to evaluate.

        Returns:
            ap50_95 (float) : COCO style AP of IoU=50:95
            ap50 (float) : VOC 2007 metric AP of IoU=50
            summary (sr): summary info of evaluation.
        """
        # TODO half to amp_test
        tensor_type = torch.cuda.HalfTensor if half else torch.cuda.FloatTensor
        # tensor_type = torch.FloatTensor
        model = model.eval()
        if half:
            model = model.half()
        ids = []
        data_list = {}
        progress_bar = tqdm if is_main_process() else iter

        inference_time = 0
        nms_time = 0
        n_samples = max(len(self.dataloader) - 1, 1)

        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones(1, 3, test_size[0], test_size[1]).cuda()
            model(x)
            model = model_trt

        for cur_iter, (imgs, _, info_imgs, ids) in enumerate(progress_bar(self.dataloader)):
            with torch.no_grad():
                imgs = imgs.type(tensor_type)
                
                # skip the the last iters since batchsize might be not enough for batch inference
                is_time_record = cur_iter < len(self.dataloader) - 1
                if is_time_record:
                    start = time.time()

                outputs = model(imgs) # outputs这里已经是元组了
                
                if decoder is not None:
                    outputs = decoder(outputs, dtype=outputs.type())
                
                if is_time_record:
                    infer_end = time_synchronized()
                    inference_time += infer_end - start
                
                outputs = postprocess(outputs, self.num_classes, self.confthre, self.nmsthre)
                
                if is_time_record:
                    nms_end = time_synchronized()
                    nms_time += nms_end - infer_end
            # print("outputs: ", outputs, info_imgs, ids) # [None] [tensor([321]), tensor([500])] tensor([4])
            data_list.update(self.convert_to_voc_format(outputs, info_imgs, ids))

        statistics = torch.cuda.FloatTensor([inference_time, nms_time, n_samples])
        if distributed:
            data_list = gather(data_list, dst=0)
            data_list = ChainMap(*data_list)
            torch.distributed.reduce(statistics, dst=0)
        # print("data_list: ", data_list) # {0:(None,None,None), 1:(None,None,None), 2:(None,None,None), 3:(None,None,None), 4:(None,None,None)}
        eval_results = self.evaluate_prediction(data_list, statistics)
        synchronize()
        return eval_results

    def convert_to_voc_format(self, outputs, info_imgs, ids):
        predictions = {}
        for (output, img_h, img_w, img_id) in zip(outputs, info_imgs[0], info_imgs[1], ids):
            
            if output is None:
                predictions[int(img_id)] = (None, None, None)
                continue
            
            output = output.cpu()

            bboxes = output[:, 0:4]

            # preprocessing: resize
            scale = min(self.img_size[0] / float(img_h), self.img_size[1] / float(img_w))
            bboxes /= scale

            cls = output[:, 6]
            scores = output[:, 4] * output[:, 5]

            predictions[int(img_id)] = (bboxes, cls, scores)
        return predictions

    def evaluate_prediction(self, data_dict, statistics):
        if not is_main_process():
            return 0, 0, None

        logger.info("Evaluate in main process...")

        inference_time = statistics[0].item()
        nms_time = statistics[1].item()
        n_samples = statistics[2].item()

        a_infer_time = 1000 * inference_time / (n_samples * self.dataloader.batch_size)
        a_nms_time = 1000 * nms_time / (n_samples * self.dataloader.batch_size)

        time_info = ", ".join(["Average {} time: {:.2f} ms".format(k, v) for k, v in zip(["forward", "NMS", "inference"],
                                                                                         [a_infer_time, a_nms_time, (a_infer_time + a_nms_time)])])

        info = time_info + "\n"

        all_boxes = [[[] for _ in range(self.num_images)] for _ in range(self.num_classes)]
        """
        print("all_boxes: ", type(all_boxes), all_boxes) # [[[], [], [], [], []],
                                                            [[], [], [], [], []]]
        """
        for img_num in range(self.num_images):
            bboxes, cls, scores = data_dict[img_num]
            # print("bboxes, cls, scores: ", bboxes, cls, scores) # None, None, None
            if bboxes is None:
                for j in range(self.num_classes):
                    all_boxes[j][img_num] = np.empty([0, 5], dtype=np.float32)
                continue
            for j in range(self.num_classes):
                mask_c = cls == j
                if sum(mask_c) == 0:
                    all_boxes[j][img_num] = np.empty([0, 5], dtype=np.float32)
                    continue

                c_dets = torch.cat((bboxes, scores.unsqueeze(1)), dim=1)
                all_boxes[j][img_num] = c_dets[mask_c].numpy()

            sys.stdout.write("im_eval: {:d}/{:d} \r".format(img_num + 1, self.num_images))
            sys.stdout.flush()

        with tempfile.TemporaryDirectory() as tempdir:
            mAP50, mAP70 = self.dataloader.dataset.evaluate_detections(all_boxes, tempdir)
            return mAP50, mAP70, info


def parse_rec(filename):
    """ Parse a PASCAL VOC xml file """
    # path = "D:\AICV-YoloXReCPU\datasets\VOCdevkit\VOC2007\Annotations"
    """
    path = "/mnt/datasets/VOCdevkit/VOC2007/Annotations"
    if (path in filename) == False:
        filename = path + os.sep + filename
    """
        
    tree = ET.parse(filename)
    
    objects = []
    for obj in tree.findall("object"):
        obj_struct = {}
        obj_struct["name"] = obj.find("name").text
        obj_struct["pose"] = obj.find("pose").text
        obj_struct["truncated"] = int(obj.find("truncated").text)
        obj_struct["difficult"] = int(obj.find("difficult").text)
        bbox = obj.find("bndbox")
        obj_struct["bbox"] = [
            int(bbox.find("xmin").text),
            int(bbox.find("ymin").text),
            int(bbox.find("xmax").text),
            int(bbox.find("ymax").text)]
        objects.append(obj_struct)

    return objects


def voc_ap(rec, prec, use_07_metric=False):
    """ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.0
        for t in np.arange(0.0, 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.0
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.0], rec, [1.0]))
        mpre = np.concatenate(([0.0], prec, [0.0]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def voc_eval(detpath, annopath, imagesetfile, classname, cachedir, ovthresh=0.5, use_07_metric=False,):
    # first load gt
    if not os.path.isdir(cachedir):
        os.mkdir(cachedir)
    cachefile = os.path.join(cachedir, "annots.pkl")
    
    # read list of images
    with open(imagesetfile, "r") as f:
        lines = f.readlines()
    imagenames = [x.strip() for x in lines]

    if not os.path.isfile(cachefile):
        # load annots
        recs = {}
        for i, imagename in enumerate(imagenames):
            print("annopath.format(imagename)", annopath.format(imagename))
            recs[imagename] = parse_rec(annopath.format(imagename))
            if i % 100 == 0:
                print("Reading annotation for {:d}/{:d}".format(i + 1, len(imagenames)))
        # save
        print("Saving cached annotations to {:s}".format(cachefile))
        with open(cachefile, "wb") as f:
            pickle.dump(recs, f)
    else:
        # load
        with open(cachefile, "rb") as f:
            recs = pickle.load(f)
    
    # extract gt objects for this class
    class_recs = {}
    npos = 0
    for imagename in imagenames:
        # print("recs: ", recs)
        R = [obj for obj in recs[imagename] if obj["name"] == classname]
        bbox = np.array([x["bbox"] for x in R])
        difficult = np.array([x["difficult"] for x in R]).astype(np.bool)
        det = [False] * len(R)
        npos = npos + sum(~difficult)
        class_recs[imagename] = {"bbox": bbox, "difficult": difficult, "det": det}
    
    # read dets
    detfile = detpath.format(classname)

    with open(detfile, "r") as f:
        lines = f.readlines()
    
    if len(lines) == 0:
        return 0, 0, 0
    
    splitlines = [x.strip().split(" ") for x in lines]
    image_ids = [x[0] for x in splitlines]
    confidence = np.array([float(x[1]) for x in splitlines])
    BB = np.array([[float(z) for z in x[2:]] for x in splitlines])

    # sort by confidence
    sorted_ind = np.argsort(-confidence)
    BB = BB[sorted_ind, :]
    image_ids = [image_ids[x] for x in sorted_ind]

    # go down dets and mark TPs and FPs
    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    for d in range(nd):
        R = class_recs[image_ids[d]]
        bb = BB[d, :].astype(float)
        ovmax = -np.inf
        BBGT = R["bbox"].astype(float)

        if BBGT.size > 0:
            # compute overlaps
            # intersection
            ixmin = np.maximum(BBGT[:, 0], bb[0])
            iymin = np.maximum(BBGT[:, 1], bb[1])
            ixmax = np.minimum(BBGT[:, 2], bb[2])
            iymax = np.minimum(BBGT[:, 3], bb[3])
            iw = np.maximum(ixmax - ixmin + 1.0, 0.0)
            ih = np.maximum(iymax - iymin + 1.0, 0.0)
            inters = iw * ih

            # union
            uni = ((bb[2] - bb[0] + 1.0) * (bb[3] - bb[1] + 1.0)
                   + (BBGT[:, 2] - BBGT[:, 0] + 1.0) * (BBGT[:, 3] - BBGT[:, 1] + 1.0)
                   - inters)

            overlaps = inters / uni
            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)

        if ovmax > ovthresh:
            if not R["difficult"][jmax]:
                if not R["det"][jmax]:
                    tp[d] = 1.0
                    R["det"][jmax] = 1
                else:
                    fp[d] = 1.0
        else:
            fp[d] = 1.0

    # compute precision recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec, use_07_metric)

    return rec, prec, ap