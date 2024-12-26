#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.
"""
Don't forget to change input info:
def make_parser() : Weight path
Class Exp() : Class number | Dataset path
"""

# import io
import os
import cv2
import sys
import ast
import math
# import json
import time
import random
import pickle
import inspect
import tempfile
import argparse
import warnings
import functools
import importlib
# import itertools
import subprocess
# import contextlib
import torchvision
import numpy as np
import xml.etree.ElementTree as ET

from tqdm import tqdm
from thop import profile
from loguru import logger
from copy import deepcopy
from functools import wraps
# from tabulate import tabulate
from datetime import timedelta
from collections import ChainMap

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import torch.backends.cudnn as cudnn
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.dataset import Dataset as torchDataset

_LOCAL_PROCESS_GROUP = None
DEFAULT_TIMEOUT = timedelta(minutes=30)


VOC_CLASSES = ("aeroplane", "bicycle", "bird", "boat", "bottle",
               "bus", "car", "cat", "chair", "cow", "diningtable", 
               "dog", "horse", "motorbike", "person", "pottedplant", 
               "sheep", "sofa", "train", "tvmonitor")


def compute_margin(logits, labels): # logits.shape=[obj_num, total_cls_num]
    # 初始化要用到的列表
    correct_logits, max_wrong_logits, margin = [], [], []
    
    for i in range(logits.shape[0]):
        # 提取正确类别的logits
        correct_logits.append(logits[i][int(labels[i])].item())

        # 将当前目标的logits从tensor转换为list方便操作
        current_obj_logits = logits[i].numpy().tolist()

        # 移除正确类别的logits方便提取最大错误类别logits
        current_obj_logits.pop(int(labels[i]))

        # 提取最大错误类别的logits
        max_wrong_logits.append(max(current_obj_logits))

    # 计算每个目标的margin, 负数即分类错误, 正数即分类正确
    for i in range(len(correct_logits)):
        margin.append(correct_logits[i] - max_wrong_logits[i])
    
    # 计算当前批次的平均margin
    if len(margin) != 0:
        margin = sum(margin) / len(margin)
    else:
        margin = 0

    return margin


class YOLOXHead(nn.Module):
    def __init__(self, num_classes,width=1.0, strides=[8, 16, 32], in_channels=[256, 512, 1024], act="silu", depthwise=False,):
        """
        Args:
            act (str): activation type of conv. Defalut value: "silu".
            depthwise (bool): whether apply depthwise conv in conv branch. Defalut value: False.
        """
        super().__init__()

        self.n_anchors = 1
        self.num_classes = num_classes
        self.decode_in_inference = True  # for deploy, set to False

        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        self.obj_preds = nn.ModuleList()
        self.stems = nn.ModuleList()
        Conv = DWConv if depthwise else BaseConv

        for i in range(len(in_channels)):
            self.stems.append(BaseConv(in_channels=int(in_channels[i] * width),out_channels=int(256 * width),ksize=1,stride=1,act=act))
            self.cls_convs.append(nn.Sequential(*[Conv(in_channels=int(256 * width),out_channels=int(256 * width),ksize=3,stride=1,act=act),
                                                  Conv(in_channels=int(256 * width),out_channels=int(256 * width),ksize=3,stride=1,act=act)]))
            self.reg_convs.append(nn.Sequential(*[Conv(in_channels=int(256 * width),out_channels=int(256 * width),ksize=3,stride=1,act=act),
                                                  Conv(in_channels=int(256 * width),out_channels=int(256 * width),ksize=3,stride=1,act=act)]))
            self.cls_preds.append(nn.Conv2d(in_channels=int(256 * width),out_channels=self.n_anchors*self.num_classes,kernel_size=1,stride=1,padding=0))
            self.reg_preds.append(nn.Conv2d(in_channels=int(256 * width),out_channels=4,kernel_size=1,stride=1,padding=0))
            self.obj_preds.append(nn.Conv2d(in_channels=int(256 * width),out_channels=self.n_anchors * 1,kernel_size=1,stride=1,padding=0))

        self.use_l1 = False
        self.l1_loss = nn.L1Loss(reduction="none")
        self.bcewithlog_loss = nn.BCEWithLogitsLoss(reduction="none")
        self.iou_loss = IOUloss(reduction="none")
        self.strides = strides
        self.grids = [torch.zeros(1)] * len(in_channels)

    def initialize_biases(self, prior_prob): # 初始化网络内参数,本模块并没调用此函数
        for conv in self.cls_preds:
            b = conv.bias.view(self.n_anchors, -1)
            b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

        for conv in self.obj_preds:
            b = conv.bias.view(self.n_anchors, -1)
            b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def forward(self, xin, labels=None, imgs=None):
        outputs = []
        outputs_train = []
        origin_preds = []
        x_shifts = []
        y_shifts = []
        expanded_strides = []

        for k, (cls_conv, reg_conv, stride_this_level, x) in enumerate(zip(self.cls_convs, self.reg_convs, self.strides, xin)):
            x = self.stems[k](x)
            cls_x = x
            reg_x = x

            cls_feat = cls_conv(cls_x)
            cls_output = self.cls_preds[k](cls_feat)

            reg_feat = reg_conv(reg_x)
            reg_output = self.reg_preds[k](reg_feat)
            obj_output = self.obj_preds[k](reg_feat)

            if self.training:
                output = torch.cat([reg_output, obj_output, cls_output], 1)
                output, grid = self.get_output_and_grid(output, k, stride_this_level, xin[0].type())
                x_shifts.append(grid[:, :, 0])
                y_shifts.append(grid[:, :, 1])
                expanded_strides.append(torch.zeros(1, grid.shape[1]).fill_(stride_this_level).type_as(xin[0]))
                if self.use_l1:
                    batch_size = reg_output.shape[0]
                    hsize, wsize = reg_output.shape[-2:]
                    reg_output = reg_output.view(batch_size, self.n_anchors, 4, hsize, wsize)
                    reg_output = reg_output.permute(0, 1, 3, 4, 2).reshape(batch_size, -1, 4)
                    origin_preds.append(reg_output.clone())

            else:
                def training_process(cls_output, reg_output, obj_output, stride_this_level, x_shifts, y_shifts, expanded_strides, xin, origin_preds):
                    output = torch.cat([reg_output, obj_output, cls_output], 1)
                    output, grid = self.get_output_and_grid(output, k, stride_this_level, xin[0].type())
                    x_shifts.append(grid[:, :, 0])
                    y_shifts.append(grid[:, :, 1])
                    expanded_strides.append(torch.zeros(1, grid.shape[1]).fill_(stride_this_level).type_as(xin[0]))
                    if self.use_l1:
                        batch_size = reg_output.shape[0]
                        hsize, wsize = reg_output.shape[-2:]
                        reg_output = reg_output.view(batch_size, self.n_anchors, 4, hsize, wsize)
                        reg_output = reg_output.permute(0, 1, 3, 4, 2).reshape(batch_size, -1, 4)
                        origin_preds.append(reg_output.clone())
                    return x_shifts, y_shifts, expanded_strides, output
                
                x_shifts, y_shifts, expanded_strides, output_train = training_process(cls_output, reg_output, \
                obj_output, stride_this_level, x_shifts, y_shifts, expanded_strides, xin, origin_preds)
                
                output = torch.cat([reg_output, obj_output.sigmoid(), cls_output.sigmoid()], 1)

            outputs.append(output)
            outputs_train.append(output_train)

        if self.training:
            return self.get_losses(imgs, x_shifts, y_shifts, expanded_strides, labels, torch.cat(outputs, 1), origin_preds, dtype=xin[0].dtype)
        else:
            # addition code for acquiring margin
            margin = self.get_losses(imgs, x_shifts, y_shifts, expanded_strides, labels, torch.cat(outputs_train, 1), origin_preds, dtype=xin[0].dtype)

            self.hw = [x.shape[-2:] for x in outputs]
            outputs = torch.cat([x.flatten(start_dim=2) for x in outputs], dim=2).permute(0, 2, 1) # outputs.shape = [batch, n_anchors_all, 85]
            if self.decode_in_inference:
                return self.decode_outputs(outputs, dtype=xin[0].type()), margin
            else:
                return outputs, margin

    def get_output_and_grid(self, output, k, stride, dtype):
        grid = self.grids[k]

        batch_size = output.shape[0]
        n_ch = 5 + self.num_classes
        hsize, wsize = output.shape[-2:]
        if grid.shape[2:4] != output.shape[2:4]:
            yv, xv = torch.meshgrid([torch.arange(hsize), torch.arange(wsize)], indexing='ij')
            grid = torch.stack((xv, yv), 2).view(1, 1, hsize, wsize, 2).type(dtype)
            self.grids[k] = grid

        output = output.view(batch_size, self.n_anchors, n_ch, hsize, wsize)
        output = output.permute(0, 1, 3, 4, 2).reshape(batch_size, self.n_anchors * hsize * wsize, -1)
        grid = grid.view(1, -1, 2)
        output[..., :2] = (output[..., :2] + grid) * stride
        output[..., 2:4] = torch.exp(output[..., 2:4]) * stride
        return output, grid

    def decode_outputs(self, outputs, dtype):
        grids = []
        strides = []
        for (hsize, wsize), stride in zip(self.hw, self.strides):
            yv, xv = torch.meshgrid([torch.arange(hsize), torch.arange(wsize)], indexing='ij')
            grid = torch.stack((xv, yv), 2).view(1, -1, 2)
            grids.append(grid)
            shape = grid.shape[:2]
            strides.append(torch.full((*shape, 1), stride))

        grids = torch.cat(grids, dim=1).type(dtype)
        strides = torch.cat(strides, dim=1).type(dtype)

        outputs[..., :2] = (outputs[..., :2] + grids) * strides
        outputs[..., 2:4] = torch.exp(outputs[..., 2:4]) * strides
        return outputs

    def get_losses(self,imgs,x_shifts,y_shifts,expanded_strides,labels,outputs,origin_preds,dtype):
        bbox_preds = outputs[:, :, :4]  # [batch, n_anchors_all, 4]
        obj_preds = outputs[:, :, 4].unsqueeze(-1)  # [batch, n_anchors_all, 1]
        cls_preds = outputs[:, :, 5:]  # [batch, n_anchors_all, n_cls]

        # calculate targets
        nlabel = (labels.sum(dim=2) > 0).sum(dim=1)  # number of objects

        total_num_anchors = outputs.shape[1]
        x_shifts = torch.cat(x_shifts, 1)  # [1, n_anchors_all]
        y_shifts = torch.cat(y_shifts, 1)  # [1, n_anchors_all]
        expanded_strides = torch.cat(expanded_strides, 1)
        if self.use_l1:
            origin_preds = torch.cat(origin_preds, 1)

        cls_targets = []
        reg_targets = []
        l1_targets = []
        obj_targets = []
        fg_masks = []

        num_fg = 0.0
        num_gts = 0.0
        gt_class_batch = []

        for batch_idx in range(outputs.shape[0]):
            num_gt = int(nlabel[batch_idx])
            num_gts += num_gt
            if num_gt == 0:
                cls_target = outputs.new_zeros((0, self.num_classes))
                reg_target = outputs.new_zeros((0, 4))
                l1_target = outputs.new_zeros((0, 4))
                obj_target = outputs.new_zeros((total_num_anchors, 1))
                fg_mask = outputs.new_zeros(total_num_anchors).bool()
            else:
                gt_bboxes_per_image = labels[batch_idx, :num_gt, 1:5]
                gt_classes = labels[batch_idx, :num_gt, 0]
                bboxes_preds_per_image = bbox_preds[batch_idx]

                try:
                    (gt_matched_classes,fg_mask,pred_ious_this_matching,matched_gt_inds,num_fg_img,) = self.get_assignments(  # noqa
                        batch_idx, num_gt, total_num_anchors, gt_bboxes_per_image, gt_classes, bboxes_preds_per_image, expanded_strides,
                        x_shifts, y_shifts, cls_preds, bbox_preds, obj_preds, labels, imgs)
                except RuntimeError:
                    logger.error("OOM RuntimeError is raised due to the huge memory cost during label assignment. \
                                 CPU mode is applied in this batch. If you want to avoid this issue, \
                                     try to reduce the batch size or image size.")
                    torch.cuda.empty_cache()
                    
                    (gt_matched_classes, fg_mask, pred_ious_this_matching, matched_gt_inds, num_fg_img,) = self.get_assignments(
                        batch_idx,num_gt,total_num_anchors,gt_bboxes_per_image,gt_classes,bboxes_preds_per_image,expanded_strides,
                        x_shifts,y_shifts,cls_preds,bbox_preds,obj_preds,labels,imgs,"cpu") # noqa

                torch.cuda.empty_cache()
                num_fg += num_fg_img

                cls_target = F.one_hot(gt_matched_classes.to(torch.int64), self.num_classes) * pred_ious_this_matching.unsqueeze(-1)
                gt_class_batch.extend(list(gt_matched_classes.cpu().numpy()))
                obj_target = fg_mask.unsqueeze(-1)
                reg_target = gt_bboxes_per_image[matched_gt_inds]
                if self.use_l1:
                    l1_target = self.get_l1_target(outputs.new_zeros((num_fg_img, 4)),
                                                   gt_bboxes_per_image[matched_gt_inds],
                                                   expanded_strides[0][fg_mask],
                                                   x_shifts=x_shifts[0][fg_mask],
                                                   y_shifts=y_shifts[0][fg_mask])

            cls_targets.append(cls_target)
            reg_targets.append(reg_target)
            obj_targets.append(obj_target.to(dtype))
            fg_masks.append(fg_mask)
            if self.use_l1:
                l1_targets.append(l1_target)

        cls_targets = torch.cat(cls_targets, 0)
        reg_targets = torch.cat(reg_targets, 0)
        obj_targets = torch.cat(obj_targets, 0)
        fg_masks = torch.cat(fg_masks, 0)
        if self.use_l1:
            l1_targets = torch.cat(l1_targets, 0)
        
        # 取得GT中能与Pred匹配的小目标
        gt_width = reg_targets[:, 2]
        gt_height = reg_targets[:, 3]
        gt_areas = gt_width * gt_height
        small_index = gt_areas <= 1024.0
        medium_index = ~(gt_areas <= 1024.0) & (gt_areas <= 9216.0)
        large_index = ~(gt_areas <= 9216.0)

        num_fg = max(num_fg, 1)
        loss_iou = (self.iou_loss(bbox_preds.view(-1, 4)[fg_masks], reg_targets)).sum() / num_fg
        loss_obj = (self.bcewithlog_loss(obj_preds.view(-1, 1), obj_targets)).sum() / num_fg
        # loss_cls = (self.bcewithlog_loss(cls_preds.view(-1, self.num_classes)[fg_masks], cls_targets)).sum() / num_fg

        cls_preds_filtered = cls_preds.view(-1, self.num_classes)[fg_masks] # cls_preds_filtered就是logits
        loss_cls = self.bcewithlog_loss(cls_preds_filtered, cls_targets)
        loss_cls_small, loss_cls_medium, loss_cls_large = loss_cls[small_index].sum(), loss_cls[medium_index].sum(), loss_cls[large_index].sum()

        # 计算当前批次每个尺度目标的平均margin
        small_logits = cls_preds_filtered[small_index].detach().cpu()
        small_labels = torch.tensor(gt_class_batch).to(cls_preds.device)[small_index].detach().cpu()
        margin_small_avg = compute_margin(small_logits, small_labels)

        midle_logits = cls_preds_filtered[medium_index].detach().cpu()
        midle_labels = torch.tensor(gt_class_batch).to(cls_preds.device)[medium_index].detach().cpu()
        margin_midle_avg = compute_margin(midle_logits, midle_labels)
        
        large_logits = cls_preds_filtered[large_index].detach().cpu()
        large_labels = torch.tensor(gt_class_batch).to(cls_preds.device)[large_index].detach().cpu()
        margin_large_avg = compute_margin(large_logits, large_labels)

        total_logits = cls_preds_filtered.detach().cpu()
        total_labels = torch.tensor(gt_class_batch).to(cls_preds.device).detach().cpu()
        margin_total_avg = compute_margin(total_logits, total_labels)

        loss_cls = loss_cls.sum() / num_fg

        if self.use_l1:
            loss_l1 = (self.l1_loss(origin_preds.view(-1, 4)[fg_masks], l1_targets)).sum() / num_fg
        else:
            loss_l1 = 0.0

        reg_weight = 5.0
        loss = reg_weight * loss_iou + loss_obj + loss_cls + loss_l1

        # return (loss,reg_weight * loss_iou,loss_obj,loss_cls,loss_l1,num_fg / max(num_gts, 1))
        return [margin_small_avg, margin_midle_avg, margin_large_avg, margin_total_avg]

    def get_l1_target(self, l1_target, gt, stride, x_shifts, y_shifts, eps=1e-8):
        l1_target[:, 0] = gt[:, 0] / stride - x_shifts
        l1_target[:, 1] = gt[:, 1] / stride - y_shifts
        l1_target[:, 2] = torch.log(gt[:, 2] / stride + eps)
        l1_target[:, 3] = torch.log(gt[:, 3] / stride + eps)
        return l1_target

    @torch.no_grad()
    def get_assignments(self,batch_idx,num_gt,total_num_anchors,gt_bboxes_per_image,gt_classes,bboxes_preds_per_image,expanded_strides,
        x_shifts,y_shifts,cls_preds,bbox_preds,obj_preds,labels,imgs,mode="gpu"):

        if mode == "cpu":
            print("------------CPU Mode for This Batch-------------")
            gt_bboxes_per_image = gt_bboxes_per_image.cpu().float()
            bboxes_preds_per_image = bboxes_preds_per_image.cpu().float()
            gt_classes = gt_classes.cpu().float()
            expanded_strides = expanded_strides.cpu().float()
            x_shifts = x_shifts.cpu()
            y_shifts = y_shifts.cpu()

        fg_mask, is_in_boxes_and_center = self.get_in_boxes_info(gt_bboxes_per_image,expanded_strides,x_shifts,y_shifts,total_num_anchors,num_gt)

        bboxes_preds_per_image = bboxes_preds_per_image[fg_mask]
        cls_preds_ = cls_preds[batch_idx][fg_mask]
        obj_preds_ = obj_preds[batch_idx][fg_mask]
        num_in_boxes_anchor = bboxes_preds_per_image.shape[0]

        if mode == "cpu":
            gt_bboxes_per_image = gt_bboxes_per_image.cpu()
            bboxes_preds_per_image = bboxes_preds_per_image.cpu()

        pair_wise_ious = bboxes_iou(gt_bboxes_per_image, bboxes_preds_per_image, False)

        gt_cls_per_image = (F.one_hot(gt_classes.to(torch.int64), self.num_classes).float().unsqueeze(1).repeat(1, num_in_boxes_anchor, 1))
        pair_wise_ious_loss = -torch.log(pair_wise_ious + 1e-8)

        if mode == "cpu":
            cls_preds_, obj_preds_ = cls_preds_.cpu(), obj_preds_.cpu()

        with torch.cuda.amp.autocast(enabled=False):
            cls_preds_ = (cls_preds_.float().unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_()
                          * obj_preds_.float().unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_())
            pair_wise_cls_loss = F.binary_cross_entropy(cls_preds_.sqrt_(), gt_cls_per_image, reduction="none").sum(-1)
            
        del cls_preds_

        cost = (pair_wise_cls_loss + 3.0 * pair_wise_ious_loss + 100000.0 * (~is_in_boxes_and_center))

        (num_fg,gt_matched_classes,pred_ious_this_matching,matched_gt_inds) = self.dynamic_k_matching(cost,pair_wise_ious,gt_classes,num_gt,fg_mask)
        
        del pair_wise_cls_loss, cost, pair_wise_ious, pair_wise_ious_loss

        if mode == "cpu":
            gt_matched_classes = gt_matched_classes.cuda()
            fg_mask = fg_mask.cuda()
            pred_ious_this_matching = pred_ious_this_matching.cuda()
            matched_gt_inds = matched_gt_inds.cuda()

        return (gt_matched_classes, fg_mask, pred_ious_this_matching, matched_gt_inds, num_fg)

    def get_in_boxes_info(self, gt_bboxes_per_image, expanded_strides, x_shifts,y_shifts, total_num_anchors, num_gt):
        expanded_strides_per_image = expanded_strides[0]
        x_shifts_per_image = x_shifts[0] * expanded_strides_per_image
        y_shifts_per_image = y_shifts[0] * expanded_strides_per_image
        x_centers_per_image = ((x_shifts_per_image + 0.5 * expanded_strides_per_image)
                               .unsqueeze(0).repeat(num_gt, 1))  # [n_anchor] -> [n_gt, n_anchor]
        y_centers_per_image = ((y_shifts_per_image + 0.5 * expanded_strides_per_image)
                               .unsqueeze(0).repeat(num_gt, 1))

        gt_bboxes_per_image_l = ((gt_bboxes_per_image[:, 0] - 0.5 * gt_bboxes_per_image[:, 2]).unsqueeze(1).repeat(1, total_num_anchors))
        gt_bboxes_per_image_r = ((gt_bboxes_per_image[:, 0] + 0.5 * gt_bboxes_per_image[:, 2]).unsqueeze(1).repeat(1, total_num_anchors))
        gt_bboxes_per_image_t = ((gt_bboxes_per_image[:, 1] - 0.5 * gt_bboxes_per_image[:, 3]).unsqueeze(1).repeat(1, total_num_anchors))
        gt_bboxes_per_image_b = ((gt_bboxes_per_image[:, 1] + 0.5 * gt_bboxes_per_image[:, 3]).unsqueeze(1).repeat(1, total_num_anchors))

        b_l = x_centers_per_image - gt_bboxes_per_image_l
        b_r = gt_bboxes_per_image_r - x_centers_per_image
        b_t = y_centers_per_image - gt_bboxes_per_image_t
        b_b = gt_bboxes_per_image_b - y_centers_per_image
        bbox_deltas = torch.stack([b_l, b_t, b_r, b_b], 2)

        is_in_boxes = bbox_deltas.min(dim=-1).values > 0.0
        is_in_boxes_all = is_in_boxes.sum(dim=0) > 0
        
        # in fixed center
        center_radius = 2.5

        gt_bboxes_per_image_l = (gt_bboxes_per_image[:, 0]).unsqueeze(1).repeat(1, total_num_anchors) - center_radius * expanded_strides_per_image.unsqueeze(0)
        gt_bboxes_per_image_r = (gt_bboxes_per_image[:, 0]).unsqueeze(1).repeat(1, total_num_anchors) + center_radius * expanded_strides_per_image.unsqueeze(0)
        gt_bboxes_per_image_t = (gt_bboxes_per_image[:, 1]).unsqueeze(1).repeat(1, total_num_anchors) - center_radius * expanded_strides_per_image.unsqueeze(0)
        gt_bboxes_per_image_b = (gt_bboxes_per_image[:, 1]).unsqueeze(1).repeat(1, total_num_anchors) + center_radius * expanded_strides_per_image.unsqueeze(0)

        c_l = x_centers_per_image - gt_bboxes_per_image_l
        c_r = gt_bboxes_per_image_r - x_centers_per_image
        c_t = y_centers_per_image - gt_bboxes_per_image_t
        c_b = gt_bboxes_per_image_b - y_centers_per_image
        
        center_deltas = torch.stack([c_l, c_t, c_r, c_b], 2)
        
        is_in_centers = center_deltas.min(dim=-1).values > 0.0
        is_in_centers_all = is_in_centers.sum(dim=0) > 0

        # in boxes and in centers
        is_in_boxes_anchor = is_in_boxes_all | is_in_centers_all

        is_in_boxes_and_center = (is_in_boxes[:, is_in_boxes_anchor] & is_in_centers[:, is_in_boxes_anchor])
        
        return is_in_boxes_anchor, is_in_boxes_and_center

    def dynamic_k_matching(self, cost, pair_wise_ious, gt_classes, num_gt, fg_mask):
        # Dynamic K
        # ---------------------------------------------------------------
        matching_matrix = torch.zeros_like(cost, dtype=torch.uint8)

        ious_in_boxes_matrix = pair_wise_ious
        n_candidate_k = min(10, ious_in_boxes_matrix.size(1))
        topk_ious, _ = torch.topk(ious_in_boxes_matrix, n_candidate_k, dim=1)
        dynamic_ks = torch.clamp(topk_ious.sum(1).int(), min=1)
        dynamic_ks = dynamic_ks.tolist()
        
        for gt_idx in range(num_gt):
            _, pos_idx = torch.topk(cost[gt_idx], k=dynamic_ks[gt_idx], largest=False)
            matching_matrix[gt_idx][pos_idx] = 1

        del topk_ious, dynamic_ks, pos_idx

        anchor_matching_gt = matching_matrix.sum(0)
        
        if (anchor_matching_gt > 1).sum() > 0:
            _, cost_argmin = torch.min(cost[:, anchor_matching_gt > 1], dim=0)
            matching_matrix[:, anchor_matching_gt > 1] *= 0
            matching_matrix[cost_argmin, anchor_matching_gt > 1] = 1
            
        fg_mask_inboxes = matching_matrix.sum(0) > 0
        
        num_fg = fg_mask_inboxes.sum().item()

        fg_mask[fg_mask.clone()] = fg_mask_inboxes

        matched_gt_inds = matching_matrix[:, fg_mask_inboxes].argmax(0)
        
        gt_matched_classes = gt_classes[matched_gt_inds]

        pred_ious_this_matching = (matching_matrix * pair_wise_ious).sum(0)[fg_mask_inboxes]
        
        return num_fg, gt_matched_classes, pred_ious_this_matching, matched_gt_inds
    

def bboxes_iou(bboxes_a, bboxes_b, xyxy=True):
    if bboxes_a.shape[1] != 4 or bboxes_b.shape[1] != 4:
        raise IndexError

    if xyxy:
        tl = torch.max(bboxes_a[:, None, :2], bboxes_b[:, :2])
        br = torch.min(bboxes_a[:, None, 2:], bboxes_b[:, 2:])
        area_a = torch.prod(bboxes_a[:, 2:] - bboxes_a[:, :2], 1)
        area_b = torch.prod(bboxes_b[:, 2:] - bboxes_b[:, :2], 1)
    else:
        tl = torch.max((bboxes_a[:, None, :2] - bboxes_a[:, None, 2:] / 2),(bboxes_b[:, :2] - bboxes_b[:, 2:] / 2))
        br = torch.min((bboxes_a[:, None, :2] + bboxes_a[:, None, 2:] / 2),(bboxes_b[:, :2] + bboxes_b[:, 2:] / 2))

        area_a = torch.prod(bboxes_a[:, 2:], 1)
        area_b = torch.prod(bboxes_b[:, 2:], 1)
    en = (tl < br).type(tl.type()).prod(dim=2)
    area_i = torch.prod(br - tl, 2) * en  # * ((tl < br).all())
    return area_i / (area_a[:, None] + area_b - area_i)


class IOUloss(nn.Module):
    def __init__(self, reduction="none", loss_type="iou"):
        super(IOUloss, self).__init__()
        self.reduction = reduction
        self.loss_type = loss_type

    def forward(self, pred, target):
        assert pred.shape[0] == target.shape[0]

        pred = pred.view(-1, 4)
        target = target.view(-1, 4)
        tl = torch.max((pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2))
        br = torch.min((pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2))

        area_p = torch.prod(pred[:, 2:], 1)
        area_g = torch.prod(target[:, 2:], 1)

        en = (tl < br).type(tl.type()).prod(dim=1)
        area_i = torch.prod(br - tl, 1) * en
        area_u = area_p + area_g - area_i
        iou = (area_i) / (area_u + 1e-16)

        if self.loss_type == "iou":
            loss = 1 - iou ** 2
        elif self.loss_type == "giou":
            c_tl = torch.min((pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2))
            c_br = torch.max((pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2))
            area_c = torch.prod(c_br - c_tl, 1)
            giou = iou - (area_c - area_u) / area_c.clamp(1e-16)
            loss = 1 - giou.clamp(min=-1.0, max=1.0)

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        return loss


def get_activation(name="silu", inplace=True):
    if name == "silu":
        module = nn.SiLU(inplace=inplace)
    elif name == "relu":
        module = nn.ReLU(inplace=inplace)
    elif name == "lrelu":
        module = nn.LeakyReLU(0.1, inplace=inplace)
    else:
        raise AttributeError("Unsupported act type: {}".format(name))
    return module

class BaseConv(nn.Module):
    """A Conv2d -> Batchnorm -> silu/leaky relu block"""
    def __init__(self, in_channels, out_channels, ksize, stride, groups=1, bias=False, act="silu"):
        super().__init__()
        # same padding
        pad = (ksize - 1) // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=ksize, stride=stride, padding=pad, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = get_activation(act, inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))


class DWConv(nn.Module):
    """Depthwise Conv + Conv"""

    def __init__(self, in_channels, out_channels, ksize, stride=1, act="silu"):
        super().__init__()
        self.dconv = BaseConv(in_channels, in_channels, ksize=ksize, stride=stride, groups=in_channels, act=act)
        self.pconv = BaseConv(in_channels, out_channels, ksize=1, stride=1, groups=1, act=act)

    def forward(self, x):
        x = self.dconv(x)
        return self.pconv(x)


class YOLOPAFPN(nn.Module):
    """
    YOLOv3 model. Darknet 53 is the default backbone of this model.
    """
    def __init__(self,depth=1.0,width=1.0,in_features=("dark3", "dark4", "dark5"),in_channels=[256, 512, 1024],depthwise=False,act="silu"):
        super().__init__()
        self.backbone = CSPDarknet(depth, width, depthwise=depthwise, act=act)
        self.in_features = in_features
        self.in_channels = in_channels
        Conv = DWConv if depthwise else BaseConv

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.lateral_conv0 = BaseConv(int(in_channels[2] * width), int(in_channels[1] * width), 1, 1, act=act)
        self.C3_p4 = CSPLayer(int(2 * in_channels[1] * width),int(in_channels[1] * width),round(3 * depth),False,depthwise=depthwise,act=act) # cat

        self.reduce_conv1 = BaseConv(int(in_channels[1] * width), int(in_channels[0] * width), 1, 1, act=act)
        self.C3_p3 = CSPLayer(int(2 * in_channels[0] * width),int(in_channels[0] * width),round(3 * depth),False,depthwise=depthwise,act=act)

        # bottom-up conv
        self.bu_conv2 = Conv(int(in_channels[0] * width), int(in_channels[0] * width), 3, 2, act=act)
        self.C3_n3 = CSPLayer(int(2 * in_channels[0] * width),int(in_channels[1] * width),round(3 * depth),False,depthwise=depthwise,act=act)

        # bottom-up conv
        self.bu_conv1 = Conv(int(in_channels[1] * width), int(in_channels[1] * width), 3, 2, act=act)
        self.C3_n4 = CSPLayer(int(2 * in_channels[1] * width),int(in_channels[2] * width),round(3 * depth),False,depthwise=depthwise,act=act)

    def forward(self, input):
        """
        Args:
            inputs: input images.

        Returns:
            Tuple[Tensor]: FPN feature.
        """
        #  backbone
        out_features = self.backbone(input)
        features = [out_features[f] for f in self.in_features]
        [x2, x1, x0] = features

        fpn_out0 = self.lateral_conv0(x0)  # 1024->512/32
        f_out0 = self.upsample(fpn_out0)  # 512/16
        f_out0 = torch.cat([f_out0, x1], 1)  # 512->1024/16
        f_out0 = self.C3_p4(f_out0)  # 1024->512/16

        fpn_out1 = self.reduce_conv1(f_out0)  # 512->256/16
        f_out1 = self.upsample(fpn_out1)  # 256/8
        f_out1 = torch.cat([f_out1, x2], 1)  # 256->512/8
        pan_out2 = self.C3_p3(f_out1)  # 512->256/8

        p_out1 = self.bu_conv2(pan_out2)  # 256->256/16
        p_out1 = torch.cat([p_out1, fpn_out1], 1)  # 256->512/16
        pan_out1 = self.C3_n3(p_out1)  # 512->512/16

        p_out0 = self.bu_conv1(pan_out1)  # 512->512/32
        p_out0 = torch.cat([p_out0, fpn_out0], 1)  # 512->1024/32
        pan_out0 = self.C3_n4(p_out0)  # 1024->1024/32

        outputs = (pan_out2, pan_out1, pan_out0)
        return outputs


class Darknet(nn.Module):
    # number of blocks from dark2 to dark5.
    depth2blocks = {21: [1, 2, 2, 1], 53: [2, 8, 8, 4]}

    def __init__(self,depth,in_channels=3,stem_out_channels=32,out_features=("dark3", "dark4", "dark5")):
        """
        Args:
            depth (int): depth of darknet used in model, usually use [21, 53] for this param.
            in_channels (int): number of input channels, for example, use 3 for RGB image.
            stem_out_channels (int): number of output channels of darknet stem.
                It decides channels of darknet layer2 to layer5.
            out_features (Tuple[str]): desired output layer name.
        """
        super().__init__()
        assert out_features, "please provide output features of Darknet"
        self.out_features = out_features
        self.stem = nn.Sequential(BaseConv(in_channels, stem_out_channels, ksize=3, stride=1, act="lrelu"),
                                  *self.make_group_layer(stem_out_channels, num_blocks=1, stride=2))
        in_channels = stem_out_channels * 2  # 64

        num_blocks = Darknet.depth2blocks[depth]
        # create darknet with `stem_out_channels` and `num_blocks` layers.
        # to make model structure more clear, we don't use `for` statement in python.
        self.dark2 = nn.Sequential(*self.make_group_layer(in_channels, num_blocks[0], stride=2))
        in_channels *= 2  # 128
        self.dark3 = nn.Sequential(*self.make_group_layer(in_channels, num_blocks[1], stride=2))
        in_channels *= 2  # 256
        self.dark4 = nn.Sequential(*self.make_group_layer(in_channels, num_blocks[2], stride=2))
        in_channels *= 2  # 512

        self.dark5 = nn.Sequential(*self.make_group_layer(in_channels, num_blocks[3], stride=2),
                                   *self.make_spp_block([in_channels, in_channels * 2], in_channels * 2))

    def make_group_layer(self, in_channels: int, num_blocks: int, stride: int = 1):
        "starts with conv layer then has `num_blocks` `ResLayer`"
        return [BaseConv(in_channels, in_channels * 2, ksize=3, stride=stride, act="lrelu"),
                *[(ResLayer(in_channels * 2)) for _ in range(num_blocks)]]

    def make_spp_block(self, filters_list, in_filters):
        m = nn.Sequential(*[BaseConv(in_filters, filters_list[0], 1, stride=1, act="lrelu"),
                            BaseConv(filters_list[0], filters_list[1], 3, stride=1, act="lrelu"),
                            SPPBottleneck(in_channels=filters_list[1],out_channels=filters_list[0],activation="lrelu"),
                            BaseConv(filters_list[0], filters_list[1], 3, stride=1, act="lrelu"),
                            BaseConv(filters_list[1], filters_list[0], 1, stride=1, act="lrelu")])
        
        return m

    def forward(self, x):
        outputs = {}
        x = self.stem(x)
        outputs["stem"] = x
        x = self.dark2(x)
        outputs["dark2"] = x
        x = self.dark3(x)
        outputs["dark3"] = x
        x = self.dark4(x)
        outputs["dark4"] = x
        x = self.dark5(x)
        outputs["dark5"] = x
        return {k: v for k, v in outputs.items() if k in self.out_features}


class CSPDarknet(nn.Module):
    def __init__(self, dep_mul, wid_mul, out_features=("dark3", "dark4", "dark5"), depthwise=False, act="silu"):
        super().__init__()
        assert out_features, "please provide output features of Darknet"
        self.out_features = out_features
        Conv = DWConv if depthwise else BaseConv

        base_channels = int(wid_mul * 64) # 64
        base_depth = max(round(dep_mul * 3), 1) # 3

        # stem
        self.stem = Focus(3, base_channels, ksize=3, act=act)

        # dark2
        self.dark2 = nn.Sequential(Conv(base_channels, base_channels * 2, 3, 2, act=act),
                                   CSPLayer(base_channels * 2, base_channels * 2, n=base_depth, depthwise=depthwise, act=act))

        # dark3
        self.dark3 = nn.Sequential(Conv(base_channels * 2, base_channels * 4, 3, 2, act=act),
                                   CSPLayer(base_channels * 4, base_channels * 4, n=base_depth * 3, depthwise=depthwise, act=act))

        # dark4
        self.dark4 = nn.Sequential(Conv(base_channels * 4, base_channels * 8, 3, 2, act=act),
                                   CSPLayer(base_channels * 8, base_channels * 8, n=base_depth * 3, depthwise=depthwise, act=act))

        # dark5
        self.dark5 = nn.Sequential(Conv(base_channels * 8, base_channels * 16, 3, 2, act=act),
                                   SPPBottleneck(base_channels * 16, base_channels * 16, activation=act),
                                   CSPLayer(base_channels * 16, base_channels * 16, n=base_depth, shortcut=False, depthwise=depthwise, act=act))

    def forward(self, x):
        outputs = {}
        x = self.stem(x)
        outputs["stem"] = x
        x = self.dark2(x)
        outputs["dark2"] = x
        x = self.dark3(x)
        outputs["dark3"] = x
        x = self.dark4(x)
        outputs["dark4"] = x
        x = self.dark5(x)
        outputs["dark5"] = x
        return {k: v for k, v in outputs.items() if k in self.out_features}


class SiLU(nn.Module):
    """export-friendly version of nn.SiLU()"""
    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)


class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, in_channels, out_channels, shortcut=True, expansion=0.5, depthwise=False, act="silu"):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        Conv = DWConv if depthwise else BaseConv
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv2 = Conv(hidden_channels, out_channels, 3, stride=1, act=act)
        self.use_add = shortcut and in_channels == out_channels

    def forward(self, x):
        y = self.conv2(self.conv1(x))
        if self.use_add:
            y = y + x
        return y


class ResLayer(nn.Module):
    "Residual layer with `in_channels` inputs."

    def __init__(self, in_channels: int):
        super().__init__()
        mid_channels = in_channels // 2
        self.layer1 = BaseConv(in_channels, mid_channels, ksize=1, stride=1, act="lrelu")
        self.layer2 = BaseConv(mid_channels, in_channels, ksize=3, stride=1, act="lrelu")

    def forward(self, x):
        out = self.layer2(self.layer1(x))
        return x + out


class SPPBottleneck(nn.Module):
    """Spatial pyramid pooling layer used in YOLOv3-SPP"""

    def __init__(self, in_channels, out_channels, kernel_sizes=(5, 9, 13), activation="silu"):
        super().__init__()
        hidden_channels = in_channels // 2
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=activation)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=ks, stride=1, padding=ks // 2) for ks in kernel_sizes])
        conv2_channels = hidden_channels * (len(kernel_sizes) + 1)
        self.conv2 = BaseConv(conv2_channels, out_channels, 1, stride=1, act=activation)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.cat([x] + [m(x) for m in self.m], dim=1)
        x = self.conv2(x)
        return x


class CSPLayer(nn.Module):
    """C3 in yolov5, CSP Bottleneck with 3 convolutions"""

    def __init__(self, in_channels, out_channels, n=1, shortcut=True, expansion=0.5, depthwise=False, act="silu"):
        """
        Args:
            in_channels (int): input channels.
            out_channels (int): output channels.
            n (int): number of Bottlenecks. Default value: 1.
        """
        # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        hidden_channels = int(out_channels * expansion)  # hidden channels
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv2 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv3 = BaseConv(2 * hidden_channels, out_channels, 1, stride=1, act=act)
        module_list = [Bottleneck(hidden_channels, hidden_channels, shortcut, 1.0, depthwise, act=act) for _ in range(n)]
        self.m = nn.Sequential(*module_list)

    def forward(self, x):
        x_1 = self.conv1(x)
        x_2 = self.conv2(x)
        x_1 = self.m(x_1)
        x = torch.cat((x_1, x_2), dim=1)
        return self.conv3(x)


class Focus(nn.Module):
    """Focus width and height information into channel space."""

    def __init__(self, in_channels, out_channels, ksize=1, stride=1, act="silu"):
        super().__init__()
        self.conv = BaseConv(in_channels * 4, out_channels, ksize, stride, act=act)

    def forward(self, x):
        # shape of x (b,c,w,h) -> y(b,4c,w/2,h/2)
        patch_top_left = x[..., ::2, ::2]
        patch_top_right = x[..., ::2, 1::2]
        patch_bot_left = x[..., 1::2, ::2]
        patch_bot_right = x[..., 1::2, 1::2]
        x = torch.cat((patch_top_left, patch_bot_left, patch_top_right, patch_bot_right), dim=1)
        return self.conv(x)


class YOLOX(nn.Module):

    def __init__(self, backbone=None, head=None):
        super().__init__()
        if backbone is None:
            backbone = YOLOPAFPN()
        if head is None:
            head = YOLOXHead(80)

        self.backbone = backbone
        self.head = head

    def forward(self, x, targets=None):
        # fpn output content features of [dark3, dark4, dark5]
        fpn_outs = self.backbone(x)

        if self.training:
            assert targets is not None
            loss, iou_loss, conf_loss, cls_loss, l1_loss, num_fg = self.head(fpn_outs, targets, x)
            outputs = {"total_loss" : loss,
                       "iou_loss"   : iou_loss,
                       "l1_loss"    : l1_loss,
                       "conf_loss"  : conf_loss,
                       "cls_loss"   : cls_loss,
                       "num_fg"     : num_fg}
        else:
            outputs, margin = self.head(fpn_outs, targets, x)

        return outputs, margin


@functools.lru_cache()
def _get_global_gloo_group():
    """
    Return a process group based on gloo backend, containing all the ranks
    The result is cached.
    """
    if dist.get_backend() == "nccl":
        return dist.new_group(backend="gloo")
    else:
        return dist.group.WORLD


def get_world_size() -> int:
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def _serialize_to_tensor(data, group):
    backend = dist.get_backend(group)
    assert backend in ["gloo", "nccl"]
    device = torch.device("cpu" if backend == "gloo" else "cuda")

    buffer = pickle.dumps(data)
    if len(buffer) > 1024 ** 3:
        logger.warning("Rank {} trying to all-gather {:.2f} GB of data on device {}".format(get_rank(), len(buffer) / (1024 ** 3), device))
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to(device=device)
    return tensor


def _pad_to_largest_tensor(tensor, group):
    """
    Returns:
        list[int]: size of the tensor, on each rank
        Tensor: padded tensor that has the max size
    """
    world_size = dist.get_world_size(group=group)
    assert (world_size >= 1), "comm.gather/all_gather must be called from ranks within the given group!"
    local_size = torch.tensor([tensor.numel()], dtype=torch.int64, device=tensor.device)
    size_list = [torch.zeros([1], dtype=torch.int64, device=tensor.device) for _ in range(world_size)]
    dist.all_gather(size_list, local_size, group=group)
    size_list = [int(size.item()) for size in size_list]

    max_size = max(size_list)

    # we pad the tensor because torch all_gather does not support
    # gathering tensors of different shapes
    if local_size != max_size:
        padding = torch.zeros((max_size - local_size,), dtype=torch.uint8, device=tensor.device)
        tensor = torch.cat((tensor, padding), dim=0)
    return size_list, tensor


def gather(data, dst=0, group=None):
    if get_world_size() == 1:
        return [data]
    if group is None:
        group = _get_global_gloo_group()
    if dist.get_world_size(group=group) == 1:
        return [data]
    rank = dist.get_rank(group=group)

    tensor = _serialize_to_tensor(data, group)
    size_list, tensor = _pad_to_largest_tensor(tensor, group)

    # receiving Tensor from all ranks
    if rank == dst:
        max_size = max(size_list)
        tensor_list = [torch.empty((max_size,), dtype=torch.uint8, device=tensor.device) for _ in size_list]
        dist.gather(tensor, tensor_list, dst=dst, group=group)

        data_list = []
        for size, tensor in zip(size_list, tensor_list):
            buffer = tensor.cpu().numpy().tobytes()[:size]
            data_list.append(pickle.loads(buffer))
        return data_list
    else:
        dist.gather(tensor, [], dst=dst, group=group)
        return []


def is_main_process() -> bool:
    return get_rank() == 0


def postprocess(prediction, num_classes, conf_thre=0.7, nms_thre=0.45, class_agnostic=False):
    box_corner = prediction.new(prediction.shape)
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    prediction[:, :, :4] = box_corner[:, :, :4]

    output = [None for _ in range(len(prediction))]
    for i, image_pred in enumerate(prediction):

        # If none are remaining => process next image
        if not image_pred.size(0):
            continue
        # Get score and class with highest confidence
        class_conf, class_pred = torch.max(image_pred[:, 5: 5 + num_classes], 1, keepdim=True)

        conf_mask = (image_pred[:, 4] * class_conf.squeeze() >= conf_thre).squeeze()
        # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
        detections = torch.cat((image_pred[:, :5], class_conf, class_pred.float()), 1)
        detections = detections[conf_mask]
        if not detections.size(0):
            continue

        if class_agnostic:
            nms_out_index = torchvision.ops.nms(detections[:, :4], detections[:, 4] * detections[:, 5], nms_thre)
        else:
            nms_out_index = torchvision.ops.batched_nms( detections[:, :4], detections[:, 4] * detections[:, 5], detections[:, 6], nms_thre)

        detections = detections[nms_out_index]
        if output[i] is None:
            output[i] = detections
        else:
            output[i] = torch.cat((output[i], detections))

    return output


def synchronize():
    """
    Helper function to synchronize (barrier) among all processes when using distributed training
    """
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    dist.barrier()


def time_synchronized():
    """pytorch-accurate time"""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()


def xyxy2xywh(bboxes):
    bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 0]
    bboxes[:, 3] = bboxes[:, 3] - bboxes[:, 1]
    return bboxes


class VOCEvaluator:
    def __init__(self, dataloader, img_size, confthre, nmsthre, num_classes):
        self.dataloader = dataloader
        self.img_size = img_size
        self.confthre = confthre
        self.nmsthre = nmsthre
        self.num_classes = num_classes
        self.num_images = len(dataloader.dataset)

    def evaluate(self, model, distributed=False, half=False, trt_file=None, decoder=None, test_size=None):
        # TODO half to amp_test
        tensor_type = torch.cuda.HalfTensor if half else torch.cuda.FloatTensor
        model = model.eval()
        if half:
            model = model.half()
        ids = []
        data_list = {}
        progress_bar = tqdm if is_main_process() else iter

        inference_time = 0
        nms_time = 0
        n_samples = max(len(self.dataloader) - 1, 1)

        margin_small, margin_midle, margin_large, margin_total = [], [], [], []

        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones(1, 3, test_size[0], test_size[1]).cuda()
            model(x)
            model = model_trt

        for cur_iter, (imgs, labs, info_imgs, ids) in enumerate(progress_bar(self.dataloader)):
            with torch.no_grad():
                imgs = imgs.type(tensor_type)
                labs = labs.type(tensor_type)

                # skip the the last iters since batchsize might be not enough for batch inference
                is_time_record = cur_iter < len(self.dataloader) - 1
                if is_time_record:
                    start = time.time()

                outputs, margin = model(imgs, labs) # outputs is torch.tensor

                if decoder is not None:
                    outputs = decoder(outputs, dtype=outputs.type())
                
                margin_small.append(margin[0])
                margin_midle.append(margin[1])
                margin_large.append(margin[2])
                margin_total.append(margin[3])

                if is_time_record:
                    infer_end = time_synchronized()
                    inference_time += infer_end - start

                outputs = postprocess(outputs, self.num_classes, self.confthre, self.nmsthre)
                
                if is_time_record:
                    nms_end = time_synchronized()
                    nms_time += nms_end - infer_end

            data_list.update(self.convert_to_voc_format(outputs, info_imgs, ids))
        
        print(f'Testset Small Objects Margin: {sum(margin_small) / len(margin_small)}')
        print(f'Testset Midle Objects Margin: {sum(margin_midle) / len(margin_midle)}')
        print(f'Testset Large Objects Margin: {sum(margin_large) / len(margin_large)}')
        print(f'Testset Total Objects Margin: {sum(margin_total) / len(margin_total)}')

        statistics = torch.cuda.FloatTensor([inference_time, nms_time, n_samples])
        if distributed:
            data_list = gather(data_list, dst=0)
            data_list = ChainMap(*data_list)
            torch.distributed.reduce(statistics, dst=0)

        eval_results, mAP50, scale_AP = self.evaluate_prediction(data_list, statistics)
        synchronize()
        # return eval_results
        return [sum(margin_small) / len(margin_small), sum(margin_midle) / len(margin_midle),
                sum(margin_large) / len(margin_large), sum(margin_total) / len(margin_total)], mAP50, scale_AP

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

        time_info = ", ".join(["Average {} time: {:.2f} ms".format(k, v)
                               for k, v in zip(["forward", "NMS", "inference"],
                                               [a_infer_time, a_nms_time, (a_infer_time + a_nms_time)],)])

        info = time_info + "\n"

        all_boxes = [[[] for _ in range(self.num_images)] for _ in range(self.num_classes)]
        
        for img_num in range(self.num_images):
            bboxes, cls, scores = data_dict[img_num]
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
                all_boxes[j][img_num] = c_dets[mask_c].numpy() # all_boxes: 外层为类别, 中间为每张图片预测一遍该类别的框, 如果存在则输出坐标, 如果不存在则写入空列表

            sys.stdout.write("im_eval: {:d}/{:d} \r".format(img_num + 1, self.num_images))
            sys.stdout.flush()
        
        # for i in range(len(all_boxes)):
            # print("-----------------")
            # for j in range(len(all_boxes[i])):
                # print(all_boxes[i][j])
        
        with tempfile.TemporaryDirectory() as tempdir:
            mAP50, mAP70, scale_AP = self.dataloader.dataset.evaluate_detections(all_boxes, tempdir)
            return [mAP50, mAP70, info], mAP70, scale_AP


class Dataset(torchDataset):
    def __init__(self, input_dimension, mosaic=True):
        super().__init__()
        self.__input_dim = input_dimension[:2]
        self.enable_mosaic = mosaic

    @property
    def input_dim(self):
        if hasattr(self, "_input_dim"):
            return self._input_dim
        return self.__input_dim

    @staticmethod
    def mosaic_getitem(getitem_fn):
        @wraps(getitem_fn)
        def wrapper(self, index):
            if not isinstance(index, int):
                self.enable_mosaic = index[0]
                index = index[1]

            ret_val = getitem_fn(self, index)

            return ret_val

        return wrapper


class AnnotationTransform(object):
    def __init__(self, class_to_ind=None, keep_difficult=True):
        self.class_to_ind = class_to_ind or dict(zip(VOC_CLASSES, range(len(VOC_CLASSES))))
        self.keep_difficult = keep_difficult

    def __call__(self, target):
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


def parse_rec(filename):
    """ Parse a PASCAL VOC xml file """
    
    # 使用GPU时此句需注释掉, 使用CPU需要加上此句
    path = "F:/VOC_test/VOCdevkit/VOC2007/Annotations"
    if (path in filename) == False:
        filename = path + os.sep + filename
    
    
    tree = ET.parse(filename)
    objects = []
    for obj in tree.findall("object"):
        obj_struct = {}
        obj_struct["name"] = obj.find("name").text
        obj_struct["pose"] = obj.find("pose").text
        obj_struct["truncated"] = int(obj.find("truncated").text)
        obj_struct["difficult"] = int(obj.find("difficult").text)
        bbox = obj.find("bndbox")
        obj_struct["bbox"] = [int(bbox.find("xmin").text),
                              int(bbox.find("ymin").text),
                              int(bbox.find("xmax").text),
                              int(bbox.find("ymax").text),]
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


def voc_eval(detpath, annopath, imagesetfile, classname, cachedir, ovthresh=0.5, use_07_metric=False, flag=None):
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
    
    if flag == 'scale':
        if classname == 'small':
            areaRng = [0, 32**2]
        if classname == 'medium':
            areaRng = [32**2, 96**2]
        if classname == 'large':
            areaRng = [96**2, 512**2]
    
    # extract gt objects for this class 将GT分成不同的类别进行处理, 多次调用voc_eval, 每次处理一个类别
    class_recs = {}
    npos = 0
    for imagename in imagenames:
        if flag == 'class':
            R = [obj for obj in recs[imagename] if obj["name"] == classname]
        if flag == 'scale':
            R = [obj for obj in recs[imagename] if areaRng[0] < (obj["bbox"][2] - obj["bbox"][0]) * (obj["bbox"][3] - obj["bbox"][1]) <= areaRng[1]]
        bbox = np.array([x["bbox"] for x in R])
        difficult = np.array([x["difficult"] for x in R]).astype(bool)
        det = [False] * len(R)
        npos = npos + sum(~difficult)
        class_recs[imagename] = {"bbox": bbox, "difficult": difficult, "det": det}

    # read dets 从txt文件中读取Pred
    detfile = detpath.format(classname)
    with open(detfile, "r") as f:
        lines = f.readlines()

    if len(lines) == 0:
        return 0, 0, 0
    
    # 单类的所有图片的检测结果
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
            uni = ((bb[2] - bb[0] + 1.0) * (bb[3] - bb[1] + 1.0) + (BBGT[:, 2] - BBGT[:, 0] + 1.0) * (BBGT[:, 3] - BBGT[:, 1] + 1.0) - inters)

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


class VOCDetection(Dataset):
    def __init__(self, data_dir, image_sets=[("2007", "trainval"), ("2012", "trainval")], img_size=(416, 416),
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
                self.ids.append((rootpath, line.strip()))

        self.annotations = self._load_coco_annotations()
        self.imgs = None
        if cache:
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
        self._write_voc_results_file_class(all_boxes)
        self._write_voc_results_file_scale(all_boxes)
        IouTh = np.linspace(0.5, 0.95, int(np.round((0.95 - 0.5) / 0.05)) + 1, endpoint=True)
        mAPs, scaleAPs = [], []
        for iou in IouTh:
            mAP, scale_AP = self._do_python_eval(output_dir, iou)
            mAPs.append(mAP)
            print(f"mAP: {mAP} with IoU: {iou}")
            scaleAPs.append(scale_AP)

        print("--------------------------------------------------------------")
        print("map_5095:", np.mean(mAPs))
        print("map_50:", mAPs[0])
        print("--------------------------------------------------------------")
        return np.mean(mAPs), mAPs[0], scaleAPs[0]

    def _get_voc_results_file_template(self):
        filename = "comp4_det_test" + "_{:s}.txt"
        filedir = os.path.join(self.root, "results", "VOC" + self._year, "Main")
        if not os.path.exists(filedir):
            os.makedirs(filedir)
        path = os.path.join(filedir, filename)
        return path
    
    def _write_voc_results_file_class(self, all_boxes):
        for cls_ind, cls in enumerate(VOC_CLASSES):
            cls_ind = cls_ind
            if cls == "__background__":
                continue
            print("Writing {} VOC results file".format(cls))
            filename = self._get_voc_results_file_template().format(cls)
            with open(filename, "wt") as f:
                for im_ind, index in enumerate(self.ids): # self.ids姑且认为是所有图片的{路径,名称}的列表
                # im_ind是序号, index是{路径,名称}
                    index = index[1]
                    dets = all_boxes[cls_ind][im_ind]
                    if dets == []:
                        continue
                    for k in range(dets.shape[0]):
                        f.write("{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n".format(index, dets[k, -1],
                                                                                   dets[k, 0] + 1, dets[k, 1] + 1,
                                                                                   dets[k, 2] + 1, dets[k, 3] + 1))
    
    # 最后生成3个文件, 即按照小、中、大目标将所有目标分类
    def _write_voc_results_file_scale(self, all_boxes):
        areaRng = [[0, 32**2], [32**2, 96**2], [96**2, 512**2]]
        areaNam = ['small', 'medium', 'large']
        for rng in range(len(areaRng)):
            filename = self._get_voc_results_file_template().format(areaNam[rng])
            with open(filename, "wt") as f:
                for cls_ind, cls in enumerate(VOC_CLASSES):
                    cls_ind = cls_ind
                    if cls == "__background__":
                        continue
                    for im_ind, index in enumerate(self.ids):
                        index = index[1]
                        dets = all_boxes[cls_ind][im_ind]
                        # if dets == []:
                        if len(dets) == 0:
                            continue
                        for k in range(dets.shape[0]):
                            if areaRng[rng][0] < (dets[k][2] - dets[k][0]) * (dets[k][3] - dets[k][1]) < areaRng[rng][1]:
                                f.write("{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n".format(index, dets[k, -1],
                                                                                           dets[k, 0] + 1, dets[k, 1] + 1,
                                                                                           dets[k, 2] + 1, dets[k, 3] + 1))
    
    def class_ap_eval(self, annopath, imagesetfile, cachedir, iou, output_dir, use_07_metric):
        aps = []
        for i, cls in enumerate(VOC_CLASSES):
            if cls == "__background__":
                continue
            filename = self._get_voc_results_file_template().format(cls)
            rec, prec, ap = voc_eval(filename, annopath, imagesetfile, cls, cachedir, ovthresh=iou, use_07_metric=use_07_metric, flag='class')
            aps += [ap]
            if iou == 0.5:
                print("AP for {} = {:.4f}".format(cls, ap))
            if output_dir is not None:
                with open(os.path.join(output_dir, cls + "_pr.pkl"), "wb") as f:
                    pickle.dump({"rec": rec, "prec": prec, "ap": ap}, f)
        return aps
    
    def scale_ap_eval(self, annopath, imagesetfile, cachedir, iou, output_dir, use_07_metric):
        areaNam, aps = ['small', 'medium', 'large'], []
        for i in range(len(areaNam)):
            filename = self._get_voc_results_file_template().format(areaNam[i])
            rec, prec, ap = voc_eval(filename, annopath, imagesetfile, areaNam[i], cachedir, ovthresh=iou, use_07_metric=use_07_metric, flag='scale')
            aps += [ap]
            if iou == 0.5:
                print("AP for {} = {:.4f}".format(areaNam[i], ap))
            if output_dir is not None:
                with open(os.path.join(output_dir, areaNam[i] + "_pr.pkl"), "wb") as f:
                    pickle.dump({"rec": rec, "prec": prec, "ap": ap}, f)
        return aps

    def _do_python_eval(self, output_dir="output", iou=0.5):
        rootpath = os.path.join(self.root, "VOC" + self._year)
        name = self.image_set[0][1]
        annopath = os.path.join(rootpath, "Annotations", "{:s}.xml")
        imagesetfile = os.path.join(rootpath, "ImageSets", "Main", name + ".txt")
        cachedir = os.path.join(self.root, "annotations_cache", "VOC" + self._year, name)
        if not os.path.exists(cachedir): os.makedirs(cachedir)
        
        # The PASCAL VOC metric changed in 2010
        use_07_metric = True if int(self._year) < 2010 else False
        # print("Eval IoU : {:.2f}".format(iou))
        if output_dir is not None and not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        
        cls_aps = self.class_ap_eval(annopath, imagesetfile, cachedir, iou, output_dir, use_07_metric)
        scl_aps = self.scale_ap_eval(annopath, imagesetfile, cachedir, iou, output_dir, use_07_metric)
        
        if iou == 0.5:
            # print("Mean AP = {:.4f}".format(np.mean(aps)))
            # print("~~~~~~~~")
            # print("Results:")
            # for ap in aps:
            #     print("{:.3f}".format(ap))
            # print("{:.3f}".format(np.mean(aps)))
            # print("~~~~~~~~")
            print("")
            # print("--------------------------------------------------------------")
            # print("Results computed with the **unofficial** Python eval code.")
            # print("Results should be very close to the official MATLAB eval code.")
            # print("Recompute with `./tools/reval.py --matlab ...` for your paper.")
            # print("-- Thanks, The Management")
            # print("--------------------------------------------------------------")

        return np.mean(cls_aps), scl_aps


def preproc(img, input_size, swap=(2, 0, 1)):
    if len(img.shape) == 3:
        padded_img = np.ones((input_size[0], input_size[1], 3), dtype=np.uint8) * 114
    else:
        padded_img = np.ones(input_size, dtype=np.uint8) * 114

    r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
    resized_img = cv2.resize(img, (int(img.shape[1] * r), int(img.shape[0] * r)), interpolation=cv2.INTER_LINEAR,).astype(np.uint8)
    padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img

    padded_img = padded_img.transpose(swap)
    padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
    return padded_img, r


class ValTransform:
    def __init__(self, swap=(2, 0, 1), legacy=False):
        self.swap = swap
        self.legacy = legacy

    # assume input is cv2 img for now
    def __call__(self, img, res, input_size):
        img, _ = preproc(img, input_size, self.swap)
        if self.legacy:
            img = img[::-1, :, :].copy()
            img /= 255.0
            img -= np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
            img /= np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
        return img, np.zeros((1, 5))


def xyxy2cxcywh(bboxes):
    bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 0]
    bboxes[:, 3] = bboxes[:, 3] - bboxes[:, 1]
    bboxes[:, 0] = bboxes[:, 0] + bboxes[:, 2] * 0.5
    bboxes[:, 1] = bboxes[:, 1] + bboxes[:, 3] * 0.5
    return bboxes


def _mirror(image, boxes, prob=0.5):
    _, width, _ = image.shape
    if random.random() < prob:
        image = image[:, ::-1]
        boxes[:, 0::2] = width - boxes[:, 2::-2]
    return image, boxes


class TrainTransform:
    def __init__(self, max_labels=50, flip_prob=0.5, hsv_prob=1.0):
        self.max_labels = max_labels
        self.flip_prob = flip_prob
        self.hsv_prob = hsv_prob

    def __call__(self, image, targets, input_dim):
        boxes = targets[:, :4].copy()
        labels = targets[:, 4].copy()
        if len(boxes) == 0:
            targets = np.zeros((self.max_labels, 5), dtype=np.float32)
            image, r_o = preproc(image, input_dim)
            return image, targets

        image_o = image.copy()
        targets_o = targets.copy()
        height_o, width_o, _ = image_o.shape
        boxes_o = targets_o[:, :4]
        labels_o = targets_o[:, 4]
        # bbox_o: [xyxy] to [c_x,c_y,w,h]
        boxes_o = xyxy2cxcywh(boxes_o)

        if random.random() < self.hsv_prob:
            augment_hsv(image)
        image_t, boxes = _mirror(image, boxes, self.flip_prob)
        height, width, _ = image_t.shape
        image_t, r_ = preproc(image_t, input_dim)
        # boxes [xyxy] 2 [cx,cy,w,h]
        boxes = xyxy2cxcywh(boxes)
        boxes *= r_

        mask_b = np.minimum(boxes[:, 2], boxes[:, 3]) > 1
        boxes_t = boxes[mask_b]
        labels_t = labels[mask_b]

        if len(boxes_t) == 0:
            image_t, r_o = preproc(image_o, input_dim)
            boxes_o *= r_o
            boxes_t = boxes_o
            labels_t = labels_o

        labels_t = np.expand_dims(labels_t, 1)

        targets_t = np.hstack((labels_t, boxes_t))
        padded_labels = np.zeros((self.max_labels, 5))
        padded_labels[range(len(targets_t))[: self.max_labels]] = targets_t[: self.max_labels]
        padded_labels = np.ascontiguousarray(padded_labels, dtype=np.float32)
        return image_t, padded_labels


class Exp():
    def __init__(self, test_path, num_cls, depth, width, test_size, output_dir):
        super().__init__()
        
        self.data_num_workers = 0
        
        # ---------------- model config ---------------- #
        self.num_classes = num_cls # detect classes number of model
        self.depth = depth # factor of model depth
        self.width = width # factor of model width
        self.act = "silu" # activation name "relu" or "silu"

        # name of experiment
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
        self.test_conf = 0.01
        self.test_size = test_size
        self.nmsthre = 0.65
        self.output_dir = output_dir
        self.testdata = test_path

    def get_model(self):

        def init_yolo(M):
            for m in M.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03

        if getattr(self, "model", None) is None:
            in_channels = [256, 512, 1024]
            backbone = YOLOPAFPN(self.depth, self.width, in_channels=in_channels, act=self.act)
            head = YOLOXHead(self.num_classes, self.width, in_channels=in_channels, act=self.act)
            self.model = YOLOX(backbone, head)

        self.model.apply(init_yolo)
        self.model.head.initialize_biases(1e-2)
        return self.model
    
    def get_eval_loader(self, batch_size, is_distributed, testdev=False, legacy=False):
        valdataset = VOCDetection(data_dir=os.path.join(self.testdata, "VOCdevkit"),
                                  image_sets=[('2007', 'test')],
                                  img_size=self.test_size,
                                  preproc=TrainTransform(max_labels=50, flip_prob=0, hsv_prob=0)) # ValTransform(legacy=legacy)

        if is_distributed:
            batch_size = batch_size // dist.get_world_size()
            sampler = torch.utils.data.distributed.DistributedSampler(valdataset, shuffle=False)
        else:
            sampler = torch.utils.data.SequentialSampler(valdataset)

        dataloader_kwargs = {"num_workers": self.data_num_workers,
                             "pin_memory": True,
                             "sampler": sampler,}
        
        dataloader_kwargs["batch_size"] = batch_size
        val_loader = torch.utils.data.DataLoader(valdataset, **dataloader_kwargs)

        return val_loader

    def get_evaluator(self, batch_size, is_distributed, testdev=False, legacy=False):
        val_loader = self.get_eval_loader(batch_size, is_distributed, testdev, legacy)
        evaluator = VOCEvaluator(dataloader=val_loader,
                                 img_size=self.test_size,
                                 confthre=self.test_conf,
                                 nmsthre=self.nmsthre,
                                 num_classes=self.num_classes)
        
        return evaluator


    def merge(self, cfg_list):
        assert len(cfg_list) % 2 == 0
        for k, v in zip(cfg_list[0::2], cfg_list[1::2]):
            if hasattr(self, k): # only update value with same key
                src_value = getattr(self, k)
                src_type = type(src_value)
                if src_value is not None and src_type != type(v):
                    try:
                        v = src_type(v)
                    except Exception:
                        v = ast.literal_eval(v)
                setattr(self, k, v)


def get_exp_by_file(exp_file, test_path, num_cls, depth, width, test_size, output_dir):
    try:
        # sys.path.append(os.path.dirname(exp_file))
        # current_exp = importlib.import_module(os.path.basename(exp_file).split(".")[0])
        # exp = current_exp.Exp()
        exp = Exp(test_path, num_cls, depth, width, test_size, output_dir)
    except Exception:
        raise ImportError("{} doesn't contains class named 'Exp'".format(exp_file))
    return exp


def get_exp_by_name(exp_name):
    exp = exp_name.replace("-", "_")  # convert string like "yolox-s" to "yolox_s"
    module_name = ".".join(["yolox", "exp", "default", exp])
    exp_object = importlib.import_module(module_name).Exp()
    return exp_object


def get_exp(test_path, num_cls, depth, width, test_size, output_dir, exp_file=None, exp_name=None):
    """
    get Exp object by file or name. If exp_file and exp_name
    are both provided, get Exp by exp_file.

    Args:
        exp_file (str): file path of experiment.
        exp_name (str): name of experiment. "yolo-s",
    """
    assert (exp_file is not None or exp_name is not None), "plz provide exp file or exp name."
    if exp_file is not None:
        return get_exp_by_file(exp_file, test_path, num_cls, depth, width, test_size, output_dir)
    else:
        return get_exp_by_name(exp_name)


def _distributed_worker(local_rank,main_func,world_size,num_gpus_per_machine,machine_rank,backend,dist_url,args,timeout=DEFAULT_TIMEOUT,):
    assert (torch.cuda.is_available()), "cuda is not available. Please check your installation."
    global_rank = machine_rank * num_gpus_per_machine + local_rank
    logger.info("Rank {} initialization finished.".format(global_rank))
    try:
        dist.init_process_group(backend=backend, init_method=dist_url, world_size=world_size, rank=global_rank, timeout=timeout)
    except Exception:
        logger.error("Process group URL: {}".format(dist_url))
        raise

    # nonlocal _LOCAL_PROCESS_GROUP

    # Setup the local process group (which contains ranks within the same machine)
    # assert _LOCAL_PROCESS_GROUP is None
    num_machines = world_size // num_gpus_per_machine
    for i in range(num_machines):
        ranks_on_i = list(range(i * num_gpus_per_machine, (i + 1) * num_gpus_per_machine))
        pg = dist.new_group(ranks_on_i)
        if i == machine_rank:
            _LOCAL_PROCESS_GROUP = pg

    # synchronize is needed here to prevent a possible timeout after calling init_process_group
    # See: https://github.com/facebookresearch/maskrcnn-benchmark/issues/172
    synchronize()

    assert num_gpus_per_machine <= torch.cuda.device_count()
    torch.cuda.set_device(local_rank)

    main_func(*args)


def synchronize():
    """
    Helper function to synchronize (barrier) among all processes when using distributed training
    """
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    dist.barrier()


def _find_free_port():
    """
    Find an available port of current machine / node.
    """
    import socket

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("", 0)) # Binding to port 0 will cause the OS to find an available port for us
    port = sock.getsockname()[1]
    sock.close() # NOTE: there is still a chance the port could be taken by other processes.
    
    return port


def configure_nccl():
    """Configure multi-machine environment variables of NCCL."""
    os.environ["NCCL_LAUNCH_MODE"] = "PARALLEL"
    os.environ["NCCL_IB_HCA"] = subprocess.getoutput("pushd /sys/class/infiniband/ > /dev/null; for i in mlx5_*; "
                                                     "do cat $i/ports/1/gid_attrs/types/* 2>/dev/null "
                                                     "| grep v >/dev/null && echo $i ; done; popd > /dev/null")
    os.environ["NCCL_IB_GID_INDEX"] = "3"
    os.environ["NCCL_IB_TC"] = "106"


def fuse_conv_and_bn(conv, bn):
    # Fuse convolution and batchnorm layers https://tehnokv.com/posts/fusing-batchnorm-and-conv/
    fusedconv = (
        nn.Conv2d(
            conv.in_channels,
            conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            groups=conv.groups,
            bias=True).requires_grad_(False).to(conv.weight.device))

    # prepare filters
    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
    fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.shape))

    # prepare spatial bias
    b_conv = (torch.zeros(conv.weight.size(0), device=conv.weight.device) if conv.bias is None else conv.bias)
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
    fusedconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)

    return fusedconv


def fuse_model(model):
    from yolox.models.network_blocks import BaseConv

    for m in model.modules():
        if type(m) is BaseConv and hasattr(m, "bn"):
            m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
            delattr(m, "bn")  # remove batchnorm
            m.forward = m.fuseforward  # update forward
    return model


def get_rank() -> int:
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def get_local_rank() -> int:
    """
    Returns:
        The rank of the current process within the local (per-machine) process group.
    """
    if _LOCAL_PROCESS_GROUP is None:
        return get_rank()

    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank(group=_LOCAL_PROCESS_GROUP)


def get_model_info(model, tsize):
    stride = 64
    img = torch.zeros((1, 3, stride, stride), device=next(model.parameters()).device)
    flops, params = profile(deepcopy(model), inputs=(img,), verbose=False)
    params /= 1e6
    flops /= 1e9
    flops *= tsize[0] * tsize[1] / stride / stride * 2  # Gflops
    info = "Params: {:.2f}M, Gflops: {:.2f}".format(params, flops)
    return info


def get_caller_name(depth=0):
    """
    Args:
        depth (int): Depth of caller conext, use 0 for caller depth.
        Default value: 0.

    Returns:
        str: module name of the caller
    """
    # the following logic is a little bit faster than inspect.stack() logic
    frame = inspect.currentframe().f_back
    for _ in range(depth):
        frame = frame.f_back

    return frame.f_globals["__name__"]


class StreamToLoguru:
    """
    stream object that redirects writes to a logger instance.
    """
    def __init__(self, level="INFO", caller_names=("apex", "pycocotools")):
        """
        Args:
            level(str): log level string of loguru. Default value: "INFO".
            caller_names(tuple): caller names of redirected module.
                Default value: (apex, pycocotools).
        """
        self.level = level
        self.linebuf = ""
        self.caller_names = caller_names

    def write(self, buf):
        full_name = get_caller_name(depth=1)
        module_name = full_name.rsplit(".", maxsplit=-1)[0]
        if module_name in self.caller_names:
            for line in buf.rstrip().splitlines(): # use caller level log
                logger.opt(depth=2).log(self.level, line.rstrip())
        else:
            sys.__stdout__.write(buf)

    def flush(self):
        pass


def redirect_sys_output(log_level="INFO"):
    redirect_logger = StreamToLoguru(log_level)
    sys.stderr = redirect_logger
    sys.stdout = redirect_logger


def setup_logger(save_dir, distributed_rank=0, filename="log.txt", mode="a"):
    """setup logger for training and testing.
    Args:
        save_dir(str): location to save log file
        distributed_rank(int): device rank when multi-gpu environment
        filename (string): log save name.
        mode(str): log file write mode, `append` or `override`. default is `a`.

    Return:
        logger instance.
    """
    loguru_format = ("<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
                     "<level>{level: <8}</level> | "
                     "<cyan>{name}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>")

    logger.remove()
    save_file = os.path.join(save_dir, filename)
    if mode == "o" and os.path.exists(save_file):
        os.remove(save_file)
    
    # only keep logger in rank0 process
    if distributed_rank == 0:
        logger.add(sys.stderr, format=loguru_format, level="INFO", enqueue=True)
        logger.add(save_file)

    # redirect stdout/stderr to loguru
    redirect_sys_output("INFO")


def make_parser():
    parser = argparse.ArgumentParser("YOLOX Eval")
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")
    
    # distributed
    parser.add_argument("--dist-backend", default="nccl", type=str, help="distributed backend")
    parser.add_argument("--dist-url", default=None, type=str, help="url used to set up distributed training")
    parser.add_argument("-b", "--batch-size", type=int, default=2, help="batch size")
    parser.add_argument("-d", "--devices", default=None, type=int, help="device for training")
    parser.add_argument("--num_machines", default=1, type=int, help="num of node for training")
    parser.add_argument("--machine_rank", default=0, type=int, help="node rank for multi-node training")
    parser.add_argument("-f", "--exp_file", default="D:/AICV-YoloX2/exps/default/yolox_s.py", type=str, help="input expriment file")

    parser.add_argument("-c", "--ckpt", default="/home/fyp/ChengYuxuan/YoloXDSTRe3/YOLOX_outputs/Baseline_S/epoch_50_ckpt.pth", type=str)

    parser.add_argument("--conf", default=None, type=float, help="test conf")
    parser.add_argument("--nms", default=None, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=None, type=int, help="test img size")
    parser.add_argument("--seed", default=None, type=int, help="eval seed")
    parser.add_argument("--fp16", dest="fp16", default=False, action="store_true", help="Adopting mix precision evaluating.")
    parser.add_argument("--fuse", dest="fuse", default=False, action="store_true", help="Fuse conv and bn for testing.")
    parser.add_argument("--trt", dest="trt", default=False, action="store_true", help="Using TensorRT model for testing.")
    parser.add_argument("--legacy", dest="legacy", default=False, action="store_true", help="To be compatible with older versions")
    parser.add_argument("--test", dest="test", default=False, action="store_true", help="Evaluating on test-dev set.")
    parser.add_argument("--speed", dest="speed", default=False, action="store_true", help="speed test only.")
    parser.add_argument("opts", help="Modify config options using the command-line", default=None, nargs=argparse.REMAINDER)
    return parser


@logger.catch
def main(exp, args, num_gpu):
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn("You have chosen to seed testing. This will turn on the CUDNN deterministic setting, ")

    is_distributed = num_gpu > 1

    # set environment variables for distributed training
    configure_nccl()
    cudnn.benchmark = True

    rank = get_local_rank()

    file_name = os.path.join(exp.output_dir, args.experiment_name)

    if rank == 0:
        os.makedirs(file_name, exist_ok=True)

    setup_logger(file_name, distributed_rank=rank, filename="val_log.txt", mode="a")
    # logger.info("Args: {}".format(args))

    if args.conf is not None:
        exp.test_conf = args.conf
    if args.nms is not None:
        exp.nmsthre = args.nms
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)

    model = exp.get_model()
    # logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))
    # logger.info("Model Structure:\n{}".format(str(model)))

    evaluator = exp.get_evaluator(args.batch_size, is_distributed, args.test, args.legacy)
    evaluator.per_class_AP = True
    evaluator.per_class_AR = True

    torch.cuda.set_device(rank)
    model.cuda(rank)
    model.eval()

    if not args.speed and not args.trt:
        if args.ckpt is None:
            ckpt_file = os.path.join(file_name, "best_ckpt.pth")
        else:
            ckpt_file = args.ckpt
        logger.info("loading checkpoint from {}".format(ckpt_file))
        loc = "cuda:{}".format(rank)
        ckpt = torch.load(ckpt_file, map_location=loc)
        model.load_state_dict(ckpt["model"])
        logger.info("loaded checkpoint done.")

    if is_distributed:
        model = DDP(model, device_ids=[rank])

    if args.fuse:
        logger.info("\tFusing model...")
        model = fuse_model(model)

    if args.trt:
        assert (not args.fuse and not is_distributed and args.batch_size == 1), \
                "TensorRT model is not support model fusing and distributed inferencing!"
        trt_file = os.path.join(file_name, "model_trt.pth")
        assert os.path.exists(trt_file), "TensorRT model is not found!\n Run tools/trt.py first!"
        model.head.decode_in_inference = False
        decoder = model.head.decode_outputs
    else:
        trt_file = None
        decoder = None

    # start evaluate
    margin, mAP50, scale_AP = evaluator.evaluate(model, is_distributed, args.fp16, trt_file, decoder, exp.test_size) # *_, summary
    # logger.info("\n" + summary)

    return margin, mAP50, scale_AP


def launch(main_func, num_gpus_per_machine, num_machines=1, machine_rank=0, backend="nccl", dist_url=None, args=(), timeout=DEFAULT_TIMEOUT):
    """
    Args:
        main_func: a function that will be called by `main_func(*args)`
        num_machines (int): the total number of machines
        machine_rank (int): the rank of this machine (one per machine)
        dist_url (str): url to connect to for distributed training, including protocol
                       e.g. "tcp://127.0.0.1:8686".
                       Can be set to auto to automatically select a free port on localhost
        args (tuple): arguments passed to main_func
    """
    world_size = num_machines * num_gpus_per_machine
    
    if world_size > 1: # 有卡就并行训练
        # TODO prctl in spawned processes https://github.com/pytorch/pytorch/pull/14391
        if dist_url == "auto":
            assert (num_machines == 1), "dist_url=auto cannot work with distributed training."
            port = _find_free_port()
            dist_url = f"tcp://127.0.0.1:{port}"

        start_method = "spawn"
        cache = vars(args[1]).get("cache", False)

        # To use numpy memmap for caching image into RAM, we have to use fork method
        if cache:
            assert sys.platform != "win32", ("As Windows platform doesn't support fork method, do not add --cache in your training command")
            start_method = "fork"

        mp.start_processes(_distributed_worker, nprocs=num_gpus_per_machine,
                           args=(main_func,world_size,num_gpus_per_machine,machine_rank,backend,dist_url,args),
                           daemon=False, start_method=start_method)
    else: # 没卡就直接一个主函数
        margin, mAP50, scale_AP = main_func(*args)
        return margin, mAP50, scale_AP


# if __name__ == "__main__":
def eval_voc_scale_v1(ckpt):
    args = make_parser().parse_args()

    # addition code
    args.ckpt = ckpt

    num_cls=20
    depth=0.33 # factor of model depth # nano: 0.33 | tiny: 0.330 | s: 0.33 | m: 0.67 | l: 1.0 | x: 1.33
    width=0.50 # factor of model width # nano: 0.25 | tiny: 0.375 | s: 0.50 | m: 0.75 | l: 1.0 | x: 1.25
    test_size=(512, 512)

    output_dir="./YOLOX_outputs"
    test_path = "/home/uic/VOC_test/"

    exp = get_exp(test_path, num_cls, depth, width, test_size, output_dir, args.exp_file, args.name)
    exp.merge(args.opts)

    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    num_gpu = torch.cuda.device_count() if args.devices is None else args.devices
    assert num_gpu <= torch.cuda.device_count()

    dist_url = "auto" if args.dist_url is None else args.dist_url
    margin, mAP50, scale_AP = launch(main, num_gpu, args.num_machines, args.machine_rank, backend=args.dist_backend, dist_url=dist_url, args=(exp, args, num_gpu))

    # addition code
    return margin, mAP50, scale_AP
