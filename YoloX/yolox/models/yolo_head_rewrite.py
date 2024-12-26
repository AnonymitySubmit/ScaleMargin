# -*- coding: utf-8 -*-

import copy
import math
import numpy as np
import statistics
from loguru import logger

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from typing import Optional

# 设定显示tensor的全部内容
torch.set_printoptions(threshold=100000)


"""Cover yolox/utils/boxes.py/bboxes_iou(), yolox/models/network_blocks.py/BaseConv()+DWConv()"""


class ECELoss(nn.Module):
    """Calculates the Expected Calibration Error of a model.
    (This isn't necessary for temperature scaling, just a cool metric).

    The input to this loss is the logits of a model, NOT the softmax scores.

    This divides the confidence outputs into equally-sized interval bins.
    In each bin, we compute the confidence gap:

    bin_gap = | avg_confidence_in_bin - accuracy_in_bin |

    We then return a weighted average of the gaps, based on the number
    of samples in each bin

    See: Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
    "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI.
    2015."""
    def __init__(self, n_bins=15): # the default num of bin is 15
        """n_bins (int): number of confidence interval bins"""
        super(ECELoss, self).__init__()

        # initialize n_bins num of bins
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)

        # take all expect the last one
        self.bin_lowers = bin_boundaries[:-1]

        # take all expect the first one
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels):
        # fetch confidence & prediciton from logits
        sigmoids = F.sigmoid(logits)
        confidences, predictions = torch.max(sigmoids, 1)
        accuracies = predictions.eq(labels)

        # print(f"accuracy: {accuracies}")

        # initialize ece tensor
        ece = torch.zeros(1, device=logits.device)
        ece_list = []

        # take all confidence bins to divide logits
        # take last 10 bins to divide logits
        # take last 5 bins to divide logits

        # Calculated |confidence - accuracy| in each bin
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()

                # print(f"accuracy_float: {accuracies[in_bin]}, {accuracies[in_bin].float()}")
                # print(f"accuracy: {accuracy_in_bin}, avg_confidence: {avg_confidence_in_bin}")

                # ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
                ece += (avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
                # ece = (avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
                # ece = ece.cpu().item()
                # ece_list.append(ece)
        
        # print('-'*20)

        return ece.cpu().item()
        # return ece_list


class _Loss(nn.Module):
    reduction: str

    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super(_Loss, self).__init__()
        if size_average is not None or reduce is not None:
            self.reduction: str = Reduction.legacy_get_string(size_average, reduce)
        else:
            self.reduction = reduction


def binary_cross_entropy_with_logits(input, target, weight=None, reduction='mean', pos_weight=None, size_average=None, reduce=None):
    if size_average is not None or reduce is not None:
        reduction = Reduction.legacy_get_string(size_average, reduce)
    if not (target.size() == input.size()):
        raise ValueError("Target size ({}) must be the same as input size ({})".format(target.size(), input.size()))

    max_val = (-input).clamp(min=0)

    if pos_weight is None:
        loss = input - input * target + max_val + ((-max_val).exp() + (-input - max_val).exp()).log()
    else:
        log_weight = 1 + (pos_weight - 1) * target
        loss = input - input * target + log_weight * (max_val + ((-max_val).exp() + (-input - max_val).exp()).log())

    if weight is not None:
        loss = loss * weight

    if reduction == 'none':
        return loss
    elif reduction == 'mean':
        return loss.mean()
    else:
        return loss.sum()


class BCEWithLogitsLoss_(_Loss):
    reduction_list = ['sum', 'mean', 'none']
    def __init__(self, weight: Optional[Tensor] = None, reduction: str = 'mean', pos_weight: Optional[Tensor] = None):
        super().__init__()
        if reduction not in self.reduction_list:
            raise ValueError(f'Unsupported reduction {reduction}')
        self.weight = weight
        self.pos_weight = pos_weight
        self.reduction = reduction

    def forward(self, input, target):
        return binary_cross_entropy_with_logits(input, target, self.weight, self.reduction, self.pos_weight)


def sigmoid_focal_loss(inputs: torch.Tensor,
                       targets: torch.Tensor,
                       alpha: float = -1,
                       gamma: float = 2,
                       reduction: str = "none") -> torch.Tensor:
    inputs = inputs.float()
    targets = targets.float()
    p = torch.sigmoid(inputs) # probability
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()

    return loss


def scalewise_sigmoid_focal_loss(inputs: torch.Tensor,
                                 targets: torch.Tensor,

                                 small_index: torch.Tensor,
                                 midle_index: torch.Tensor,
                                 large_index: torch.Tensor,

                                 num_s: int,
                                 num_l: int,

                                 alpha: float = 0.75,
                                 gamma: float = 4,
                                
                                 reduction: str = "none") -> torch.Tensor:
    """
    Small Object Detection with Scale Adaptive Balance Mechanism ICSP 2020
    """
    inputs = inputs.float()
    targets = targets.float()
    p = torch.sigmoid(inputs) # probability
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")

    loss_small, loss_midle, loss_large = ce_loss[small_index], ce_loss[midle_index], ce_loss[large_index]
    targ_small, targ_midle, targ_large = targets[small_index], targets[midle_index], targets[large_index]
    prob_small, prob_midle, prob_large = p[small_index], p[midle_index], p[large_index]

    # p_t = p * targets + (1 - p) * (1 - targets)
    # loss = ce_loss * ((1 - p_t) ** gamma)

    lamda_s = (num_s / (num_s + num_l))
    p_t_s = prob_small * targ_small + (1 - prob_small) * (1 - targ_small)
    loss_small = loss_small * ((1 - p_t_s) ** (gamma+lamda_s)) * lamda_s

    lamda_l = (num_l / (num_s + num_l))
    p_t_l = prob_large * targ_large + (1 - prob_large) * (1 - targ_large)
    loss_large = loss_large * ((1 - p_t_l) ** (gamma+lamda_l)) * lamda_l

    p_t_m = prob_midle * targ_midle + (1 - prob_midle) * (1 - targ_midle)
    loss_midle = ce_loss * ((1 - p_t_m) ** gamma)

    if alpha >= 0:
        # alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        # loss = alpha_t * loss

        alpha_t_s = alpha * targ_small + (1 - alpha) * (1 - targ_small)
        alpha_t_m = alpha * targ_midle + (1 - alpha) * (1 - targ_midle)
        alpha_t_l = alpha * targ_large + (1 - alpha) * (1 - targ_large)

        loss_small = alpha_t_s * loss_small
        loss_midle = alpha_t_m * loss_midle
        loss_large = alpha_t_l * loss_large

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss_small.sum() + loss_midle.sum() + loss_large.sum() # loss.sum()

    return loss


def sca_weighting(num_small, num_large, loss_small, loss_midle, loss_large, current_epoch, total_epoch):
    """
    Small Object Detection with Scale Adaptive Balance Mechanism ICSP 2020
    """
    def exp_decrease(coef, epoch, total):
        A = (1 - 1/coef) / (1 - math.exp(-2))
        ratio = A * (1 - math.exp(-2/total * (total-epoch))) + 1/coef
        return coef * ratio
    
    def exp_increase(coef, epoch, total):
        ratio = (1 / coef - 1) * (1 - np.exp(-3/total * epoch)) + 1
        return coef * ratio
    
    if num_large != 0:
        if num_large / (num_large+num_small) > 1:
            ratio = exp_decrease(num_large / (num_large+num_small), current_epoch, total_epoch)
        elif num_large / (num_large+num_small) < 1:
            ratio = exp_increase(num_large / (num_large+num_small), current_epoch, total_epoch)
        else: ratio = 1
        loss_small = loss_small * ratio
    
    if num_small != 0:
        if num_small / (num_large+num_small) > 1:
            ratio = exp_decrease(num_small / (num_large+num_small), current_epoch, total_epoch)
        elif num_small / (num_large+num_small) < 1:
            ratio = exp_increase(num_small / (num_large+num_small), current_epoch, total_epoch)
        else: ratio = 1
        loss_large = loss_large * ratio

    loss_cls = loss_small + loss_midle + loss_large

    return loss_cls


def scb_weighting(pred_gt_index, gt_boxes, gt_point, loss_cls, epoch, total):
    """
    Scale-Balanced Loss for Object Detection Pattern Recognition 2021 BUPT
    - scb_weight = 3 * (math.log(Mi) + 1) / Mi # Mi is anchor num of current obj
    - YoloX compute loss for each anchor, but one GT may have much anchors
    - This is used for reweighting after BCEwithlogitsLoss, so we must have a loss func
    """
    loss_cls = loss_cls.sum(1)

    def exp_decrease(coef, epoch, total):
        A = (1 - 1/coef) / (1 - math.exp(-2))
        ratio = A * (1 - math.exp(-2/total * (total-epoch))) + 1/coef
        return coef * ratio
    
    def exp_increase(coef, epoch, total):
        ratio = (1 / coef - 1) * (1 - np.exp(-3/total * epoch)) + 1
        return coef * ratio
    
    def prefix_sums(old_list):
        new_list = [0]
        current_sum = 0
        for num in old_list:
            current_sum += num
            new_list.append(current_sum)
            if len(new_list) == len(old_list):
                break
        return new_list
    
    gt_point = prefix_sums(gt_point)

    # 每个批次有多张图片
    for i in range(len(pred_gt_index)):
        # 每张图片有多个目标
        for j in range(len(pred_gt_index[i])):
            if len(pred_gt_index[i][j]) != 0 and 0 < gt_boxes[i][j] <= 32**2:
                scb_weight = 3*(math.log(len(pred_gt_index[i][j]))+1)/len(pred_gt_index[i][j])

                for k in pred_gt_index[i][j]:
                    loss_cls[k+gt_point[i]] = loss_cls[k+gt_point[i]] * exp_decrease(scb_weight, epoch, total)
    
    return loss_cls


def scm_weighting_v1(logits, labels, areas, loss_cls):
    """
    Firstly, setup maximum margin;
    Then (margin - min) / (max - min)
    Then find out medium value
    Then ratio = progress / medium
    Finally, loss / ratio
    """
    # 将二维张量转换为一维张量
    loss_cls = loss_cls.sum(1)
    
    # initial some parameters
    progress, max_margin, min_m = [], [], 0

    # 获取每个样本的边距值和可能达到
    # 的最大边距值, 面积为0则跳过
    for i in range(len(areas)):
        if areas[i] != 0: # gt_areas莫名其妙会出现0, 也是绝了
            # 计算每个目标对应尺度的最大边距
            x = areas[i].item() / 32**2
            max_m = 4 * math.log(2*x+1)
            max_margin.append(max_m)

            # 获取正确与错误logit值计算边距
            correct_logit = logits[i][int(labels[i])].item()
            current_obj_logits = logits[i].detach().numpy().tolist()
            current_obj_logits.pop(int(labels[i]))
            max_wrong_logit = max(current_obj_logits)
            margin = correct_logit - max_wrong_logit

            # if margin > max_m:
            #     print(f"margin {round(margin, 2)}")
            #     print(f"max_m: {round(max_m, 2)}")
            #     print(f"area: {round(math.sqrt(areas[i].item()), 2)}")

            # if margin < 0:
            #     print(f"margin: {round(margin, 2)}")
            #     print(f"max_m: {round(max_m, 2)}")
            #     print(f"area: {round(math.sqrt(areas[i].item()), 2)}")
            
            # print("\n")

            # 通过边距值计算当前目标拟合进度
            progress.append(margin)

            # 更新最小边距值
            if min_m > margin:
                min_m = margin
        else:
            progress.append('occupy')
            max_margin.append('occupy')
    
    # min减0.5防止除以0
    min_m = min_m - 0.5

    # print(f"min: {min_m}")

    # 计算每个非占位符样本的进度
    # 将最小值减去1, 用以计算进度
    # (margin-min_m) / (max_m-min_m)
    for i in range(len(progress)):
        if progress[i] != 'occupy':
            progress[i] = (progress[i]-min_m) / (max_margin[i]-min_m)

    # 使用最大值进行归一化失败的原因是给训练
    # 进度正常的图片加权过多, 因为不能用训练
    # 最好的图片作为标杆进行加权, 因此改为使
    # 用训练进度中的中位值作为标杆进行加权
    median = statistics.median([i for i in progress if i != 'occupy'])

    # print(f"median: {median}")
    
    # 由于存在占位符, 必须手动做归一化
    # 每个比例值除以最大比例进行归一化
    for i in range(len(progress)):
        if progress[i] != 'occupy':
            progress[i] = progress[i] / median
    
    obj_range = []
    
    # 将占位符替换成1, 然后再给损失函数加权
    # 只对进度落后的样本进行加权, 不对进度
    # 较好的样本进行加权
    for i in range(len(progress)):
        if progress[i] != 'occupy' and progress[i] < 1: 
            loss_cls[i] = loss_cls[i] / progress[i]
            obj_range.append(round(1/progress[i], 2))

    def count_values_in_ranges(data, ranges):
        # 初始化一个字典来存储结果
        result = {f"{range_start}-{range_end}": 0 for range_start, range_end in ranges}
        
        # 遍历数据列表，统计每个范围内的数量
        for value in data:
            for range_start, range_end in ranges:
                if range_start <= value < range_end:
                    result[f"{range_start}-{range_end}"] += 1
                    break  # 一旦找到匹配的范围就跳出内层循环
        
        return result
    
    # 定义范围
    ranges = [(0, 5), (5, 10), (10, 20), (20, 30), (30, 1000)]

    # 调用函数并打印结果
    print(f"range: {count_values_in_ranges(obj_range, ranges)}")
    
    print("\n")
    
    return loss_cls

def scm_weighting_v2(logits, labels, loss_cls, scale_index, margin_list, margin_max, margin_min):
    """
    Firstly, setup maximum margin;
    Then (margin - min) / (max - min)
    Then find out medium value
    Then ratio = progress / medium
    Finally, loss / ratio
    """
    # 将二维张量转换为一维张量
    loss_cls = loss_cls.sum(1)

    small_range, midle_range, large_range = [], [], []
    
    for i in range(len(scale_index)):
        if scale_index[i].sum() != 0:
            # initial some parameters
            progress, save_num, max_r = {}, 100, 2.5

            # 将索引张量转化为列表
            index = scale_index[i].nonzero().squeeze().tolist()
            if isinstance(index, int): index = [index]

            for j in index:
                # 获取正确与错误logit值计算边距
                correct_logit = logits[j][int(labels[j])].item()
                current_logit = logits[j].detach().numpy().tolist()
                current_logit.pop(int(labels[j]))
                max_wrong_logit = max(current_logit)

                # 更新该尺度边距列表
                progress[j] = correct_logit - max_wrong_logit
            
            if len(progress) > 0:
                # 更新最大值
                if max(list(progress.values())) > margin_max[i]:
                    margin_max[i] = max(list(progress.values()))
                
                # 更新最小值
                margin_min[i] = min(list(progress.values())) - 0.1
    
            # 计算每个非占位符样本的进度
            for j in index:
                progress[j] = (progress[j]-margin_min[i]) / (margin_max[i]-margin_min[i])

            # 更新边距列表用于超小批量训练
            if len(progress) >= save_num:
                margin_list[i] = list(progress.values())
            else:
                margin_list[i].extend(list(progress.values()))
                margin_list[i] = margin_list[i][-save_num:]
            
            # 不能用训练最好的图片作为锚点进行加权
            # if i == 0: median = 7/16
            # elif i == 1: median = 8/16
            # elif i == 2: median = 9/16
            
            median = sorted(margin_list[i], reverse=True)[int(len(margin_list[i]) * 0.5)]
        
            # 由于存在占位符, 必须手动做归一化
            # 每个比例值除以最大比例进行归一化
            for j in index:
                progress[j] = progress[j] / median
                if progress[j] < 1:
                    # if 1 / progress[j] > max_r: progress[j] = max_r
                    if i == 0: small_range.append(round(1/progress[j],2))
                    if i == 1: midle_range.append(round(1/progress[j],2))
                    if i == 2: large_range.append(round(1/progress[j],2))
                    loss_cls[j] = loss_cls[j] / progress[j]

    def count_values_in_ranges(data, ranges):
        # 初始化一个字典来存储结果
        result = {f"{range_start}-{range_end}": 0 for range_start, range_end in ranges}
        
        # 遍历数据列表，统计每个范围内的数量
        for value in data:
            for range_start, range_end in ranges:
                if range_start <= value < range_end:
                    result[f"{range_start}-{range_end}"] += 1
                    break  # 一旦找到匹配的范围就跳出内层循环
        
        return result
    
    # 定义范围
    ranges = [(0, 5), (5, 10), (10, 20), (20, 30), (30, 1000)]

    # 调用函数并打印结果
    print(f"small: {count_values_in_ranges(small_range, ranges)}")
    print(f"midle: {count_values_in_ranges(midle_range, ranges)}")
    print(f"large: {count_values_in_ranges(large_range, ranges)}")
    print(f"total ")
    
    print("\n")
    
    return loss_cls, margin_list, margin_max, margin_min

def scm_focus_obj(logits, fg_mask, scale_index, margin_list, margin_max, margin_min, init_mgn):
    # initialize some parameters
    save_num, logit_min = 100, -4

    # 当本轮该尺度目标超过阈值时直接使用本轮目标
    if scale_index[0].sum() >= save_num: margin_list[0] = []
    if scale_index[1].sum() >= save_num: margin_list[1] = []
    if scale_index[2].sum() >= save_num: margin_list[2] = []

    # extract a new logits tensor
    logits_ = logits[fg_mask]

    # convert bool tensor to list
    fg_mask_ = fg_mask.nonzero().squeeze().tolist()
    if isinstance(fg_mask_, int): fg_mask_ = [fg_mask_]

    # 更新边距列表和最大值列表
    for i in range(len(scale_index)):
        if scale_index[i].sum() != 0:
            # 将索引张量转化为列表
            index = scale_index[i].nonzero().squeeze().tolist()
            if isinstance(index, int): index = [index]

            # 初始化当前批次当前尺度的占位字典
            margin_occ = {}

            # 更新该尺度边距列表
            for j in index: 
                if logits_[j][0].item() > logit_min:
                    # 更新该尺度边距列表
                    margin_list[i].append(logits_[j][0].item())

                    # 更新占位字典
                    margin_occ[j] = True
                else:
                    margin_occ[j] = False
            
            # 更新最大值列表
            if max(margin_list[i]) > margin_max[i]:
                margin_max[i] = max(margin_list[i])
            
            # 边距最小值逐批次计算而非更新全局值
            # 逐批次计算比更新全局值的优化强度大
            margin_min[i] = min(margin_list[i])
            
            # 当本轮该尺度目标少于阈值时保留部分历史数据
            if scale_index[i].sum() < save_num: margin_list[i] = margin_list[i][-save_num:]

            margin_mean = statistics.mean(margin_list[i])

            print(f"{i}: mean: {round(margin_mean, 2)}, max: {round(margin_max[i], 2)}, min: {round(margin_min[i], 2)}, \
                        length: {len(margin_list[i])}, larger than 0: {len([i for i in margin_list[i] if i > 0])}")
            
            counter, counter_adding, counter_no_add = 0, 0, 0

            # 计算每个目标的偏离度
            for j in index:
                if margin_occ[j]:
                    counter += 1
                    delta_margin = (margin_mean-logits_[j][0].item()) / (margin_max[i] - margin_min[i])
                    # logits_[j][0] -= init_mgn[i] if delta_margin < 0 else (1 + delta_margin) * init_mgn[i]
                    if delta_margin > 0: 
                        counter_adding += 1 # print(f"delta_margin: {(1 + delta_margin) * init_mgn[i]}")
                    else:
                        counter_no_add += 1
            
            print(f"total: {counter}, smaller than mean: {counter_adding}, larger than mean: {counter_no_add}")

    print("\n")

    # assign new tensor back into old tensor
    counter = 0
    for j in fg_mask_:
        logits[j] = logits_[counter]
        counter += 1
    
    return logits, margin_list, margin_max, margin_min

def scm_focus_cls(logits, labels, scale_index, margin_list, margin_max, margin_min, init_margin):
    # initialize some parameters
    save_num = 100

    # 当本轮该尺度目标超过阈值时直接使用本轮目标
    if scale_index[0].sum() >= save_num: margin_list[0] = []
    if scale_index[1].sum() >= save_num: margin_list[1] = []
    if scale_index[2].sum() >= save_num: margin_list[2] = []

    # 更新边距列表和最大值列表
    for i in range(len(scale_index)):
        if scale_index[i].sum() != 0:
            # 将索引张量转化为列表
            index = scale_index[i].nonzero().squeeze().tolist()
            if isinstance(index, int): index = [index]

            # 更新该尺度边距列表
            for j in index: 
                margin_list[i].append(logits[j][int(labels[j])].item())
            
            # 更新最大值列表
            if max(margin_list[i]) > margin_max[i]:
                margin_max[i] = max(margin_list[i])
            
            if min(margin_list[i]) < margin_min[i]:
                margin_min[i] = min(margin_list[i])
            
            # 当本轮该尺度目标少于阈值时保留部分历史数据
            if scale_index[i].sum() < save_num: margin_list[i] = margin_list[i][-save_num:]

            margin_mean = statistics.mean(margin_list[i])
            
            # 计算每个目标的偏离度
            for j in index:
                delta_margin = (margin_mean-logits[j][int(labels[j])]) / (margin_max[i] - margin_min[i])
                margin = init_margin[i] if delta_margin < 0 else (1 + delta_margin) * init_margin[i]
                logits[j][int(labels[j])] -= margin
    
    return logits, margin_list, margin_max, margin_min

def adding_margin_obj(logits, fg_mask, scale_index, init_mgn):
    # extract a new logits tensor
    logits_, counter = logits[fg_mask], 0

    # convert bool tensor to list
    fg_mask_ = fg_mask.nonzero().squeeze().tolist()
    if isinstance(fg_mask_, int): fg_mask_ = [fg_mask_]

    def f(x, c):
        return x + math.log((1+math.exp(-x))**(c)-1)

    # modify the new tensor
    for i in range(len(scale_index)):
        if scale_index[i].sum() != 0:
            # index is bool tensor, therefore we have to
            # convert it to list, extract index from it
            index = scale_index[i].nonzero().squeeze().tolist()
            if isinstance(index, int): index = [index]
            for j in index:
                logit = logits_[j][0].item()
                logits_[j][0] -= f(logit, 1.5) * init_mgn[i]
    
    # assign new tensor back into old tensor
    for j in fg_mask_:
        logits[j] = logits_[counter]
        counter = counter + 1

    return logits

def adding_margin_cls(logits, labels, scale_index, init_mgn, epoch, total):

    def exp_decay(coef, epoch, total):
        A = (1 - 1/coef) / (1 - math.exp(-2))
        ratio = A * (1 - math.exp(-2/total * (total-epoch)))
        return coef * ratio
    
    for i in range(len(scale_index)):
        if scale_index[i].sum() != 0:
            # index is bool tensor, therefore we have to
            # convert it to list, extract index from it
            index = scale_index[i].nonzero().squeeze().tolist()
            if isinstance(index, int): index = [index]

            for j in index: 

                # logits[i] extract index linked logits tensor
                # int(labels[i]) extract correct label logit
                logits[j][int(labels[j])] -= init_mgn[i]

    return logits

def adding_margin_continue(logits, labels, area):
    # y = e^(coei*x) inwhich coei is negative
    coei = -0.4621 # 0.4621≈In(0.25)/3

    for i in range(logits.shape[0]):
        # construct margin with hyper-params
        x = math.sqrt(area[i] / 32**2)
        margin = math.exp(coei * x)
        logits[i][int(labels[i])] -= margin

    return logits

def adding_margin_classwise(logits, labels, gt_cls_list, max_m=0.5):
    """
    Learning Imbalanced Datasets with Label-Distribution-Aware Margin Loss NIPS 2019
    """
    m_list = 1.0 / np.sqrt(np.sqrt(gt_cls_list))
    m_list = m_list * (max_m / np.max(m_list))
    for i in range(logits.shape[0]):
        margin = m_list[int(labels[i])]
        logits[i][int(labels[i])] -= margin

    return logits

# --------------------------------------------------------------

def scale_index_num(reg_targets, gt_class_batch):
    # 取得GT中能与Pred匹配的小目标
    gt_width = reg_targets[:, 2]
    gt_height = reg_targets[:, 3]
    gt_areas = gt_width * gt_height
    
    def collect_objects(gt_areas):
        # 先储存两个间隔进列表
        scale_index = [0, 16 * 16]
        bins, input_size = 32, 512
        scale_index_tensor = []

        # 再以32*32为间隔划分尺度
        for i in range(1, int(input_size/bins+1)):
            scale_index.append((bins*i)**2)
        
        # 使用scale_index划分目标
        for i in range(len(scale_index)-1):
            scale_index_tensor.append((scale_index[i] < gt_areas) & (gt_areas < scale_index[i+1]))
        
        return scale_index_tensor

    # scale_index = collect_objects(gt_areas)
    
    small_index = gt_areas <= 32**2
    midle_index = (gt_areas > 32**2) & (gt_areas <= 96**2)
    large_index = gt_areas > 96**2

    num_small = gt_class_batch[small_index].shape[0]
    num_midle = gt_class_batch[midle_index].shape[0]
    num_large = gt_class_batch[large_index].shape[0]

    return gt_areas, small_index, midle_index, large_index, num_small, num_midle, num_large

def loss_ratio(loss_iou, small_index, midle_index, large_index):
    # 计算损失值中小目标比例(注意: 标量的shape为空元组)
    if loss_iou.shape[0] == 0: # loss_iou为空张量时
        ratio_small = 0.0
    elif loss_iou.shape[0] == 1: # loss_iou只有1个值时
        if small_index[0] == True:
            ratio_small = 1.0
        else:
            ratio_small = 0.0
    else: # loss_iou为多值向量时
        ratio_small = loss_iou[small_index].sum() / loss_iou.sum()
        ratio_midle = loss_iou[midle_index].sum() / loss_iou.sum()
        ratio_large = loss_iou[large_index].sum() / loss_iou.sum()
    ratio_scale = [ratio_small, ratio_midle, ratio_large]
    return ratio_scale

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

def batch_avg_margin_standard(cls_preds_view, gt_class_batch, small_index, midle_index, large_index):
    # 计算当前批次每个尺度目标的平均margin
    small_logits = cls_preds_view[small_index].detach().cpu()
    small_labels = gt_class_batch[small_index].detach().cpu()
    margin_small_avg = compute_margin(small_logits, small_labels)

    midle_logits = cls_preds_view[midle_index].detach().cpu()
    midle_labels = gt_class_batch[midle_index].detach().cpu()
    margin_midle_avg = compute_margin(midle_logits, midle_labels)
    
    large_logits = cls_preds_view[large_index].detach().cpu()
    large_labels = gt_class_batch[large_index].detach().cpu()
    margin_large_avg = compute_margin(large_logits, large_labels)

    total_logits = cls_preds_view.detach().cpu()
    total_labels = gt_class_batch.detach().cpu()
    margin_total_avg = compute_margin(total_logits, total_labels)

    return margin_small_avg, margin_midle_avg, margin_large_avg, margin_total_avg

def batch_avg_margin_adaptive(cls_preds_view, gt_class_batch, scale_index):
    # 计算当前批次每个尺度目标的平均margin
    margin_list = []
    for i in range(len(scale_index)):
        scale_logits = cls_preds_view[scale_index[i]].detach().cpu()
        scale_labels = gt_class_batch[scale_index[i]].detach().cpu()
        margin_list.append(compute_margin(scale_logits, scale_labels))

    return margin_list

def loss_distributer(gt_areas, loss_cls):
    # divide every cls loss into bins based on object size, 
    # bin value is 8^2, finally we form a loss scale curve
    loss_scale_list = []
    for i in range(len(gt_areas)):
        temp = int(gt_areas[i].item() // 8**2) * 8**2
        loss_scale_list.append({temp: loss_cls[i].sum().item()})
    return loss_scale_list

def scale_wise_cls_loss(loss_cls_small, loss_cls_midle, loss_cls_large, num_small, num_midle, num_large):
    loss_cls_scale = {}
    loss_cls_scale['small'] = (num_small, loss_cls_small.detach().to('cpu').item())
    loss_cls_scale['midle'] = (num_midle, loss_cls_midle.detach().to('cpu').item())
    loss_cls_scale['large'] = (num_large, loss_cls_large.detach().to('cpu').item())
    return loss_cls_scale

def scm_extra_strength(loss_cls, cls_preds_copy, cls_targets, small_index):
    # 计算原始逻辑张量的损失值
    criterion = nn.BCEWithLogitsLoss(reduction="none")
    loss_ori = criterion(cls_preds_copy, cls_targets)

    # 将损失值张量降一个维度
    loss_cls, loss_ori, extra_list = loss_cls.sum(1), loss_ori.sum(1), []
    
    # 将张量放到CPU并列表化
    origin = loss_ori[small_index].detach().cpu().tolist()
    adding = loss_cls[small_index].detach().cpu().tolist()

    for i in range(len(adding)):
        extra = (adding[i]-origin[i])/origin[i]
        extra_list.append(extra if extra > 0 else 0)
    
    return sum(extra_list)/len(extra_list) if len(extra_list) != 0 else 1

def scb_extra_strength(pred_gt_index, gt_boxes):
    scb_weight = []

    # 每个批次有多张图片
    for i in range(len(pred_gt_index)):
        # 每张图片有多个目标
        for j in range(len(pred_gt_index[i])):
            if len(pred_gt_index[i][j]) != 0 and 0 < gt_boxes[i][j] <= 32**2:
                scb_weight_ = 3*(math.log(len(pred_gt_index[i][j]))+1)/len(pred_gt_index[i][j])
                scb_weight.append(scb_weight_ - 1)
    
    return sum(scb_weight)/len(scb_weight) if len(scb_weight) != 0 else 1

def sca_extra_strength(num_small, num_large):
    return num_large / num_small - 1 if num_small != 0 else 1

# --------------------------------------------------------------

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
        self.ECELoss = ECELoss()
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

    def forward(self, xin, labels=None, imgs=None, cls_list=None, cls_max=None, cls_min=None, obj_list=None, obj_max=None, obj_min=None, current_epoch=None, total_epoch=None):
        outputs = []
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
                output = torch.cat([reg_output, obj_output.sigmoid(), cls_output.sigmoid()], 1)

            outputs.append(output)

        if self.training:
            return self.get_losses(imgs, x_shifts, y_shifts, expanded_strides, labels, torch.cat(outputs, 1), origin_preds, dtype=xin[0].dtype, 
                                   cls_list=cls_list, cls_max=cls_max, cls_min=cls_min, obj_list=obj_list, obj_max=obj_max, obj_min=obj_min, 
                                   current_epoch=current_epoch, total_epoch=total_epoch)
        else:
            self.hw = [x.shape[-2:] for x in outputs]
            outputs = torch.cat([x.flatten(start_dim=2) for x in outputs], dim=2).permute(0, 2, 1) # [batch, n_anchors_all, 85]
            if self.decode_in_inference: # 验证时从这里返回结果,不从下面的else返回
                return self.decode_outputs(outputs, dtype=xin[0].type())
            else:
                return outputs

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
        grids, strides = [], []
        
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

    def get_losses(self, imgs, x_shifts, y_shifts, expanded_strides, labels, outputs, origin_preds, dtype, cls_list, cls_max, cls_min, obj_list, obj_max, obj_min, current_epoch, total_epoch):
        bbox_preds = outputs[:, :, :4] # [batch, n_anchors_all, 4]
        obj_preds = outputs[:, :, 4].unsqueeze(-1)  # [batch, n_anchors_all, 1]
        cls_preds = outputs[:, :, 5:] # [batch, n_anchors_all, n_cls]
        
        # calculate targets, this labels shape like [bs, 120, 5]
        # 120即一张图片中最多存在120个目标, 但实际上一般只有几个目标
        # 因此先前的代码会用零行将张量填补到120行, 5即每行为[cls, xcen, ycen, w, h]
        nlabel = (labels.sum(dim=2) > 0).sum(dim=1) # number of objects
        
        total_num_anchors = outputs.shape[1]
        x_shifts = torch.cat(x_shifts, 1) # [1, n_anchors_all]
        y_shifts = torch.cat(y_shifts, 1) # [1, n_anchors_all]
        expanded_strides = torch.cat(expanded_strides, 1)
        if self.use_l1: # self.use_l1 = False
            origin_preds = torch.cat(origin_preds, 1)

        cls_targets = []
        reg_targets = []
        l1_targets = []
        obj_targets = []
        fg_masks = []

        num_fg = 0.0
        num_gts = 0.0
        gt_class_batch = []

        # scale balance loss code
        pred_gt_index = []
        pred_gt_boxes = []
        pred_gt_point = []

        for batch_idx in range(outputs.shape[0]): # 遍历每张图片
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
                    (gt_matched_classes,fg_mask,pred_ious_this_matching,matched_gt_inds,num_fg_img,pred_with_same_gt,) = self.get_assignments(  # noqa
                     True, batch_idx, num_gt, total_num_anchors, gt_bboxes_per_image, gt_classes, bboxes_preds_per_image, expanded_strides,
                     x_shifts, y_shifts, cls_preds, bbox_preds, obj_preds, labels, imgs)
                except RuntimeError:
                    logger.error("OOM RuntimeError is raised due to the huge memory cost during label assignment. CPU mode is applied in this batch.")
                    logger.error("If you want to avoid this issue,try to reduce the batch size or image size.")
                    torch.cuda.empty_cache()
                    (gt_matched_classes, fg_mask, pred_ious_this_matching, matched_gt_inds, num_fg_img, pred_with_same_gt,) = self.get_assignments(
                        False,batch_idx,num_gt,total_num_anchors,gt_bboxes_per_image,gt_classes,bboxes_preds_per_image,expanded_strides,
                        x_shifts,y_shifts,cls_preds,bbox_preds,obj_preds,labels,imgs,"cpu") # noqa
                    
                    # 判断是否存在无目标图片
                    # if gt_matched_classes == 0:
                    #     num_gt = num_gt - int(nlabel[batch_idx])
                    #     temp.append(batch_idx)
                    #     del gt_matched_classes, fg_mask, pred_ious_this_matching, matched_gt_inds, num_fg_img
                    #     continue
                    
                torch.cuda.empty_cache()
                num_fg += num_fg_img

                cls_target = F.one_hot(gt_matched_classes.to(torch.int64), self.num_classes) * pred_ious_this_matching.unsqueeze(-1)
                obj_target = fg_mask.unsqueeze(-1)
                reg_target = gt_bboxes_per_image[matched_gt_inds]
                gt_class_batch.extend(list(gt_matched_classes.cpu().numpy()))

                if self.use_l1:
                    l1_target = self.get_l1_target(outputs.new_zeros((num_fg_img, 4)),
                                                   gt_bboxes_per_image[matched_gt_inds],
                                                   expanded_strides[0][fg_mask],
                                                   x_shifts=x_shifts[0][fg_mask],
                                                   y_shifts=y_shifts[0][fg_mask])
                
                # scale balance loss code
                pred_gt_index.append(pred_with_same_gt)
                gtbox_img = gt_bboxes_per_image[:,2]*gt_bboxes_per_image[:,3]
                pred_gt_boxes.append(gtbox_img.cpu().tolist())
                pred_gt_point.append(matched_gt_inds.shape[0])
            
            # else结束--------------------

            cls_targets.append(cls_target)
            reg_targets.append(reg_target)
            obj_targets.append(obj_target.to(dtype))
            fg_masks.append(fg_mask)
            if self.use_l1: l1_targets.append(l1_target)
            
        # for循环结束---------------------------
        
        # 去除bbox_preds、obj_preds、cls_preds、origin_preds中无法分配标签的图片
        # if temp != []: # 当确实存在必须去除的图片时
        #     logger.error("Images which cannot assignment label: {}".format(len(temp)))
        #     temp.sort()
        #     if len(temp) == 1:
        #         bbox_preds = torch.cat((bbox_preds[:temp[0]], bbox_preds[temp[0]+1:]))
        #         obj_preds = torch.cat((obj_preds[:temp[0]], obj_preds[temp[0]+1:]))
        #         cls_preds = torch.cat((cls_preds[:temp[0]], cls_preds[temp[0]+1:]))
        #     else:
        #         bbox_preds = del_image(bbox_preds, temp)
        #         obj_preds = del_image(obj_preds, temp)
        #         cls_preds = del_image(cls_preds, temp)
        # del temp
        
        # 拼接批量中各个图片GT值成一个张量
        cls_targets = torch.cat(cls_targets, 0) # 
        reg_targets = torch.cat(reg_targets, 0) # xywh
        obj_targets = torch.cat(obj_targets, 0) # obj_targets.shape=[bs*5376]
        fg_masks = torch.cat(fg_masks, 0) # fg_masks.shape=[bs*334]
        if self.use_l1: l1_targets = torch.cat(l1_targets, 0)

        # cls_targets.shape = reg_targets.shape = torch.sum(fg_masks) != obj_targets.shape

        # 将gt_class_batch放入cuda
        gt_class_batch = torch.tensor(gt_class_batch).to(cls_preds.device)

        # 取得尺度向的index用于抽取损失值张量
        gt_areas, small_index, midle_index, large_index, num_small, num_midle, num_large = scale_index_num(reg_targets, gt_class_batch)

        class_margin = True
        scale_margin = False
        scale_mgnrew = False
        scale_balan  = False
        scale_adapt  = False
        scale_decay  = False

        # 计算本轮次各个损失值
        num_fg = max(num_fg, 1)

        # cls_targets与reg_targets和gt_areas拥有相同的行数, 只不过cls_targets每行20个元素(voc类别数), reg_targets每行4个元素(边框坐标)
        # cls_preds.view与reg_preds.view与fg_masks拥有相同行数, 经过fg_masks索引后cls_preds与reg_preds与cls_targets与reg_targets相同
        # obj_preds.view与obj_targets与fg_masks行数相同, 但obj_preds.view并不经过fg_masks索引, 如果想抽取尺度信息就必须经过fg_masks索引

        # --------------------------------------------------------------

        # 计算iou loss, 由于要计算不同尺度目标损失值, 因此对iou loss进行拆分操作
        # loss_iou = (self.iou_loss(bbox_preds.view(-1, 4)[fg_masks], reg_targets)).sum() / num_fg

        # 获取本轮次各尺度损失值比例
        loss_iou = self.iou_loss(bbox_preds.view(-1, 4)[fg_masks], reg_targets)
        ratio_scale = loss_ratio(loss_iou, small_index, midle_index, large_index)
        loss_iou = loss_iou.sum() / num_fg

        # --------------------------------------------------------------
        
        # 计算obj loss, obj loss是这个区域的确存在目标的置信度, 原始代码
        loss_obj = (self.bcewithlog_loss(obj_preds.view(-1, 1), obj_targets)).sum() / num_fg

        # obj_preds_view = obj_preds.view(-1, 1) # torch.sum(obj_targets) = torch.sum(fg_masks)

        # obj loss的边距函数, scale margin loss
        # obj_preds_view = adding_margin_obj(obj_preds_view, fg_masks, [small_index, midle_index, large_index], [1, 0.5, 0.25])

        # obj loss的边距函数, scale focus margin
        # obj_preds_view, obj_list, obj_max, obj_min = scm_focus_obj(obj_preds_view, fg_masks, [small_index, midle_index, large_index], obj_list, obj_max, obj_min, [1, 0.5, 0.25])

        # loss_obj = self.bcewithlog_loss(obj_preds_view, obj_targets).sum() / num_fg

        # --------------------------------------------------------------

        # 计算cls loss, cls loss是分类置信度, 即N个类别每个都给出一个置信度, 最高者为最终输出
        # loss_cls = (self.bcewithlog_loss(cls_preds.view(-1, self.num_classes)[fg_masks], cls_targets)).sum() / num_fg

        cls_preds_view = cls_preds.view(-1, self.num_classes)[fg_masks] # cls_preds_view就是logits

        cls_preds_copy = cls_preds_view.clone()

        if scale_mgnrew: # 将pred logits与gt cls复制一份备用, 防止被修改后无法获取到原始logits
            origin_cls_preds, origin_gt_class = cls_preds_view.clone().detach().cpu(), gt_class_batch.clone().detach().cpu()
        
        if scale_margin:
            # 为不同尺度的目标添加额外的边距, adding_margin()默认减去边距, 所以输入正数等于减去边距
            # 如果只给小目标使用这个公式, 完全没有任何提升, 可能是因为给过小的目标减去的边距值过小了
            cls_preds_view = adding_margin_cls(cls_preds_view, gt_class_batch, [small_index, midle_index, large_index], [0.5, 0, 0], current_epoch, total_epoch)

            # intra-scale local focus margin-based loss with mean value deviation calculation
            # cls_preds_view, cls_list = adding_margin_cls_v2(cls_preds_view, gt_class_batch, [small_index, midle_index, large_index], cls_list, [1, 0.5, 0.25])

        if class_margin: # Label Distribution Aware Margin Loss
            gt_cls_list = [3148, 2836, 4503, 3277, 4553, 2154, 8948, 4166, 9569, 2464, 2400, 5231, 2795, 2784, 35451, 3731, 3056, 3034, 2576, 2803]
            cls_preds_view = adding_margin_classwise(cls_preds_view, gt_class_batch, gt_cls_list)

        # -------------------------------

        # BCELosswithlogits loss
        loss_cls = self.bcewithlog_loss(cls_preds_view, cls_targets) # cls_preds_view.shape = [obj_num, cls_num]

        # 经过测试这些张量都拥有相同的长度
        # print(f"shape:  {loss_cls.shape}, {cls_preds_view.shape}, {gt_class_batch.shape}, {gt_areas.shape}")

        # -------------------------------

        if scale_adapt: # 简化版scale adaptive loss, 因为YoloX无法使用Focal Loss
            # 将总损失值按照尺度划分成多份 loss_cls[small_index].shape = [obj_num, cls_num], loss_cls_small.device = cuda:0
            loss_cls_small, loss_cls_midle, loss_cls_large = loss_cls[small_index].sum(), loss_cls[midle_index].sum(), loss_cls[large_index].sum()
            loss_cls = sca_weighting(num_small, num_large, loss_cls_small, loss_cls_midle, loss_cls_large, current_epoch, total_epoch)

        if scale_balan: # scale balance weighting
            loss_cls = scb_weighting(pred_gt_index, pred_gt_boxes, pred_gt_point, loss_cls, current_epoch, total_epoch)

        if scale_mgnrew: # scale margin weighting
            loss_cls = scm_weighting_v1(origin_cls_preds, origin_gt_class, gt_areas.cpu(), loss_cls)
            # loss_cls, cls_list, cls_max, cls_min = scm_weighting_v2(origin_cls_preds, origin_gt_class, loss_cls, [small_index, midle_index, large_index], cls_list, cls_max, cls_min)

        # -------------------------------

        # 将本轮次不同尺度目标的损失值与目标数组成字典方便传回给训练主函数
        # loss_cls_scale = scale_wise_cls_loss(loss_cls_small, loss_cls_midle, loss_cls_large, num_small, num_midle, num_large)
        
        # 计算本轮次各尺度目标的平均margin
        # margin_small, margin_midle, margin_large, margin_total = batch_avg_margin_standard(cls_preds_view, gt_class_batch, small_index, midle_index, large_index)
        # margin_list = batch_avg_margin_adaptive(cls_preds_view, gt_class_batch, scale_index)

        # 将所有cls loss按照目标尺度分进间隔中, 间隔为8*8
        # loss_scale_list = loss_distributer(gt_areas, loss_cls)

        # num_fg即所有存在锚框匹配的目标 | num_gts即该批次所有目标, 大于num_fg
        # loss_cls = loss_cls / num_fg
        loss_cls = loss_cls.sum() / num_fg

        # ---------------------------------
        # 计算ece loss, ece loss用于评估置信度与精度的差异, 判断是否过拟合或欠拟合
        # loss_ece_total = self.ECELoss(cls_preds_view, gt_class_batch)
        # loss_ece_small = self.ECELoss(cls_preds_view[small_index], gt_class_batch[small_index])
        # loss_ece_midle = self.ECELoss(cls_preds_view[midle_index], gt_class_batch[midle_index])
        # loss_ece_large = self.ECELoss(cls_preds_view[large_index], gt_class_batch[large_index])

        # print(f"gt_class_batch: {gt_class_batch.shape}") # [611]
        # print(f"cls_preds: {cls_preds_view.shape}") # [611, 20]
        # print(f"small: {cls_preds_view[small_index].shape}, {gt_class_batch[small_index].shape}") # [69, 20], [69]
        # print(f"midle: {cls_preds_view[midle_index].shape}, {gt_class_batch[midle_index].shape}") # [330, 20] [330]
        # print(f"large: {cls_preds_view[large_index].shape}, {gt_class_batch[large_index].shape}")

        # loss_ece_small = [cls_preds_view[small_index].to('cpu'), gt_class_batch[small_index].to('cpu')]
        # loss_ece_midle = [cls_preds_view[midle_index].to('cpu'), gt_class_batch[midle_index].to('cpu')]
        # loss_ece_large = [cls_preds_view[large_index].to('cpu'), gt_class_batch[large_index].to('cpu')]
        # loss_ece_total = [cls_preds_view.to('cpu'), gt_class_batch.to('cpu')]
        # ---------------------------------

        # 计算l1 loss, l1 loss提供一定的正则化效果防止过拟合
        if self.use_l1:
            loss_l1 = (self.l1_loss(origin_preds.view(-1, 4)[fg_masks], l1_targets)).sum() / num_fg
        else:
            loss_l1 = 0.0
        
        # 计算总损失值,加权相加
        loss = 5.0 * loss_iou + loss_obj + loss_cls + loss_l1

        # --------------------------------------------------------------
        # return loss, 5.0 * loss_iou, loss_obj, loss_cls, loss_l1, num_fg / max(num_gts, 1)
        # ---------------------------------
        return loss, 5.0 * loss_iou, loss_obj, loss_cls, loss_l1, num_fg / max(num_gts, 1), cls_list, cls_max, cls_min, obj_list, obj_max, obj_min
        # ---------------------------------
        # return loss, 5.0 * loss_iou, loss_obj, loss_cls, loss_l1, num_fg / max(num_gts, 1), ratio_scale
        # ---------------------------------
        # return loss, 5.0 * loss_iou, loss_obj, loss_cls, loss_l1, num_fg / max(num_gts, 1), loss_scale_list
        # ---------------------------------
        # return loss, 5.0 * loss_iou, loss_obj, loss_cls, loss_l1, num_fg / max(num_gts, 1), loss_cls_scale
        # ---------------------------------
        # return loss, 5.0 * loss_iou, loss_obj, loss_cls, loss_l1, num_fg / max(num_gts, 1), \
        #        [margin_small, margin_midle, margin_large, margin_total, margin_list]
        # ---------------------------------
        # return loss, 5.0 * loss_iou, loss_obj, loss_cls, loss_l1, num_fg / max(num_gts, 1), \
        # [loss_iou_small.detach().item(), loss_iou_midle.detach().item(), loss_iou_large.detach().item()], \
        # [loss_cls_small.detach().item(), loss_cls_midle.detach().item(), loss_cls_large.detach().item()], \
        # [loss_obj_small.detach().item(), loss_obj_midle.detach().item(), loss_obj_large.detach().item()], \
    
    def get_l1_target(self, l1_target, gt, stride, x_shifts, y_shifts, eps=1e-8):
        l1_target[:, 0] = gt[:, 0] / stride - x_shifts
        l1_target[:, 1] = gt[:, 1] / stride - y_shifts
        l1_target[:, 2] = torch.log(gt[:, 2] / stride + eps)
        l1_target[:, 3] = torch.log(gt[:, 3] / stride + eps)
        return l1_target
    
    @torch.no_grad() # 处理每张图片拥有的目标
    def get_assignments(self, flag, batch_idx, num_gt, total_num_anchors, gt_bboxes_per_image, gt_classes, bboxes_preds_per_image,
        expanded_strides, x_shifts, y_shifts, cls_preds, bbox_preds, obj_preds, labels, imgs, mode="gpu"):
        
        if mode == "cpu":
            print("------------CPU Mode for This Batch-------------")
            gt_bboxes_per_image = gt_bboxes_per_image.cpu().float()
            bboxes_preds_per_image = bboxes_preds_per_image.cpu().float()
            gt_classes = gt_classes.cpu().float()
            expanded_strides = expanded_strides.cpu().float()
            x_shifts = x_shifts.cpu()
            y_shifts = y_shifts.cpu()

        fg_mask, is_in_boxes_and_center = self.get_in_boxes_info(gt_bboxes_per_image,expanded_strides,x_shifts,y_shifts,total_num_anchors,num_gt)
        
        # 判断是否需要检测有无正样本点
        # if flag == False:
        #     count = 0
        #     for i in range(len(fg_mask)):
        #         if fg_mask[i] == True:
        #             count = 1
        #     if count != 1:
        #         return 0, 0, 0, 0, 0

        # fg_mask 存在目标的网格点位, 当前input size是512, 
        # fg_mask是5376, 是一个固定值, 因此bboxes_preds_per_image
        # 也是一个固定值, 即整张图片每个网格点都预测一个目标
        bboxes_preds_per_image = bboxes_preds_per_image[fg_mask]
        
        cls_preds_ = cls_preds[batch_idx][fg_mask]
        obj_preds_ = obj_preds[batch_idx][fg_mask]

        # 经过筛选的存在目标的区域
        num_in_boxes_anchor = bboxes_preds_per_image.shape[0]

        if mode == "cpu":
            gt_bboxes_per_image = gt_bboxes_per_image.cpu()
            bboxes_preds_per_image = bboxes_preds_per_image.cpu()
        
        # gt_bboxes_per_image.shape=[obj_num, 4]即每张图片拥有的目标的坐标
        # pair_wise_ious.shape=[obj_num, bbox_pred]即bboxes_preds_per_image
        # 经过筛选后留下存在目标的网格的bbox_pred, 然后每个目标都对所有的bbox_pred
        # 计算一遍iou, 只要经过阈值过滤就是每个目标能匹配的锚框数量
        pair_wise_ious = bboxes_iou(gt_bboxes_per_image, bboxes_preds_per_image, False)
        
        gt_cls_per_image = (F.one_hot(gt_classes.to(torch.int64), self.num_classes).float().unsqueeze(1).repeat(1, num_in_boxes_anchor, 1))
        pair_wise_ious_loss = -torch.log(pair_wise_ious + 1e-8)
        
        if mode == "cpu":
            cls_preds_, obj_preds_ = cls_preds_.cpu(), obj_preds_.cpu()

        with torch.cuda.amp.autocast(enabled=False):
            cls_preds_ = (cls_preds_.float().unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_() * obj_preds_.float().unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_())
            pair_wise_cls_loss = F.binary_cross_entropy(cls_preds_.sqrt_(), gt_cls_per_image, reduction="none").sum(-1)
        
        del cls_preds_

        cost = (pair_wise_cls_loss + 3.0 * pair_wise_ious_loss + 100000.0 * (~is_in_boxes_and_center))
        
        (num_fg,gt_matched_classes,pred_ious_this_matching,matched_gt_inds,pred_with_same_gt) = self.dynamic_k_matching(cost,pair_wise_ious,gt_classes,num_gt,fg_mask)
        
        del pair_wise_cls_loss, cost, pair_wise_ious, pair_wise_ious_loss

        if mode == "cpu":
            gt_matched_classes = gt_matched_classes.cuda()
            fg_mask = fg_mask.cuda()
            pred_ious_this_matching = pred_ious_this_matching.cuda()
            matched_gt_inds = matched_gt_inds.cuda()

        return (gt_matched_classes, fg_mask, pred_ious_this_matching, matched_gt_inds, num_fg, pred_with_same_gt)

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
        n_candidate_k = min(10, ious_in_boxes_matrix.size(1)) # 设定每行保留的iou数量
        topk_ious, _ = torch.topk(ious_in_boxes_matrix, n_candidate_k, dim=1)
        dynamic_ks = torch.clamp(topk_ious.sum(1).int(), min=1)
        dynamic_ks = dynamic_ks.tolist() # dynamic_ks即每个GT需要匹配的锚框数量的列表
        
        for gt_idx in range(num_gt):
            _, pos_idx = torch.topk(cost[gt_idx], k=dynamic_ks[gt_idx], largest=False)
            matching_matrix[gt_idx][pos_idx] = 1 # matching_matrix即总共需要的锚框数

        del topk_ious, pos_idx, dynamic_ks

        # 过滤共用的Pred, 先将矩阵按GT向相加, 大于1的网格点即出现共用
        anchor_matching_gt = matching_matrix.sum(0)
        if (anchor_matching_gt > 1).sum() > 0:
            _, cost_argmin = torch.min(cost[:, anchor_matching_gt > 1], dim=0)
            matching_matrix[:, anchor_matching_gt > 1] *= 0
            matching_matrix[cost_argmin, anchor_matching_gt > 1] = 1
        
        # 这里虽然经过重合过滤, 但还是可能某个网格点可以匹配多个GT
        fg_mask_inboxes = matching_matrix.sum(0) > 0
        
        # 存在GT匹配的锚框数量, 这个值会返回给get_losses()
        # 但我想要每个GT对应的Pred数量
        num_fg = fg_mask_inboxes.sum().item()

        # fg_mask即整张图片拥有能匹配GT的Pred的网格点
        fg_mask[fg_mask.clone()] = fg_mask_inboxes

        matched_gt_inds = matching_matrix[:, fg_mask_inboxes].argmax(0)

        # scale balance loss code
        scale_balan = True

        if scale_balan:
            matching_copy, grid_indx = matching_matrix[:, fg_mask_inboxes], []
            for i in range(len(matching_copy)):
                pred_index = torch.nonzero((matching_copy[i]!=0), as_tuple=False).squeeze().cpu().tolist()
                grid_indx.append([pred_index]) if isinstance(pred_index, int) else grid_indx.append(pred_index)
        else:
            grid_indx = []
        
        gt_matched_classes = gt_classes[matched_gt_inds]

        pred_ious_this_matching = (matching_matrix * pair_wise_ious).sum(0)[fg_mask_inboxes]
        
        return num_fg, gt_matched_classes, pred_ious_this_matching, matched_gt_inds, grid_indx
    
    
def del_image(a, temp):
    tmp = [] # 创建1个空列表
    tmp.append(a[:temp[0]]) # 首先取得0到第1个数间的张量
    for i in range(1, len(temp)-1): # 然后取得中间各个间隔内的张量
        tmp.append(a[temp[i-1]+1: temp[i]])
    tmp.append(a[temp[len(temp)-2]+1: a.shape[0]-1]) # 最后取得倒数第2个数到最后1个数间的张量
    a = torch.cat(tmp, 0) # 在维度0拼接这些张量
    return a


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