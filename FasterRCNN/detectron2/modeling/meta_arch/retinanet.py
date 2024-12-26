# Copyright (c) Facebook, Inc. and its affiliates.

import math
import copy
import torch
import logging
import numpy as np

from torch import Tensor, nn
from torch.nn import functional as F

from typing import List, Tuple
from fvcore.nn import sigmoid_focal_loss_jit

from detectron2.config import configurable
from detectron2.utils.events import get_event_storage
from detectron2.structures import Boxes, ImageList, Instances, pairwise_iou
from detectron2.layers import CycleBatchNormList, ShapeSpec, batched_nms, cat, get_norm

from ..matcher import Matcher
from .build import META_ARCH_REGISTRY
from ..backbone import Backbone, build_backbone
from ..anchor_generator import build_anchor_generator
from .dense_detector import DenseDetector, permute_to_N_HWA_K
from ..box_regression import Box2BoxTransform, _dense_box_regression_loss

__all__ = ["RetinaNet"]


logger = logging.getLogger(__name__)


@META_ARCH_REGISTRY.register()
class RetinaNet(DenseDetector):
    """Implement RetinaNet in :paper:`RetinaNet`."""
    @configurable
    def __init__(self, *,
                 backbone: Backbone,
                 head: nn.Module,
                 head_in_features,
                 anchor_generator,
                 box2box_transform,
                 anchor_matcher,
                 num_classes,
                 focal_loss_alpha=0.25,
                 focal_loss_gamma=2.0,
                 smooth_l1_beta=0.0,
                 box_reg_loss_type="smooth_l1",
                 test_score_thresh=0.05,
                 test_topk_candidates=1000,
                 test_nms_thresh=0.5,
                 max_detections_per_image=100,
                 pixel_mean,
                 pixel_std,
                 vis_period=0,
                 input_format="BGR"):
        
        """NOTE: this interface is experimental.

        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            head (nn.Module): a module that predicts logits and regression deltas
                for each level from a list of per-level features
            head_in_features (Tuple[str]): Names of the input feature maps to be used in head
            anchor_generator (nn.Module): a module that creates anchors from a
                list of features. Usually an instance of :class:`AnchorGenerator`
            box2box_transform (Box2BoxTransform): defines the transform from anchors boxes to
                instance boxes
            anchor_matcher (Matcher): label the anchors by matching them with ground truth.
            num_classes (int): number of classes. Used to label background proposals.

            # Loss parameters:
            focal_loss_alpha (float): focal_loss_alpha
            focal_loss_gamma (float): focal_loss_gamma
            smooth_l1_beta (float): smooth_l1_beta
            box_reg_loss_type (str): Options are "smooth_l1", "giou", "diou", "ciou"

            # Inference parameters:
            test_score_thresh (float): Inference cls score threshold, only anchors with
                score > INFERENCE_TH are considered for inference (to improve speed)
            test_topk_candidates (int): Select topk candidates before NMS
            test_nms_thresh (float): Overlap threshold used for non-maximum suppression
                (suppress boxes with IoU >= this threshold)
            max_detections_per_image (int):
                Maximum number of detections to return per image during inference
                (100 is based on the limit established for the COCO dataset).

            pixel_mean, pixel_std: see :class:`DenseDetector`."""
        
        super().__init__(backbone, head, head_in_features, pixel_mean=pixel_mean, pixel_std=pixel_std)
        self.num_classes = num_classes

        # Anchors
        self.anchor_generator = anchor_generator
        self.box2box_transform = box2box_transform
        self.anchor_matcher = anchor_matcher

        # Loss parameters:
        self.focal_loss_alpha = focal_loss_alpha
        self.focal_loss_gamma = focal_loss_gamma
        self.smooth_l1_beta = smooth_l1_beta
        self.box_reg_loss_type = box_reg_loss_type
        
        # Inference parameters:
        self.test_score_thresh = test_score_thresh
        self.test_topk_candidates = test_topk_candidates
        self.test_nms_thresh = test_nms_thresh
        self.max_detections_per_image = max_detections_per_image
        
        # Vis parameters
        self.vis_period = vis_period
        self.input_format = input_format

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        backbone_shape = backbone.output_shape()
        feature_shapes = [backbone_shape[f] for f in cfg.MODEL.RETINANET.IN_FEATURES]
        head = RetinaNetHead(cfg, feature_shapes)
        anchor_generator = build_anchor_generator(cfg, feature_shapes)
        return {"backbone": backbone,
                "head": head,
                "anchor_generator": anchor_generator,
                "box2box_transform": Box2BoxTransform(weights=cfg.MODEL.RETINANET.BBOX_REG_WEIGHTS),
                "anchor_matcher": Matcher(cfg.MODEL.RETINANET.IOU_THRESHOLDS,
                                          cfg.MODEL.RETINANET.IOU_LABELS,
                                          allow_low_quality_matches=True,),
                "pixel_mean": cfg.MODEL.PIXEL_MEAN,
                "pixel_std": cfg.MODEL.PIXEL_STD,
                "num_classes": cfg.MODEL.RETINANET.NUM_CLASSES,
                "head_in_features": cfg.MODEL.RETINANET.IN_FEATURES,
                
                # Loss parameters:
                "focal_loss_alpha": cfg.MODEL.RETINANET.FOCAL_LOSS_ALPHA,
                "focal_loss_gamma": cfg.MODEL.RETINANET.FOCAL_LOSS_GAMMA,
                "smooth_l1_beta": cfg.MODEL.RETINANET.SMOOTH_L1_LOSS_BETA,
                "box_reg_loss_type": cfg.MODEL.RETINANET.BBOX_REG_LOSS_TYPE,
                
                # Inference parameters:
                "test_score_thresh": cfg.MODEL.RETINANET.SCORE_THRESH_TEST,
                "test_topk_candidates": cfg.MODEL.RETINANET.TOPK_CANDIDATES_TEST,
                "test_nms_thresh": cfg.MODEL.RETINANET.NMS_THRESH_TEST,
                "max_detections_per_image": cfg.TEST.DETECTIONS_PER_IMAGE,
                
                # Vis parameters
                "vis_period": cfg.VIS_PERIOD,
                "input_format": cfg.INPUT.FORMAT,}

    def forward_training(self, images, features, predictions, gt_instances):
        # Transpose the Hi*Wi*A dimension to the middle:
        # self._transpose_dense_predictions located at dense_detector.py
        pred_logits, pred_anchor_deltas = self._transpose_dense_predictions(predictions, [self.num_classes, 4])
        anchors = self.anchor_generator(features)
        gt_labels, gt_boxes, index_batch, size_batch = self.label_anchors(anchors, gt_instances)
        # gt_labels & gt_boxes are list, len of list is bs, shape is super large N [N] & [N, 4]
        
        return self.losses(anchors, pred_logits, gt_labels, pred_anchor_deltas, gt_boxes, index_batch, size_batch)

    def losses(self, anchors, pred_logits, gt_labels, pred_anchor_deltas, gt_boxes, index_batch, size_batch):
        """Args:
            anchors (list[Boxes]): a list of #feature level Boxes
            gt_labels, gt_boxes: see output of :meth:`RetinaNet.label_anchors`.
                Their shapes are (N, R) and (N, R, 4), respectively, where R is
                the total number of anchors across levels, i.e. sum(Hi x Wi x Ai)
            pred_logits, pred_anchor_deltas: both are list[Tensor]. Each element in the
                list corresponds to one level and has shape (N, Hi * Wi * Ai, K or 4).
                Where K is the number of classes used in `pred_logits`.

        Returns:
            dict[str, Tensor]:
                mapping from a named loss to a scalar tensor storing the loss.
                Used during training only. The dict keys are: "loss_cls" and "loss_box_reg"
        """
        
        num_images = len(gt_labels)
        gt_labels = torch.stack(gt_labels) # [bs, large N]

        # for i in range(len(pred_anchor_deltas)): 
        #     print(f"pred_anchor_deltas: {pred_anchor_deltas[i].shape}")
        #     pred_anchor_deltas[i] = pred_anchor_deltas[i].detach().cpu()
        # pred_anchor_deltas is list, have 5 elements, shape like [bs, N, 4]

        # print(f"{cat(pred_anchor_deltas, dim=1).shape}")
        # cat(pred_anchor_deltas, dim=1).shape=[bs, 33804, 4]
        
        # print("gt boxes: ", len(gt_boxes), gt_boxes[0].shape, gt_boxes[1].shape)
        # bs, gt_boxes[i].shape=[33804, 4]
        # gt_boxes is a list, including R number of tensor, which is xyxy

        valid_mask, opsite_valid = gt_labels >= 0, gt_labels < 0
        pos_mask = (gt_labels >= 0) & (gt_labels != self.num_classes)
        num_pos_anchors = pos_mask.sum().item() # a single number: 2 × gt_labels.shape[1]

        # pos_mask.shape = [2, 103635]
        # valid_mask.shape = [2, 103635]
        
        get_event_storage().put_scalar("num_pos_anchors", num_pos_anchors / num_images)
        normalizer = self._ema_update("loss_normalizer", max(num_pos_anchors, 1), 100)

        # ---------------------------------
        
        # regression loss
        loss_box_reg, ratio_scale, scale_index, scale_areas = _dense_box_regression_loss(anchors, self.box2box_transform, pred_anchor_deltas, gt_boxes, pos_mask, self.box_reg_loss_type, self.smooth_l1_beta)

        # ---------------------------------

        # 设置控制项, scb_loss需要依赖初始损失函数
        # 并需要将reduction设为"none"
        scale_adaptive, scale_balance = False, False
        cross_entropy = True
        if scale_balance: reduction = "none"
        else: reduction = "sum"
        
        # classification loss
        gt_labels_target = F.one_hot(gt_labels[valid_mask].to(torch.int64), num_classes=self.num_classes + 1)[:, :-1]
        
        # print(f"gt_labels_target: {gt_labels.shape}, {gt_labels[valid_mask].shape}")
        # gt_labels[valid_mask]这个操作会将所有层按顺序横拼在一起,形成单个向量, 过滤索引中为False的值
        # gt_labels[valid_mask].shape=[bs*largeN], gt_labels_target.shape=[bs*largeN, num_classes]
        # noticed: bs*gt_labels.shape[1] != gt_labels[valid_mask].shape[0], valid_mask filter out some index

        pred_logits = cat(pred_logits, dim=1) # pred_logits是list, 其中五个值, 分别为[bs, 77760, 80], [bs, 19440, 80], [bs, 4860, 80], [bs, 1260, 80], [bs, 315, 80]

        # 边距函数不能放到valid_mask操作后因为连续边距必须使用scale_areas
        # pred_logits = adding_margin_discrete(pred_logits, gt_labels, pos_mask, scale_index[0], init_margin=0.80)
        # pred_logits = adding_margin_discrete(pred_logits, gt_labels, pos_mask, scale_index[1], init_margin=0.40)
        # pred_logits = adding_margin_discrete(pred_logits, gt_labels, pos_mask, scale_index[2], init_margin=0.20)
        # pred_logits = adding_margin_continue(pred_logits, gt_labels, pos_mask, scale_areas)

        if scale_adaptive or scale_balance:
            # 由于opsite_valid是一个与gt_labels同样shape的布尔张量, 使用其抽取gt_labels
            # 后得到一个shape=[N, 2]的张量, 其中每行第1个值为在gt_labels中的行号, 第2个
            # 值为在gt_labels中对应行的列号, 因此必须通过额外操作将其重新组合成原形态列表
            # 最后得到的列表每个元素即每张图片对应的网格点中需要去除的位置
            opsite_valid, opsite_list = opsite_valid.nonzero(as_tuple=False), []
            for i in range(gt_labels.shape[0]):
                indices = torch.where(opsite_valid[:, 0] == i)[0]
                opsite_list.append(opsite_valid[indices][:, 1].cpu().tolist())

            # 构造接下来要给index_batch每个元素加上的值, 由于valid_mask操作将四行张量全部
            # 横着叠在一起, 因此必须给每张图片的目标索引加上前张图片的有效网格点数量, 因此
            # 新列表中第个元素为0, 其余元素为adding_counts同位置前元素之和(不包含当前元素)
            adding_counts = torch.sum(valid_mask, dim=1).tolist()
            def prefix_sums(old_list):
                new_list = [0]
                current_sum = 0
                for num in old_list:
                    current_sum += num
                    new_list.append(current_sum)
                    if len(new_list) == len(old_list):
                        break
                return new_list
            adding_counts = prefix_sums(adding_counts)

            # index_batch是list, 每个元素都是一张图片包含的目标和能与目标匹配的Pred框在网格点的位置索引
            # 由于后续存在pred_logits[valid_mask]这种操作会将[bs, largeN]张量按行序叠起来, 因此必须将
            # index_batch中的索引加上len(gt_labels.shape[1]), 并且需要去除opsite_valid中包含的索引
            for i in range(len(index_batch)): # 每个批次有多张图片
                for j in range(len(index_batch[i])): # 每张图片有多个目标
                    for k in range(len(index_batch[i][j])): # 每个目标有多个Pred框
                        pred_index, counter = index_batch[i][j][k], 0
                        for outer_index in opsite_list[i]: # 为每个Pred对比一遍去除索引
                            if outer_index < pred_index:
                                counter += 1
                        index_batch[i][j][k] = pred_index - counter
                        index_batch[i][j][k] = index_batch[i][j][k] + adding_counts[i]

        # 将index_batch中的所有index按照尺度分成3个列表
        if scale_adaptive:
            small_index, midle_index, large_index = [], [], []
            for i in range(len(size_batch)):
                for j in range(len(size_batch[i])):
                    if size_batch[i][j] == 'small':
                        small_index.extend(index_batch[i][j])
                    if size_batch[i][j] == 'midle':
                        midle_index.extend(index_batch[i][j])
                    if size_batch[i][j] == 'large':
                        large_index.extend(index_batch[i][j])
            scale_index = small_index, midle_index, large_index

        # pred_logits.shape = [bs, 103635, 80]
        # pred_logits[pos_mask].shape = [174, 80]

        # 此处pred_logits按行序横拼, 降低维度
        pred_logits = pred_logits[valid_mask]

        # valid_mask.shape = [2, 103635]
        # pred_logits.shape = [206958, 80]
        # gt_labels.shape = [bs, large N]

        if scale_adaptive:
            # scale adaptive focal loss | self.focal_loss_alpha = 0.25, self.focal_loss_gamma = 2
            loss_cls = scalewise_sigmoid_focal_loss(pred_logits, gt_labels_target.to(pred_logits[0].dtype), scale_index, self.focal_loss_alpha, self.focal_loss_gamma, "sum")
        elif cross_entropy:
            loss_cls = F.binary_cross_entropy_with_logits(pred_logits, gt_labels_target.to(pred_logits[0].dtype), reduction=reduction)
        else:
            # origin sigmoid focal loss | self.focal_loss_alpha = 0.25, self.focal_loss_gamma = 2
            loss_cls = sigmoid_focal_loss_jit(pred_logits, gt_labels_target.to(pred_logits[0].dtype), alpha=self.focal_loss_alpha, gamma=self.focal_loss_gamma, reduction=reduction)

        # scale balance loss | scb_weight = 3 * (math.log(Mi)+1) / Mi, inwhich β=3, α=1
        # 需要让focal loss输入"none", 再通过索引获取同一个目标的多个Pred框, 求和后才能乘以权重
        # 因此前面必须获得所有Pred框对应的目标和索引, 并且索引要随着张量形态变化而变化, 而且判断
        # 目标尺寸也应该用gt_boxes进行, 而不是使用张量搜索的方式, 最后需要将loss加成单个值返回
        if scale_balance:
            loss_cls = scb_weighting(loss_cls, index_batch)

        # ---------------------------------

        return {"loss_cls": loss_cls / normalizer, "loss_box_reg": loss_box_reg / normalizer, "ratio_scale":ratio_scale}

    @torch.no_grad()
    def label_anchors(self, anchors, gt_instances):
        """Args:
            anchors (list[Boxes]): A list of #feature level Boxes.
                The Boxes contains anchors of this image on the specific feature level.
            gt_instances (list[Instances]): a list of N `Instances`s. The i-th
                `Instances` contains the ground-truth per-instance annotations
                for the i-th input image.

        Returns:
            list[Tensor]: List of #img tensors. i-th element is a vector of labels whose length is
            the total number of anchors across all feature maps (sum(Hi * Wi * A)).
            Label values are in {-1, 0, ..., K}, with -1 means ignore, and K means background.

            list[Tensor]: i-th element is a Rx4 tensor, where R is the total number of anchors
            across feature maps. The values are the matched gt boxes for each anchor.
            Values are undefined for those anchors not labeled as foreground."""
        
        anchors = Boxes.cat(anchors) # Rx4

        gt_labels, matched_gt_boxes, index_batch, size_batch = [], [], [], []
        
        for gt_per_image in gt_instances: # type(gt_per_image.gt_boxes)=.detectron2.structures.boxes.Boxes

            # print(f"gt_per_image.gt_boxes: {gt_per_image.gt_boxes.tensor.shape}, {anchors.tensor.shape}")
            # gt_per_image.gt_boxes.tensor.shape=[obj_num, 4], anchors.tensor.shape=[33804, 4]

            match_quality_matrix = pairwise_iou(gt_per_image.gt_boxes, anchors)
            # print(f"match_quality_matrix.shape:{match_quality_matrix.shape}")
            # match_quality_matrix.shape=[obj_num, 33804] # 每个GT目标与每个网格点Pred目标的iou值

            matched_idxs, anchor_labels = self.anchor_matcher(match_quality_matrix)
            # print(f"matched_idxs.shape: {matched_idxs.shape}, {anchor_labels.shape}")
            # matched_idxs.shape=[33804], anchor_labels.shape=[33804]
            # matched_idxs是每个网格点对应的目标, anchor_labels是每个网格点Pred框与阈值比较而得的flag
            del match_quality_matrix

            # matched_idxs通过matcher.py中的torch.max而得, 由于从每个列中选最大iou因此只有行号
            # anchor_labels元素为1即此处有目标, 因此用其来保留存在目标的网格点, 这些网格点
            # 在上一行被赋予了类别, 即matched_idxs这个张量, torch.unique返回独特值列表和对应数量
            # 通过复杂张量操作将每个目标在网格点中对应的Pred框的index抽取出来, 具体可以问GPT
            gt_cls, _ = torch.unique(matched_idxs[anchor_labels == 1], return_counts=True)
            gt_cls, anchor_copy = gt_cls.cpu().tolist(), anchor_labels.clone()
            anchor_copy[anchor_copy==-1], index_list, size_list = 0, [], []

            for line, size in zip(gt_cls, gt_per_image.gt_boxes.tensor):
                index = torch.nonzero((matched_idxs==line) & anchor_copy, as_tuple=False).squeeze().tolist()
                if isinstance(index, int): index = [index]
                index_list.append(index)

                area = (size[2]-size[1]) * (size[3]-size[0])
                if area <= 32**2: size_list.append("small")
                if 32**2 < area <= 96**2: size_list.append("midle")
                if 96**2 < area: size_list.append("large")
            
            # for cls, indices in cls_dict.items(): print(f"cls_dict: {cls}, {len(indices)}")

            if len(gt_per_image) > 0: # execute this part by testing
                matched_gt_boxes_i = gt_per_image.gt_boxes.tensor[matched_idxs]
                # print(f"matched_gt_boxes_i.shape: {matched_gt_boxes_i.shape}")
                # matched_gt_boxes_i.shape=[33804, 4]
                
                # this is the line make gt_labels become ridiculous long
                # gt_classes is tensor, shape=[N], N is objects number
                gt_labels_i = gt_per_image.gt_classes[matched_idxs]
                
                gt_labels_i[anchor_labels == 0] = self.num_classes # Anchors with label 0 are treated as background.
                gt_labels_i[anchor_labels == -1] = -1 # Anchors with label -1 are ignored.
            else:
                matched_gt_boxes_i = torch.zeros_like(anchors.tensor)
                gt_labels_i = torch.zeros_like(matched_idxs) + self.num_classes
            
            gt_labels.append(gt_labels_i)
            matched_gt_boxes.append(matched_gt_boxes_i)
            index_batch.append(index_list)
            size_batch.append(size_list)

        return gt_labels, matched_gt_boxes, index_batch, size_batch

    def forward_inference(self, images: ImageList, features: List[Tensor], predictions: List[List[Tensor]]):
        pred_logits, pred_anchor_deltas = self._transpose_dense_predictions(predictions, [self.num_classes, 4])
        anchors = self.anchor_generator(features)

        results: List[Instances] = []
        for img_idx, image_size in enumerate(images.image_sizes):
            scores_per_image = [x[img_idx].sigmoid_() for x in pred_logits]
            deltas_per_image = [x[img_idx] for x in pred_anchor_deltas]
            results_per_image = self.inference_single_image(anchors, scores_per_image, deltas_per_image, image_size)
            results.append(results_per_image)
        return results

    def inference_single_image(self, anchors: List[Boxes], box_cls: List[Tensor], box_delta: List[Tensor], image_size: Tuple[int, int]):
        """
        Single-image inference. Return bounding-box detection results by thresholding
        on scores and applying non-maximum suppression (NMS).

        Arguments:
            anchors (list[Boxes]): list of #feature levels. Each entry contains
                a Boxes object, which contains all the anchors in that feature level.
            box_cls (list[Tensor]): list of #feature levels. Each entry contains
                tensor of size (H x W x A, K)
            box_delta (list[Tensor]): Same shape as 'box_cls' except that K becomes 4.
            image_size (tuple(H, W)): a tuple of the image height and width.

        Returns:
            Same as `inference`, but for only one image.
        """
        pred = self._decode_multi_level_predictions(anchors, box_cls, box_delta, self.test_score_thresh, self.test_topk_candidates, image_size)
        keep = batched_nms(pred.pred_boxes.tensor, pred.scores, pred.pred_classes, self.test_nms_thresh) # per-class NMS
            
        return pred[keep[: self.max_detections_per_image]]


class RetinaNetHead(nn.Module):
    """The head used in RetinaNet for object classification and box regression.
    It has two subnets for the two tasks, with a common structure but separate parameters."""
    
    @configurable
    def __init__(self, *, input_shape: List[ShapeSpec], num_classes, num_anchors, conv_dims: List[int], norm="", prior_prob=0.01):
        
        """NOTE: this interface is experimental.

        Args:
            input_shape (List[ShapeSpec]): input shape
            num_classes (int): number of classes. Used to label background proposals.
            num_anchors (int): number of generated anchors
            conv_dims (List[int]): dimensions for each convolution layer
            norm (str or callable):
                Normalization for conv layers except for the two output layers.
                See :func:`detectron2.layers.get_norm` for supported types.
            prior_prob (float): Prior weight for computing bias"""
        
        super().__init__()

        self._num_features = len(input_shape)
        if norm == "BN" or norm == "SyncBN":
            logger.info(f"Using domain-specific {norm} in RetinaNetHead with len={self._num_features}.")
            bn_class = nn.BatchNorm2d if norm == "BN" else nn.SyncBatchNorm

            def norm(c):
                return CycleBatchNormList(length=self._num_features, bn_class=bn_class, num_features=c)

        else:
            norm_name = str(type(get_norm(norm, 32)))
            if "BN" in norm_name:
                logger.warning(f"Shared BatchNorm (type={norm_name}) may not work well in RetinaNetHead.")

        cls_subnet = []
        bbox_subnet = []
        for in_channels, out_channels in zip([input_shape[0].channels] + list(conv_dims), conv_dims):
            cls_subnet.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1))
            if norm:
                cls_subnet.append(get_norm(norm, out_channels))
            cls_subnet.append(nn.ReLU())
            bbox_subnet.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1))
            if norm:
                bbox_subnet.append(get_norm(norm, out_channels))
            bbox_subnet.append(nn.ReLU())

        self.cls_subnet = nn.Sequential(*cls_subnet)
        self.bbox_subnet = nn.Sequential(*bbox_subnet)
        self.cls_score = nn.Conv2d(conv_dims[-1], num_anchors * num_classes, kernel_size=3, stride=1, padding=1)
        self.bbox_pred = nn.Conv2d(conv_dims[-1], num_anchors * 4, kernel_size=3, stride=1, padding=1)

        # Initialization
        for modules in [self.cls_subnet, self.bbox_subnet, self.cls_score, self.bbox_pred]:
            for layer in modules.modules():
                if isinstance(layer, nn.Conv2d):
                    torch.nn.init.normal_(layer.weight, mean=0, std=0.01)
                    torch.nn.init.constant_(layer.bias, 0)

        # Use prior in model initialization to improve stability
        bias_value = -(math.log((1 - prior_prob) / prior_prob))
        torch.nn.init.constant_(self.cls_score.bias, bias_value)

    @classmethod
    def from_config(cls, cfg, input_shape: List[ShapeSpec]):
        num_anchors = build_anchor_generator(cfg, input_shape).num_cell_anchors
        assert (len(set(num_anchors)) == 1), "Using different number of anchors between levels is not currently supported!"
        num_anchors = num_anchors[0]

        return {"input_shape": input_shape,
                "num_classes": cfg.MODEL.RETINANET.NUM_CLASSES,
                "conv_dims": [input_shape[0].channels] * cfg.MODEL.RETINANET.NUM_CONVS,
                "prior_prob": cfg.MODEL.RETINANET.PRIOR_PROB,
                "norm": cfg.MODEL.RETINANET.NORM,
                "num_anchors": num_anchors,}

    def forward(self, features: List[Tensor]):
        """Arguments:
            features (list[Tensor]): FPN feature map tensors in high to low resolution.
                Each tensor in the list correspond to different feature levels.

        Returns:
            logits (list[Tensor]): #lvl tensors, each has shape (N, AxK, Hi, Wi).
                The tensor predicts the classification probability
                at each spatial position for each of the A anchors and K object
                classes.
            bbox_reg (list[Tensor]): #lvl tensors, each has shape (N, Ax4, Hi, Wi).
                The tensor predicts 4-vector (dx,dy,dw,dh) box
                regression values for every anchor. These values are the
                relative offset between the anchor and the ground truth box."""
        
        assert len(features) == self._num_features
        logits = []
        bbox_reg = []
        for feature in features:
            logits.append(self.cls_score(self.cls_subnet(feature)))
            bbox_reg.append(self.bbox_pred(self.bbox_subnet(feature)))
        return logits, bbox_reg


# --------------------------------------------------------------------------------------

def adding_margin_discrete(logits, labels, fg_mask, index, init_margin=0):
    if len(index) != 0:
        # extract a new logits tensor
        logits_ = logits[fg_mask]

        # convert bool tensor to list
        fg_mask_ = fg_mask.nonzero().squeeze().tolist()
        if isinstance(fg_mask_, int): fg_mask_ = [fg_mask_]

        # modify the new tensor
        for i in index:
            logits_[i][labels[fg_mask][i]] -= init_margin

        # assign new tensor back into old tensor
        counter = 0
        for fg_index in fg_mask_:
            img_index, pred_index = fg_index
            logits[img_index][pred_index] = logits_[counter]
            counter += 1

    return logits

def adding_margin_continue(logits, labels, fg_mask, area):
    # y = e^(coei*x) inwhich coei is negative
    coei = -0.4621 # 0.4621≈In(0.25)/3

    # extract a new logits tensor
    logits_ = logits[fg_mask]

    # convert bool tensor to list
    fg_mask_ = fg_mask.nonzero().squeeze().tolist()
    if isinstance(fg_mask_, int): fg_mask_ = [fg_mask_]

    # modify the new tensor
    for i in range(logits_.shape[0]):
        # construct margin with hyper-params
        margin = math.exp(coei * math.sqrt(area[i] / 32**2))
        logits_[i][labels[fg_mask][i]] -= margin
    
    # assign new tensor back into old tensor
    counter = 0
    for fg_index in fg_mask_:
        img_index, pred_index = fg_index
        logits[img_index][pred_index] = logits_[counter]
        counter += 1

    return logits

# --------------------------------------------------------------------------------------

def scalewise_sigmoid_focal_loss(inputs: torch.Tensor,
                                 targets: torch.Tensor,

                                 scale_index: list,

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

    small_index, midle_index, large_index = scale_index
    num_s, num_l = len(small_index), len(large_index)
    midle_index = small_index.copy()
    midle_index.extend(large_index)
    midle_index = [i for i in range(len(ce_loss)) if i not in midle_index]

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
    loss_midle = loss_midle * ((1 - p_t_m) ** gamma)

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
        # loss = loss.sum()
        loss = loss_small.sum() + \
               loss_midle.sum() + \
               loss_large.sum()

    return loss

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

# --------------------------------------------------------------------------------------

def scb_weighting(loss_cls, index_batch):
    """
    Scale-Balanced Loss for Object Detection Pattern Recognition 2021 BUPT
    - scb_weight = 3 * (math.log(Mi) + 1) / Mi # Mi is pred num an obj can match
    - this function must placed after main loss function since it is an assistent
    """
    loss_cls = loss_cls.sum(dim=1)
    for i in range(len(index_batch)):
        for j in range(len(index_batch[i])):
            scb_weight = 3 * (math.log(len(index_batch[i][j]))+1) / len(index_batch[i][j])
            for k in range(len(index_batch[i][j])):
                loss_cls[index_batch[i][j][k]] = loss_cls[index_batch[i][j][k]] * scb_weight
    loss_cls = loss_cls.sum()
    
    return loss_cls