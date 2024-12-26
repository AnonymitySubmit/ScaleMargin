# Copyright (c) Facebook, Inc. and its affiliates.
import statistics
import numpy as np
import torch, math, copy
import torch.nn.functional as F

from torch import nn
from fvcore.nn import sigmoid_focal_loss_jit
from typing import Dict, List, Optional, Tuple, Union

from detectron2.config import configurable
from detectron2.layers import Conv2d, ShapeSpec, cat
from detectron2.structures import Boxes, ImageList, Instances, pairwise_iou
from detectron2.utils.events import get_event_storage
from detectron2.utils.memory import retry_if_cuda_oom
from detectron2.utils.registry import Registry

from ..anchor_generator import build_anchor_generator
from ..box_regression import Box2BoxTransform, _dense_box_regression_loss
from ..matcher import Matcher
from ..sampling import subsample_labels
from .build import PROPOSAL_GENERATOR_REGISTRY
from .proposal_utils import find_top_rpn_proposals

RPN_HEAD_REGISTRY = Registry("RPN_HEAD")
RPN_HEAD_REGISTRY.__doc__ = """
Registry for RPN heads, which take feature maps and perform
objectness classification and bounding box regression for anchors.

The registered object will be called with `obj(cfg, input_shape)`.
The call should return a `nn.Module` object.
"""


"""
Shape shorthand in this module:

    N: number of images in the minibatch
    L: number of feature maps per image on which RPN is run
    A: number of cell anchors (must be the same for all feature maps)
    Hi, Wi: height and width of the i-th feature map
    B: size of the box parameterization

Naming convention:

    objectness: refers to the binary classification of an anchor as object vs. not object.

    deltas: refers to the 4-d (dx, dy, dw, dh) deltas that parameterize the box2box
    transform (see :class:`box_regression.Box2BoxTransform`), or 5d for rotated boxes.

    pred_objectness_logits: predicted objectness scores in [-inf, +inf]; use
        sigmoid(pred_objectness_logits) to estimate P(object).

    gt_labels: ground-truth binary classification labels for objectness

    pred_anchor_deltas: predicted box2box transform deltas

    gt_anchor_deltas: ground-truth box2box transform deltas
"""

# voc self collect
# gt_cls_list = [3148, 2836, 4503, 3277, 4553, 2154, 8948, 4166, 9569, 2464, 
#                2400, 5231, 2795, 2784, 35451, 3731, 3056, 3034, 2576, 2803]

# coco self collect
# gt_cls_list = [257252, 7056,  43533, 8654,  5129,  6061,  4570,  9970, 10576, 12842, 
#                1865,   1983,  1283,  9820,  10542, 4766,  5500,  6565, 9223,  8014, 
#                5484,   1294,  5269,  5128,  8714,  11265, 12342, 6448, 6112,  2681, 
#                6623,   2681,  6299,  8802,  3273,  3747,  5536,  6095, 4807,  24070, 
#                7839,   20574, 5474,  7760,  6159,  14326, 9195,  5776, 4356,  6306, 
#                7262,   7758,  2883,  5807,  7005,  6296,  38073, 5779, 8631,  4192, 
#                15695,  4149,  5803,  4960,  2261,  5700,  2854,  6422, 1672,  3334, 
#                225,    5609,  2634,  24077, 6320,  6577,  1464,  4729, 198,   1945]

# coco from internet
gt_cls_list = [262465, 7113,  43867, 8725,  5135,  6069,  4571,  9973, 10759, 12884, 
                1865,   1983,  1285,  9838,  10806, 4768,  5508,  6587, 9509,  8147, 
                5513,   1294,  5303,  5131,  8720,  11431, 12354, 6496, 6192,  2682, 
                6646,   2685,  6347,  9076,  3276,  3747,  5543,  6126, 4812,  24342, 
                7913,   20650, 5479,  7770,  6165,  14358, 9458,  5851, 4373,  6399, 
                7308,   51719, 8426,  5821,  7179,  6353,  38491, 5779, 8652,  4192, 
                15714,  4157,  5805,  4970,  2262,  5703,  2855,  6434, 1673,  3334, 
                225,    5610,  2637,  24715, 6334,  6613,  1481,  6087, 198,   1954]


def build_rpn_head(cfg, input_shape):
    """
    Build an RPN head defined by `cfg.MODEL.RPN.HEAD_NAME`.
    """
    name = cfg.MODEL.RPN.HEAD_NAME
    return RPN_HEAD_REGISTRY.get(name)(cfg, input_shape)


@RPN_HEAD_REGISTRY.register()
class StandardRPNHead(nn.Module):
    """
    Standard RPN classification and regression heads described in :paper:`Faster R-CNN`.
    Uses a 3x3 conv to produce a shared hidden state from which one 1x1 conv predicts
    objectness logits for each anchor and a second 1x1 conv predicts bounding-box deltas
    specifying how to deform each anchor into an object proposal.
    """

    @configurable
    def __init__(
        self, *, in_channels: int, num_anchors: int, box_dim: int = 4, conv_dims: List[int] = (-1,)
    ):
        """
        NOTE: this interface is experimental.

        Args:
            in_channels (int): number of input feature channels. When using multiple
                input features, they must have the same number of channels.
            num_anchors (int): number of anchors to predict for *each spatial position*
                on the feature map. The total number of anchors for each
                feature map will be `num_anchors * H * W`.
            box_dim (int): dimension of a box, which is also the number of box regression
                predictions to make for each anchor. An axis aligned box has
                box_dim=4, while a rotated box has box_dim=5.
            conv_dims (list[int]): a list of integers representing the output channels
                of N conv layers. Set it to -1 to use the same number of output channels
                as input channels.
        """
        super().__init__()
        cur_channels = in_channels
        # Keeping the old variable names and structure for backwards compatiblity.
        # Otherwise the old checkpoints will fail to load.
        if len(conv_dims) == 1:
            out_channels = cur_channels if conv_dims[0] == -1 else conv_dims[0]
            # 3x3 conv for the hidden representation
            self.conv = self._get_rpn_conv(cur_channels, out_channels)
            cur_channels = out_channels
        else:
            self.conv = nn.Sequential()
            for k, conv_dim in enumerate(conv_dims):
                out_channels = cur_channels if conv_dim == -1 else conv_dim
                if out_channels <= 0:
                    raise ValueError(
                        f"Conv output channels should be greater than 0. Got {out_channels}"
                    )
                conv = self._get_rpn_conv(cur_channels, out_channels)
                self.conv.add_module(f"conv{k}", conv)
                cur_channels = out_channels
        # 1x1 conv for predicting objectness logits
        self.objectness_logits = nn.Conv2d(cur_channels, num_anchors, kernel_size=1, stride=1)
        # 1x1 conv for predicting box2box transform deltas
        self.anchor_deltas = nn.Conv2d(cur_channels, num_anchors * box_dim, kernel_size=1, stride=1)

        # Keeping the order of weights initialization same for backwards compatiblility.
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.normal_(layer.weight, std=0.01)
                nn.init.constant_(layer.bias, 0)

    def _get_rpn_conv(self, in_channels, out_channels):
        return Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            activation=nn.ReLU(),
        )

    @classmethod
    def from_config(cls, cfg, input_shape):
        # Standard RPN is shared across levels:
        in_channels = [s.channels for s in input_shape]
        assert len(set(in_channels)) == 1, "Each level must have the same channel!"
        in_channels = in_channels[0]

        # RPNHead should take the same input as anchor generator
        # NOTE: it assumes that creating an anchor generator does not have unwanted side effect.
        anchor_generator = build_anchor_generator(cfg, input_shape)
        num_anchors = anchor_generator.num_anchors
        box_dim = anchor_generator.box_dim
        assert (
            len(set(num_anchors)) == 1
        ), "Each level must have the same number of anchors per spatial position"
        return {
            "in_channels": in_channels,
            "num_anchors": num_anchors[0],
            "box_dim": box_dim,
            "conv_dims": cfg.MODEL.RPN.CONV_DIMS,
        }

    def forward(self, features: List[torch.Tensor]):
        """
        Args:
            features (list[Tensor]): list of feature maps

        Returns:
            list[Tensor]: A list of L elements.
                Element i is a tensor of shape (N, A, Hi, Wi) representing
                the predicted objectness logits for all anchors. A is the number of cell anchors.
            list[Tensor]: A list of L elements. Element i is a tensor of shape
                (N, A*box_dim, Hi, Wi) representing the predicted "deltas" used to transform anchors
                to proposals.
        """
        pred_objectness_logits = []
        pred_anchor_deltas = []
        for x in features:
            t = self.conv(x)
            pred_objectness_logits.append(self.objectness_logits(t))
            pred_anchor_deltas.append(self.anchor_deltas(t))
        return pred_objectness_logits, pred_anchor_deltas


@PROPOSAL_GENERATOR_REGISTRY.register()
class RPN(nn.Module):
    """
    Region Proposal Network, introduced by :paper:`Faster R-CNN`.
    """

    @configurable
    def __init__(
        self,
        *,
        in_features: List[str],
        head: nn.Module,
        anchor_generator: nn.Module,
        anchor_matcher: Matcher,
        box2box_transform: Box2BoxTransform,
        batch_size_per_image: int,
        positive_fraction: float,
        pre_nms_topk: Tuple[float, float],
        post_nms_topk: Tuple[float, float],
        nms_thresh: float = 0.7,
        min_box_size: float = 0.0,
        anchor_boundary_thresh: float = -1.0,
        loss_weight: Union[float, Dict[str, float]] = 1.0,
        box_reg_loss_type: str = "smooth_l1",
        smooth_l1_beta: float = 0.0,
    ):
        """
        NOTE: this interface is experimental.

        Args:
            in_features (list[str]): list of names of input features to use
            head (nn.Module): a module that predicts logits and regression deltas
                for each level from a list of per-level features
            anchor_generator (nn.Module): a module that creates anchors from a
                list of features. Usually an instance of :class:`AnchorGenerator`
            anchor_matcher (Matcher): label the anchors by matching them with ground truth.
            box2box_transform (Box2BoxTransform): defines the transform from anchors boxes to
                instance boxes
            batch_size_per_image (int): number of anchors per image to sample for training
            positive_fraction (float): fraction of foreground anchors to sample for training
            pre_nms_topk (tuple[float]): (train, test) that represents the
                number of top k proposals to select before NMS, in
                training and testing.
            post_nms_topk (tuple[float]): (train, test) that represents the
                number of top k proposals to select after NMS, in
                training and testing.
            nms_thresh (float): NMS threshold used to de-duplicate the predicted proposals
            min_box_size (float): remove proposal boxes with any side smaller than this threshold,
                in the unit of input image pixels
            anchor_boundary_thresh (float): legacy option
            loss_weight (float|dict): weights to use for losses. Can be single float for weighting
                all rpn losses together, or a dict of individual weightings. Valid dict keys are:
                    "loss_rpn_cls" - applied to classification loss
                    "loss_rpn_loc" - applied to box regression loss
            box_reg_loss_type (str): Loss type to use. Supported losses: "smooth_l1", "giou".
            smooth_l1_beta (float): beta parameter for the smooth L1 regression loss. Default to
                use L1 loss. Only used when `box_reg_loss_type` is "smooth_l1"
        """
        super().__init__()
        self.in_features = in_features
        self.rpn_head = head
        self.anchor_generator = anchor_generator
        self.anchor_matcher = anchor_matcher
        self.box2box_transform = box2box_transform
        self.batch_size_per_image = batch_size_per_image
        self.positive_fraction = positive_fraction
        # Map from self.training state to train/test settings
        self.pre_nms_topk = {True: pre_nms_topk[0], False: pre_nms_topk[1]}
        self.post_nms_topk = {True: post_nms_topk[0], False: post_nms_topk[1]}
        self.nms_thresh = nms_thresh
        self.min_box_size = float(min_box_size)
        self.anchor_boundary_thresh = anchor_boundary_thresh
        if isinstance(loss_weight, float):
            loss_weight = {"loss_rpn_cls": loss_weight, "loss_rpn_loc": loss_weight}
        self.loss_weight = loss_weight
        self.box_reg_loss_type = box_reg_loss_type
        self.smooth_l1_beta = smooth_l1_beta

    @classmethod
    def from_config(cls, cfg, input_shape: Dict[str, ShapeSpec]):
        in_features = cfg.MODEL.RPN.IN_FEATURES
        ret = {
            "in_features": in_features,
            "min_box_size": cfg.MODEL.PROPOSAL_GENERATOR.MIN_SIZE,
            "nms_thresh": cfg.MODEL.RPN.NMS_THRESH,
            "batch_size_per_image": cfg.MODEL.RPN.BATCH_SIZE_PER_IMAGE,
            "positive_fraction": cfg.MODEL.RPN.POSITIVE_FRACTION,
            "loss_weight": {
                "loss_rpn_cls": cfg.MODEL.RPN.LOSS_WEIGHT,
                "loss_rpn_loc": cfg.MODEL.RPN.BBOX_REG_LOSS_WEIGHT * cfg.MODEL.RPN.LOSS_WEIGHT,
            },
            "anchor_boundary_thresh": cfg.MODEL.RPN.BOUNDARY_THRESH,
            "box2box_transform": Box2BoxTransform(weights=cfg.MODEL.RPN.BBOX_REG_WEIGHTS),
            "box_reg_loss_type": cfg.MODEL.RPN.BBOX_REG_LOSS_TYPE,
            "smooth_l1_beta": cfg.MODEL.RPN.SMOOTH_L1_BETA,
        }

        ret["pre_nms_topk"] = (cfg.MODEL.RPN.PRE_NMS_TOPK_TRAIN, cfg.MODEL.RPN.PRE_NMS_TOPK_TEST)
        ret["post_nms_topk"] = (cfg.MODEL.RPN.POST_NMS_TOPK_TRAIN, cfg.MODEL.RPN.POST_NMS_TOPK_TEST)

        ret["anchor_generator"] = build_anchor_generator(cfg, [input_shape[f] for f in in_features])
        ret["anchor_matcher"] = Matcher(
            cfg.MODEL.RPN.IOU_THRESHOLDS, cfg.MODEL.RPN.IOU_LABELS, allow_low_quality_matches=True
        )
        ret["head"] = build_rpn_head(cfg, [input_shape[f] for f in in_features])
        return ret

    def _subsample_labels(self, label):
        """
        Randomly sample a subset of positive and negative examples, and overwrite
        the label vector to the ignore value (-1) for all elements that are not
        included in the sample.

        Args:
            labels (Tensor): a vector of -1, 0, 1. Will be modified in-place and returned.
        """
        pos_idx, neg_idx = subsample_labels(
            label, self.batch_size_per_image, self.positive_fraction, 0
        )
        # Fill with the ignore label (-1), then set positive and negative labels
        label.fill_(-1)
        label.scatter_(0, pos_idx, 1)
        label.scatter_(0, neg_idx, 0)
        return label

    @torch.jit.unused
    @torch.no_grad()
    def label_and_sample_anchors(
        self, anchors: List[Boxes], gt_instances: List[Instances]
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Args:
            anchors (list[Boxes]): anchors for each feature map.
            gt_instances: the ground-truth instances for each image.

        Returns:
            list[Tensor]:
                List of #img tensors. i-th element is a vector of labels whose length is
                the total number of anchors across all feature maps R = sum(Hi * Wi * A).
                Label values are in {-1, 0, 1}, with meanings: -1 = ignore; 0 = negative
                class; 1 = positive class.
            list[Tensor]:
                i-th element is a Rx4 tensor. The values are the matched gt boxes for each
                anchor. Values are undefined for those anchors not labeled as 1.
        """
        anchors = Boxes.cat(anchors)

        # addition code getting batch gt labels
        gt_labes = [x.gt_classes.cpu().tolist() for x in gt_instances]

        gt_boxes = [x.gt_boxes for x in gt_instances]
        image_sizes = [x.image_size for x in gt_instances]
        del gt_instances

        matched_gt_boxes, index_batch = [], []
        gt_labels, size_batch = [], []

        for image_size_i, gt_boxes_i in zip(image_sizes, gt_boxes):
            """
            image_size_i: (h, w) for the i-th image
            gt_boxes_i: ground-truth boxes for i-th image
            """

            match_quality_matrix = retry_if_cuda_oom(pairwise_iou)(gt_boxes_i, anchors)
            matched_idxs, gt_labels_i = retry_if_cuda_oom(self.anchor_matcher)(match_quality_matrix)
            # Matching is memory-expensive and may result in CPU tensors. But the result is small
            gt_labels_i = gt_labels_i.to(device=gt_boxes_i.device)
            del match_quality_matrix

            if self.anchor_boundary_thresh >= 0:
                # Discard anchors that go out of the boundaries of the image
                # NOTE: This is legacy functionality that is turned off by default in Detectron2
                anchors_inside_image = anchors.inside_box(image_size_i, self.anchor_boundary_thresh)
                gt_labels_i[~anchors_inside_image] = -1

            # A vector of labels (-1, 0, 1) for each anchor
            gt_labels_i = self._subsample_labels(gt_labels_i)

            def extract_matched_pred_point(matched_idxs, gt_labels_i, gt_boxes_i):
                # matched_idxs通过matcher.py中的torch.max而得, 选每列最大iou得到行号
                # anchor_labels中元素为1即此处有目标, 因此用其来保留存在目标的网格点
                gt_cls = torch.unique(matched_idxs[gt_labels_i == 1])
                gt_cls, anchor_copy = gt_cls.cpu().tolist(), gt_labels_i.clone()
                anchor_copy[anchor_copy==-1], index_list, size_list = 0, [], []

                for i in gt_cls:
                    index = torch.nonzero((matched_idxs==i) & anchor_copy, as_tuple=False).squeeze().tolist()
                    index_list.append([index]) if isinstance(index, int) else index_list.append(index)

                    size = gt_boxes_i.tensor[i] # 测试发现i不连续
                    area = (size[2]-size[0]) * (size[3]-size[1])
                    if 0 < area <= 32**2: size_list.append("small")
                    elif 32**2 < area <= 96**2: size_list.append("midle")
                    elif 96**2 < area: size_list.append("large")
                
                return index_list, size_list

            index_list, size_list = extract_matched_pred_point(matched_idxs, gt_labels_i, gt_boxes_i)

            if len(gt_boxes_i) == 0:
                # These values won't be used anyway since the anchor is labeled as background
                matched_gt_boxes_i = torch.zeros_like(anchors.tensor)
            else:
                # TODO wasted indexing computation for ignored boxes
                matched_gt_boxes_i = gt_boxes_i[matched_idxs].tensor

            gt_labels.append(gt_labels_i)  # N,AHW
            matched_gt_boxes.append(matched_gt_boxes_i)
            index_batch.append(index_list)
            size_batch.append(size_list)
        
        return gt_labels, matched_gt_boxes, index_batch, size_batch, gt_labes

    @torch.jit.unused
    def losses(
        self,
        anchors: List[Boxes],
        pred_objectness_logits: List[torch.Tensor],
        gt_labels: List[torch.Tensor],
        pred_anchor_deltas: List[torch.Tensor],
        gt_boxes: List[torch.Tensor],
        index_batch, size_batch, gt_class,
        margin_list, margin_max, iter_num, max_iter
    ) -> Dict[str, torch.Tensor]:
        """
        Return the losses from a set of RPN predictions and their associated ground-truth.

        Args:
            anchors (list[Boxes or RotatedBoxes]): anchors for each feature map, each
                has shape (Hi*Wi*A, B), where B is box dimension (4 or 5).
            pred_objectness_logits (list[Tensor]): A list of L elements.
                Element i is a tensor of shape (N, Hi*Wi*A) representing
                the predicted objectness logits for all anchors.
            gt_labels (list[Tensor]): Output of :meth:`label_and_sample_anchors`.
            pred_anchor_deltas (list[Tensor]): A list of L elements. Element i is a tensor of shape
                (N, Hi*Wi*A, 4 or 5) representing the predicted "deltas" used to transform anchors
                to proposals.
            gt_boxes (list[Tensor]): Output of :meth:`label_and_sample_anchors`.

        Returns:
            dict[loss name -> loss value]: A dict mapping from loss name to loss value.
                Loss names are: `loss_rpn_cls` for objectness classification and
                `loss_rpn_loc` for proposal localization.
        """
        num_images = len(gt_labels)
        gt_labels = torch.stack(gt_labels)  # (N, sum(Hi*Wi*Ai))

        # Log the number of positive/negative anchors per-image that's used in training
        pos_mask = gt_labels == 1
        num_pos_anchors = pos_mask.sum().item()
        num_neg_anchors = (gt_labels == 0).sum().item()
        storage = get_event_storage()
        storage.put_scalar("rpn/num_pos_anchors", num_pos_anchors / num_images)
        storage.put_scalar("rpn/num_neg_anchors", num_neg_anchors / num_images)

        # ---------------------------------
        
        # regression loss
        localization_loss = _dense_box_regression_loss(anchors, self.box2box_transform, pred_anchor_deltas, gt_boxes, pos_mask, self.box_reg_loss_type, self.smooth_l1_beta)

        # ---------------------------------

        valid_mask = gt_labels >= 0

        scale_adaptv = False
        scale_balanc = True
        class_margin = False
        scale_margin = False
        magn_dynamic = False

        if scale_adaptv or scale_balanc or scale_margin or magn_dynamic or class_margin: 
            reduction = "none"
        else: 
            reduction = "sum"

        pred_objectness_logits = cat(pred_objectness_logits, dim=1)

        # 经过测试发现gt_boxes相当于给所有网格点都赋予了一个GT框, 没有数值全为0的空框, 全是真实GT的复制
        # 在bs=2的测试状态中gt_boxes经过pos_mask过滤后还剩几十上百个网格点, 而我把所有目标全部减去边距
        # 但实际上每张图片少则两三个目标, 最多也才十几个目标, 远低于经过pos_mask过滤的gt_boxes
        def gt_scale_obj_num(size_batch):
            s_counter, m_counter, l_counter = 0, 0, 0
            for i in range(len(size_batch)):
                for j in range(len(size_batch[i])):
                    if size_batch[i][j] == "small": s_counter += 1
                    if size_batch[i][j] == "midle": m_counter += 1
                    if size_batch[i][j] == "large": l_counter += 1
            return [s_counter, m_counter, l_counter]

        if scale_adaptv:
            gt_num = gt_scale_obj_num(size_batch)

        # 边距函数不能放到valid_mask操作后因为连续边距必须使用scale_areas
        # pred_objectness_logits = adding_margin_continue(pred_objectness_logits, pos_mask, scale_areas)

        # test code for print out single img's pred num
        # 打印每张图片拥有的Pred数量, 结合后续pred_obj_logit在用valid_mask
        # 索引完后每张图片只有256个logit值可知这256个logit值也有很多不是Pred
        # for i in range(len(index_batch)):
        #     counter = 0
        #     for j in range(len(index_batch[i])):
        #         counter += len(index_batch[i][j])
        #     print(f"single img pred num: {counter}")

        if scale_adaptv or scale_balanc or scale_margin or magn_dynamic or class_margin:

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

            # 由于index_batch中每张图片的目标包含的索引都是乱序
            # 因此首先以这些索引为键, 目标的序号为值构造出字典
            # 再为这个字典的键进行排序操作，这样就得到了目标在
            # 网格点上的顺序, 再得到这些目标自身组成列表的序号
            # 重新以目标为键, 序号为值组成字典, 这样就得到了每张
            # 图片去掉没有目标的网格点后剩余网格点的目标排序
            new_index_batch = []

            for i in range(len(index_batch)): # 每个批次有多张图片

                obj_dict, uniques, final, sorted_obj = {}, {}, [], []

                # 将每张图片的目标的索引全部以索引为键, 目标为
                # 值组成字典(索引不会重复), 方便后续对索引排序
                for j in range(len(index_batch[i])):
                    for pred_index in index_batch[i][j]:
                        obj_dict[pred_index] = j
                
                # 对字典的键进行排序得到目标在网格点上的顺序
                sorted_keys = sorted(list(obj_dict.keys()))
                for i in sorted_keys:
                    sorted_obj.append(obj_dict[i])
            
                # 将这个顺序中相同的目标的序号组成列表, 以目标
                # 为键, 列表为值组成字典, 方便后续按目标顺序将
                # 目标索引列表组成新列表, 这个新列表内每个目标
                # 包含的序号就是去除掉False点位后留下的网格点
                # 的新序号
                for index, value in enumerate(sorted_obj): 
                    if value in uniques:
                        uniques[value].append(index)
                    else:
                        uniques[value] = [index]
                
                # 最后对这个字典的键进行排序, 以键的大小顺序重
                # 新按顺序将值组成新列表
                for key in sorted(uniques.keys()): 
                    final.append(uniques[key])
                
                new_index_batch.append(final)
            
            index_batch = new_index_batch
        
            # 为每张图片的目标的索引加上行值, 模拟valid_mask将
            # pred_logits所有行叠起来的操作
            for i in range(len(index_batch)):
                for j in range(len(index_batch[i])):
                    for k in range(len(index_batch[i][j])):
                        index_batch[i][j][k] += adding_counts[i]

        # 将index_batch中的所有index按照尺度分成3个列表
        if scale_adaptv or magn_dynamic:
            small_index, midle_index, large_index = [], [], []
            small_count, midle_count, large_count = 0, 0, 0
            for i in range(len(size_batch)):
                for j in range(len(size_batch[i])):
                    if size_batch[i][j] == 'small':
                        small_index.extend(index_batch[i][j])
                        small_count += 1
                    if size_batch[i][j] == 'midle':
                        midle_index.extend(index_batch[i][j])
                        midle_count += 1
                    if size_batch[i][j] == 'large':
                        large_index.extend(index_batch[i][j])
                        large_count += 1
            # scale_index将不同尺度的GT对应的Pred分开储存, 方便
            # 对损失值张量进行抽样, 特别注意储存的是Pred的索引
            scale_index = [small_index, midle_index, large_index]

            # scale_counter储存的是不同尺度的GT的数量, 这个才是
            # 给更新margin_list使用的计数器
            scale_counter = [small_count, midle_count, large_count]
        
        # before indexing: pred_objectness_logits.shape == [bs, matched_idxs.shape]
        pred_objectness_logits = pred_objectness_logits[valid_mask]
        # after indexing: pred_objectness_logits.shape = [512]
        # which means every img have 256 length of data, total=256*bs

        # Label Distribution Aware Margin Loss
        if class_margin:
            pred_objectness_logits = adding_margin_classwise(pred_objectness_logits, index_batch, gt_class, gt_cls_list)

        if scale_margin:
            pred_objectness_logits = adding_margin_discrete(pred_objectness_logits, index_batch, size_batch, [1.00, 0.50, 0.25])
        
        if magn_dynamic:
            margin_list, margin_obj, margin_max = margin_dynamic_compute(pred_objectness_logits, index_batch, size_batch, scale_index, margin_list, margin_max)
            pred_objectness_logits = margin_dynamic_adding(pred_objectness_logits, index_batch, size_batch, margin_list, margin_obj, margin_max, [1, 0.5, 0.25])

        # classification loss
        objectness_loss = F.binary_cross_entropy_with_logits(pred_objectness_logits, gt_labels[valid_mask].to(torch.float32), reduction=reduction)

        if scale_margin or magn_dynamic or class_margin or scale_adaptv or scale_balanc:
            objectness_loss = margin_loss_normalize(objectness_loss, index_batch)

        if scale_margin or magn_dynamic or class_margin:
            objectness_loss = objectness_loss.sum()

        if scale_adaptv:
            objectness_loss = sca_weighting(objectness_loss, gt_num, small_index, midle_index, large_index, iter_num, max_iter)

        # scale balance loss | scb_weight = 3 * (math.log(Mi)+1) / Mi, inwhich β=3, α=1
        # 需要让focal loss输入"none", 再通过索引获取同一个目标的多个Pred框, 求和后才能乘以权重
        # 因此前面必须获得所有Pred框对应的目标和索引, 并且索引要随着张量形态变化而变化, 而且判断
        # 目标尺寸也应该用gt_boxes进行, 而不是使用张量搜索的方式, 最后需要将loss加成单个值返回
        if scale_balanc:
            objectness_loss = scb_weighting(objectness_loss, index_batch, size_batch, iter_num, max_iter)

        # ---------------------------------

        normalizer = self.batch_size_per_image * num_images

        # The original Faster R-CNN paper uses a slightly different 
        # normalizer for loc loss. But it doesn't matter in practice
        losses = {"loss_rpn_cls": objectness_loss / normalizer, "loss_rpn_loc": localization_loss / normalizer}
        losses = {k: v * self.loss_weight.get(k, 1.0) for k, v in losses.items()}
        
        # losses['ratio_scale'] = ratio_scale
        losses['margin_list'] = margin_list
        losses['margin_max'] = margin_max

        return losses

    def forward(
        self,
        images: ImageList,
        features: Dict[str, torch.Tensor],
        gt_instances: Optional[List[Instances]] = None,
        margin_list = None, margin_max = None, 
        iter_num = None, max_iter = None
    ):
        """
        Args:
            images (ImageList): input images of length `N`
            features (dict[str, Tensor]): input data as a mapping from feature
                map name to tensor. Axis 0 represents the number of images `N` in
                the input data; axes 1-3 are channels, height, and width, which may
                vary between feature maps (e.g., if a feature pyramid is used).
            gt_instances (list[Instances], optional): a length `N` list of `Instances`s.
                Each `Instances` stores ground-truth instances for the corresponding image.

        Returns:
            proposals: list[Instances]: contains fields "proposal_boxes", "objectness_logits"
            loss: dict[Tensor] or None
        """
        features = [features[f] for f in self.in_features]
        anchors = self.anchor_generator(features)

        pred_objectness_logits, pred_anchor_deltas = self.rpn_head(features)

        # Transpose the Hi*Wi*A dimension to the middle: (N, A, Hi, Wi) -> (N, Hi, Wi, A) -> (N, Hi*Wi*A)
        pred_objectness_logits = [score.permute(0, 2, 3, 1).flatten(1) for score in pred_objectness_logits]
        
        # (N, A*B, Hi, Wi) -> (N, A, B, Hi, Wi) -> (N, Hi, Wi, A, B) -> (N, Hi*Wi*A, B)
        pred_anchor_deltas = [x.view(x.shape[0], -1, self.anchor_generator.box_dim, x.shape[-2], x.shape[-1]).permute(0, 3, 4, 1, 2).flatten(1, -2) for x in pred_anchor_deltas]
        
        if self.training:
            assert gt_instances is not None, "RPN requires gt_instances in training!"
            gt_labels, gt_boxes, index_batch, size_batch, gt_class = self.label_and_sample_anchors(anchors, gt_instances)
            losses = self.losses(anchors, pred_objectness_logits, gt_labels, pred_anchor_deltas, gt_boxes, index_batch, size_batch, gt_class, margin_list, margin_max, iter_num, max_iter)
        else:
            losses = {}
        
        proposals = self.predict_proposals(anchors, pred_objectness_logits, pred_anchor_deltas, images.image_sizes)
        
        return proposals, losses

    def predict_proposals(
        self,
        anchors: List[Boxes],
        pred_objectness_logits: List[torch.Tensor],
        pred_anchor_deltas: List[torch.Tensor],
        image_sizes: List[Tuple[int, int]],
    ):
        """
        Decode all the predicted box regression deltas to proposals. Find the top proposals
        by applying NMS and removing boxes that are too small.

        Returns:
            proposals (list[Instances]): list of N Instances. The i-th Instances
                stores post_nms_topk object proposals for image i, sorted by their
                objectness score in descending order.
        """
        # The proposals are treated as fixed for joint training with roi heads.
        # This approach ignores the derivative w.r.t. the proposal boxes’ coordinates that
        # are also network responses.
        with torch.no_grad():
            pred_proposals = self._decode_proposals(anchors, pred_anchor_deltas)
            return find_top_rpn_proposals(
                pred_proposals,
                pred_objectness_logits,
                image_sizes,
                self.nms_thresh,
                self.pre_nms_topk[self.training],
                self.post_nms_topk[self.training],
                self.min_box_size,
                self.training)

    def _decode_proposals(self, anchors: List[Boxes], pred_anchor_deltas: List[torch.Tensor]):
        """
        Transform anchors into proposals by applying the predicted anchor deltas.

        Returns:
            proposals (list[Tensor]): A list of L tensors. Tensor i has shape
                (N, Hi*Wi*A, B)
        """
        N = pred_anchor_deltas[0].shape[0]
        proposals = []
        # For each feature map
        for anchors_i, pred_anchor_deltas_i in zip(anchors, pred_anchor_deltas):
            B = anchors_i.tensor.size(1)
            pred_anchor_deltas_i = pred_anchor_deltas_i.reshape(-1, B)
            # Expand anchors to shape (N*Hi*Wi*A, B)
            anchors_i = anchors_i.tensor.unsqueeze(0).expand(N, -1, -1).reshape(-1, B)
            proposals_i = self.box2box_transform.apply_deltas(pred_anchor_deltas_i, anchors_i)
            # Append feature map proposals with shape (N, Hi*Wi*A, B)
            proposals.append(proposals_i.view(N, -1, B))
        return proposals

# --------------------------------------------------------------------------------------

def adding_margin_discrete(logits, index_batch, size_batch, init_margin_list):
    # 每个批次有多张图片
    for i in range(len(index_batch)):
        # 每张图片有多个目标
        for j in range(len(index_batch[i])):
            # 每个目标都有多个Pred
            margin_list = []

            # 判断单个目标的尺度并赋予相应的初始值
            if size_batch[i][j] == 'small':
                init_margin = init_margin_list[0]
            if size_batch[i][j] == 'midle':
                init_margin = init_margin_list[1]
            if size_batch[i][j] == 'large': 
                init_margin = init_margin_list[2]

            # 将单个GT所有Pred的Logit值全部提取出来组成列表
            margin_list = [logits[index_batch[i][j][k]].item() for k in range(len(index_batch[i][j]))]

            # 筛选出其中最大的Logit值, 并且将其赋给其他Pred
            for k in range(len(index_batch[i][j])): logits[index_batch[i][j][k]] = max(margin_list)

            # 给所有Pred框的Logit减去init_margin
            for k in range(len(index_batch[i][j])): logits[index_batch[i][j][k]] -= init_margin

    return logits

def adding_margin_continue(logits, fg_mask, area):
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
        logits_[i] -= math.exp(coei * math.sqrt(area[i] / 32**2))
    
    # assign new tensor back into old tensor
    counter = 0
    for fg_index in fg_mask_:
        img_index, pred_index = fg_index
        logits[img_index][pred_index] = logits_[counter]
        counter += 1

    return logits

def adding_margin_classwise(logits, index_batch, gt_class, gt_cls_list, max_m=0.5):
    """
    Learning Imbalanced Datasets with Label-Distribution-Aware Margin Loss NIPS 2019
    """
    m_list = 1.0 / np.sqrt(np.sqrt(gt_cls_list))
    m_list = m_list * (max_m / np.max(m_list))

    # 每个批次有多张图片
    for i in range(len(index_batch)):
        # 每张图片有多个目标
        for j in range(len(index_batch[i])):
            # 每个目标都有多个Pred
            for k in range(len(index_batch[i][j])):
                logits[index_batch[i][j][k]] -= m_list[gt_class[i][j]] / len(index_batch[i][j])

    return logits

def margin_dynamic_compute(logits, index_batch, size_batch, obj_len, margin_list, margin_max):
    # initialize some parameters
    save_num, margin_obj = 100, [[], [], []]

    # 当本轮该尺度目标超过阈值时直接使用本轮目标
    if len(obj_len[0]) >= save_num: margin_list[0] = []
    if len(obj_len[1]) >= save_num: margin_list[1] = []
    if len(obj_len[2]) >= save_num: margin_list[2] = []

    # 每个批次有多张图片
    for i in range(len(index_batch)):
        # 每张图片有多个目标
        for j in range(len(index_batch[i])):

            # 初始化单个目标的Pred列表
            margin_pred = []

            # 选择当前的更新序号
            if size_batch[i][j] == 'small': order = 0
            elif size_batch[i][j] == 'midle': order = 1
            elif size_batch[i][j] == 'large': order = 2
            
            # 每个目标有多个Pred框
            for index in index_batch[i][j]:
                margin_pred.append(logits[index].item())
            
            # 更新最大边距
            if margin_max[order] < max(margin_pred): 
                margin_max[order] = max(margin_pred)
            
            # 排除负样本, 负样本logit值
            # 会不断减小, 而初始值为0
            if max(margin_pred) > 0:
                margin_list[order].append(max(margin_pred))
                margin_obj[order].append(max(margin_pred))
            else:
                margin_list[order].append('occupy')
                margin_obj[order].append('occupy')
    
    # 当本轮该尺度目标少于阈值时保留部分历史数据
    if len(obj_len[0]) < save_num: margin_list[0] = margin_list[0][-save_num:]
    if len(obj_len[1]) < save_num: margin_list[1] = margin_list[1][-save_num:]
    if len(obj_len[2]) < save_num: margin_list[2] = margin_list[2][-save_num:]
    
    # def compute_delta_gamma(margin_list, last_mean):
    #     current_mean = statistics.mean(margin_list)
    #     delta_gamma = current_mean - last_mean
    #     last_mean = current_mean
    #     max_m, min_m = max(margin_list), min(margin_list)
    #     return delta_gamma, last_mean, max_m, min_m

    # 计算△γ, △γ = μ[i]-μ[i-1]
    # if len(index) >= save_num: # 本轮该尺度目标大于阈值时直接使用所有本轮该尺度目标
    #     delta_gamma, last_mean, max_m, min_m = compute_delta_gamma(margin_list, last_mean)
    # else:
    #     if len(margin_list) >= save_num: # 本轮该尺度目标小于阈值时使用历史目标边距
    #         margin_list = margin_list[-save_num:]
    #     delta_gamma, last_mean, max_m, min_m = compute_delta_gamma(margin_list, last_mean)

    # 当本批次该尺度目标大于阈值时, 直接使用本批次的目标
    # 当本批次该尺度目标不够时, 使用之前批次储存下来的目标
    # if len(index) < save_num:
    #     if len(margin_list) >= save_num:
    #         margin_list = margin_list[-save_num:]
    
    # mean_m, max_m, min_m = statistics.mean(margin_list), max(margin_list), min(margin_list)

    # 获取current_margin, 这个值根据init_margin
    # 和init△γ与current△γ之间的比例计算而得
    # if△γ大于init△γ, 则直接使用init_margin
    # if当前△γ小于0, 则此批次△γ等于上批次△γ
    # 设置init_margin为current_margin的上限
    # if delta_gamma > init_gamma: delta_gamma = init_gamma
    # if delta_gamma < 0: delta_gamma = 0
    # current_margin = delta_gamma * init_margin / init_gamma
    # if current_margin > init_margin: current_margin = init_margin

    # def decay_function(x, k=0.5, v=init_margin):
    #     if v == 2.00: m = 1
    #     if v == 1.25: m = 0.5
    #     if v == 1.00: m = 0.25
    #     y = m * math.exp(-k * x)
    #     y = min(v, y) # 当x<0时y>1, 因此将其限制在1
    #     return y

    # current_margin = decay_function(mean_m)

    return margin_list, margin_obj, margin_max

def margin_dynamic_adding(logits, index_batch, size_batch, margin_list, margin_obj, max_m, init_margin_list):
    # initialize some parameters
    counter_s, counter_m, counter_l = 0, 0, 0

    # 计算三个尺度的margin_list的均值
    margin_mean = [[i for i in list_i if isinstance(i, (int, float))] for list_i in margin_list]
    margin_mean = [statistics.mean(list_i) if len(list_i) != 0 else 0 for list_i in margin_mean]

    # 每个批次有多张图片
    for i in range(len(index_batch)):
        # 每张图片有多个目标
        for j in range(len(index_batch[i])):

            # 判断单个目标的尺度并赋予相应的初始值
            if size_batch[i][j] == 'small':
                init_margin, order, counter = init_margin_list[0], 0, counter_s
                counter_s += 1
            elif size_batch[i][j] == 'midle':
                init_margin, order, counter = init_margin_list[1], 1, counter_m
                counter_m += 1
            elif size_batch[i][j] == 'large': 
                init_margin, order, counter = init_margin_list[2], 2, counter_l
                counter_l += 1

            # 若当前样本最大边距小于0则说明此样本为负样本
            if margin_obj[order][counter] != 'occupy':

                # 这里的问题是训练后期一些GT的所有Pred框的logit均为负值, 在这种机制下会给这些
                # GT赋予较大的损失值, 而这些GT的所有Pred框都不正确, 导致负向优化

                # 并且margin_list的均值居然随训练变成负数, 越来越小, 在这种机制下给正样本施加
                # 的优化比负样本更小, 这是不对的

                # 获取每个样本的边距偏移量, 并以此计算每个样本最终需要减去的边距值, 
                # 其中init_margin锚定均值, 为了防止某个样本的值起飞, 必须使用max和min
                delta_margin = (margin_mean[order]-margin_obj[order][counter]) / max_m[order]
                margin = init_margin if delta_margin < 0 else (1 + delta_margin) * init_margin

                # 筛选出其中最大的Logit值, 并且将其赋给其他Pred
                for k in index_batch[i][j]: logits[k] = margin_obj[order][counter]

                # 给所有Pred框的Logit减去init_margin
                for k in index_batch[i][j]: logits[k] -= margin
            
            else:

                # 必须给负样本进行最大logit赋给其他Pred操作, 这样能降低负样本产生的损失值, 
                # 还可以给负样本加上margin来减小其损失值
                margin_max = max([logits[k].item() for k in index_batch[i][j]])
                for k in index_batch[i][j]: logits[k] = margin_max + init_margin

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

def sca_weighting(loss_cls, gt_num, small_index, midle_index, large_index, iter_num, max_iter):
    def exp_decrease(coef, epoch, total):
        A = (1 - 1/coef) / (1 - math.exp(-2))
        ratio = A * (1 - math.exp(-2/total * (total-epoch))) + 1/coef
        return coef * ratio
    
    def exp_increase(coef, epoch, total):
        ratio = (1 / coef - 1) * (1 - np.exp(-3/total * epoch)) + 1
        return coef * ratio

    num_small, num_large = gt_num[0], gt_num[2]
    
    midle_index = small_index.copy()
    midle_index.extend(large_index)
    midle_index = torch.tensor(midle_index).to(loss_cls.device)

    loss_cls_small = loss_cls[small_index].sum()
    loss_cls_midle = loss_cls[~midle_index].sum()
    loss_cls_large = loss_cls[large_index].sum()

    if num_large != 0:
        if num_large / (num_large+num_small) > 1:
            ratio = exp_decrease((num_large / (num_large+num_small)), iter_num, max_iter)
        elif num_large / (num_large+num_small) < 1:
            ratio = exp_increase((num_large / (num_large+num_small)), iter_num, max_iter)
        else: ratio = 1
        loss_cls_small = loss_cls_small * ratio
    
    if num_small != 0:
        if num_small / (num_large+num_small) > 1:
            ratio = exp_decrease((num_small / (num_large+num_small)), iter_num, max_iter)
        elif num_small / (num_large+num_small) < 1:
            ratio = exp_increase((num_small / (num_large+num_small)), iter_num, max_iter)
        else: ratio = 1
        loss_cls_large = loss_cls_large * ratio
    
    loss_cls = loss_cls_small + loss_cls_midle + loss_cls_large

    return loss_cls

def scb_weighting(loss_cls, index_batch, size_batch, iter_num, max_iter):
    """
    Scale-Balanced Loss for Object Detection Pattern Recognition 2021 BUPT
    - scb_weight = 3 * (math.log(Mi) + 1) / Mi # Mi is pred num an obj can match
    - this function must placed after main loss function since it is an assistent
    """
    def exp_decrease(coef, epoch, total):
        A = (1 - 1/coef) / (1 - math.exp(-2))
        ratio = A * (1 - math.exp(-2/total * (total-epoch))) + 1/coef
        return coef * ratio
    
    def exp_increase(coef, epoch, total):
        ratio = (1 / coef - 1) * (1 - np.exp(-3/total * epoch)) + 1
        return coef * ratio

    # 每个批次有多张图片
    for i in range(len(index_batch)):
        # 每张图片有多个目标
        for j in range(len(index_batch[i])):

            # 计算为每个GT框乘以的权重
            scb_weight = 3 * (math.log(len(index_batch[i][j]))+1) / len(index_batch[i][j])

            # 小目标scb_weight: 大于1时不下降, 小于1时提升;
            # 大目标scb_weight: 大于1时下降, 小于1时不提升
            if scb_weight > 1:
                if size_batch[i][j] != 'small':
                    scb_weight = exp_decrease(scb_weight, iter_num, max_iter)
            elif scb_weight < 1:
                if size_batch[i][j] == 'small':
                    scb_weight = exp_increase(scb_weight, iter_num, max_iter)

            # 每个目标有多个Pred框, 为每个框都乘以权重
            for k in index_batch[i][j]:
                loss_cls[k] = loss_cls[k] * scb_weight
    
    loss_cls = loss_cls.sum()
    
    return loss_cls

def margin_loss_normalize(loss_cls, index_batch):
    # 每个批次有多张图片
    for i in range(len(index_batch)):
        # 每张图片有多个目标
        for j in range(len(index_batch[i])):
            # 每个目标有多个Pred框
            for k in index_batch[i][j]:
                loss_cls[k] *= 1 / len(index_batch[i][j])

    return loss_cls