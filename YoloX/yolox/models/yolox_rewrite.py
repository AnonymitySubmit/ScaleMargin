# -*- coding: utf-8 -*-

import torch.nn as nn

from .yolo_head_rewrite import YOLOXHead
from .yolo_pafpn_rewrite import YOLOPAFPN


class YOLOX(nn.Module):

    def __init__(self, backbone=None, head=None):
        super().__init__()
        if backbone is None:
            backbone = YOLOPAFPN()
        if head is None:
            head = YOLOXHead(80)

        self.backbone = backbone
        self.head = head

    def forward(self, x, targets=None, cls_list=None, cls_max=None, cls_min=None, obj_list=None, obj_max=None, obj_min=None, current_epoch=None, total_epoch=None):
        # fpn output content features of [dark3, dark4, dark5]
        fpn_outs = self.backbone(x)

        if self.training:
            assert targets is not None
            # loss, iou_loss, conf_loss, cls_loss, l1_loss, num_fg, ratio_scale, loss_iou, loss_cls, loss_obj, margin, loss_cls_scale = self.head(fpn_outs, targets, x)
            # outputs = {"total_loss" : loss,
            #            "iou_loss"   : iou_loss,
            #            "l1_loss"    : l1_loss,
            #            "conf_loss"  : conf_loss,
            #            "cls_loss"   : cls_loss,
            #            "num_fg"     : num_fg,
                       
            #            "ratio_scale": ratio_scale,
            #            "loss_iou"   : loss_iou, 
            #            "loss_cls"   : loss_cls,
            #            "loss_obj"   : loss_obj,
            #            "margin"     : margin, 
            #            "cls_scale"  : loss_cls_scale}

            loss, iou_loss, conf_loss, cls_loss, l1_loss, num_fg, cls_list, cls_max, cls_min, obj_list, obj_max, obj_min = self.head(fpn_outs, targets, x, cls_list, cls_max, cls_min, 
                                                                                                                                                           obj_list, obj_max, obj_min, 
                                                                                                                                                           current_epoch, total_epoch)
            outputs = {"total_loss" : loss,
                       "iou_loss"   : iou_loss,
                       "l1_loss"    : l1_loss,
                       "conf_loss"  : conf_loss,
                       "cls_loss"   : cls_loss,
                       "num_fg"     : num_fg, 
                       "cls_list"   : cls_list, 
                       "cls_max"    : cls_max, 
                       "cls_min"    : cls_min,
                       "obj_list"   : obj_list, 
                       "obj_max"    : obj_max, 
                       "obj_min"    : obj_min}

            # scm, scb, sca, ratio = self.head(fpn_outs, targets, x, cls_list, cls_max, cls_min, obj_list, obj_max, obj_min, current_epoch, total_epoch)
            # outputs = {"ratio": ratio,
            #            "scm" : scm,
            #            "scb" : scb,
            #            "sca" : sca}
        else:
            outputs = self.head(fpn_outs)
        return outputs