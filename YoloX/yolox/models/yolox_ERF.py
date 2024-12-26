# -*- coding: utf-8 -*-

import torch.nn as nn
from .yolo_pafpn_rewrite import YOLOPAFPN


class YOLOXERF(nn.Module):

    def __init__(self, backbone=None, head=None):
        super().__init__()
        if backbone is None:
            backbone = YOLOPAFPN()
        self.backbone = backbone

    def forward(self, x, targets=None):
        # fpn output content features of [dark3, dark4, dark5]
        fpn_outs = self.backbone(x)
        
        return fpn_outs[2]