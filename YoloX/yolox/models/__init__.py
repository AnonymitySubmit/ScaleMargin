#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

#from .darknet import CSPDarknet, Darknet
#from .losses import IOUloss
#from .yolo_fpn import YOLOFPN
from .yolo_head_rewrite import YOLOXHead
from .yolo_pafpn_rewrite import YOLOPAFPN
from .yolox_rewrite import YOLOX
from .yolox_ERF import YOLOXERF