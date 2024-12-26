# Copyright (c) Facebook, Inc. and its affiliates.

from .pascal_voc import load_voc_instances, register_pascal_voc
from .lvis import load_lvis_json, register_lvis_instances, get_lvis_instances_meta
from .coco_panoptic import register_coco_panoptic, register_coco_panoptic_separated
from .coco import load_coco_json, load_sem_seg, register_coco_instances, convert_to_coco_json

# ensure the builtin datasets are registered
from . import builtin as _builtin

__all__ = [k for k in globals().keys() if not k.startswith("_")]
