# Copyright (c) OpenMMLab. All rights reserved.
from mmcv_new.utils import build_from_cfg
from mmdet_new.core.bbox.iou_calculators.builder import IOU_CALCULATORS

ROTATED_IOU_CALCULATORS = IOU_CALCULATORS


def build_iou_calculator(cfg, default_args=None):
    """Builder of IoU calculator."""
    return build_from_cfg(cfg, ROTATED_IOU_CALCULATORS, default_args)
