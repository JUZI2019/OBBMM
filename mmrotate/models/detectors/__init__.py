# Copyright (c) OpenMMLab. All rights reserved.
from .base import RotatedBaseDetector
from .gliding_vertex import GlidingVertex
from .oriented_rcnn import OrientedRCNN
from .r3det import R3Det
from .redet import ReDet
from .roi_transformer import RoITransformer
from .rotate_faster_rcnn import RotatedFasterRCNN
from .rotated_fcos import RotatedFCOS
from .rotated_reppoints import RotatedRepPoints
from .rotated_retinanet import RotatedRetinaNet
from .s2anet import S2ANet
from .single_stage import RotatedSingleStageDetector
from .two_stage import RotatedTwoStageDetector
# add
from .two_stage_dntr import RotatedTwoStageDetector_dntr
from .cascade_rcnn_dntr import CascadeRCNN_dntr
from .oriented_rcnn_dual_branch import OrientedRCNNDualBranch
from .dual_branch_mmroate_base import Dual_Branch_RotatedBaseDetector
from .two_stage_dual_branch import Dual_Branch_RotatedTwoStageDetector

__all__ = [
    'RotatedRetinaNet', 'RotatedFasterRCNN', 'OrientedRCNN', 'RoITransformer',
    'GlidingVertex', 'ReDet', 'R3Det', 'S2ANet', 'RotatedRepPoints',
    'RotatedBaseDetector', 'RotatedTwoStageDetector',
    'RotatedSingleStageDetector', 'RotatedFCOS',

    # add
    'RotatedTwoStageDetector_dntr','CascadeRCNN_dntr','OrientedRCNNDualBranch',
    'Dual_Branch_RotatedBaseDetector','Dual_Branch_RotatedTwoStageDetector',
]
