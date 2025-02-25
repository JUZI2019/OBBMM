_base_ = [
    # '../_base_/models/cascade_rcnn_r50_fpn.py',
    '../_base_/datasets/sen1ship_dual_branch.py',
    '../_base_/schedules/schedule_6x.py',
    '../_base_/default_runtime.py'
]
# '../_base_/datasets/aitodv2_detection.py',
angle_version = 'le90'
# model settings
model = dict(
    type='OrientedRCNNDualBranch',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),
    rpn_head=dict(
        type='OrientedRPNHead',
        in_channels=256,
        feat_channels=256,
        version=angle_version,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='MidpointOffsetCoder',
            angle_range=angle_version,
            target_means=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            target_stds=[1.0, 1.0, 1.0, 1.0, 0.5, 0.5]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(
            type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0)),
    roi_head=dict(
        type='Cascade_t2t_new_jit_mask_RoIHead',
        num_stages=3,
        stage_loss_weights=[1, 0.5, 0.25],
        bbox_roi_extractor=dict(
            type='RotatedSingleRoIExtractor',
            roi_layer=dict(
                type='RoIAlignRotated',
                out_size=7,
                sample_num=2,
                clockwise=True),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=[
            dict(
                type='RotatedShared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=1,
                bbox_coder=dict(
                    type='DeltaXYWHAOBBoxCoder',
                    angle_range=angle_version,
                    norm_factor=None,
                    edge_swap=True,
                    proj_xy=True,
                    target_means=(.0, .0, .0, .0, .0),
                    # target_stds=(0.1, 0.1, 0.2, 0.2, 0.1)),
                    target_stds=(0.1, 0.1, 0.3, 0.3, 0.1)),
                    reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0)),
            dict(
                type='RotatedShared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=1,
                bbox_coder=dict(
                    type='DeltaXYWHAOBBoxCoder',
                    angle_range=angle_version,
                    norm_factor=None,
                    edge_swap=True,
                    proj_xy=True,
                    target_means=(.0, .0, .0, .0, .0),
                    # target_stds=(0.05, 0.05, 0.1, 0.1, 0.05)),
                    target_stds=(0.05, 0.05, 0.15, 0.15, 0.05)),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0)),
            dict(
                type='RotatedShared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=1,
                bbox_coder=dict(
                    type='DeltaXYWHAOBBoxCoder',
                    angle_range=angle_version,
                    norm_factor=None,
                    edge_swap=True,
                    proj_xy=True,
                    target_means=(.0, .0, .0, .0, .0),
                    # target_stds=(0.033, 0.033, 0.067, 0.067, 0.033)),
                    target_stds=(0.0333, 0.0333, 0.1, 0.1, 0.0333)),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0)),
        ]),
    # model training and testing settings
    train_cfg=dict(
        # rpn=dict(
        #     assigner=dict(
        #         type='MaxIoUAssigner',
        #         pos_iou_thr=0.7,
        #         neg_iou_thr=0.3,
        #         min_pos_iou=0.3,
        #         match_low_quality=True,
        #         ignore_iof_thr=-1),
        #     sampler=dict(
        #         type='RandomSampler',
        #         num=256,
        #         pos_fraction=0.5,
        #         neg_pos_ub=-1,
        #         add_gt_as_proposals=False),
        #     allowed_border=0,
        #     pos_weight=-1,
        #     debug=False),
        # rpn_proposal=dict(
        #     nms_pre=2000,
        #     max_per_img=2000,
        #     nms=dict(type='nms', iou_threshold=0.8),
        #     min_bbox_size=0),
        rpn=dict(
            assigner=dict(
                type='RankingAssigner',
                ignore_iof_thr=-1,
                gpu_assign_thr=512,
                iou_calculator=dict(type='BboxDistanceMetric'),
                assign_metric='nwd',
                topk=5),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=0,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_pre=3000,
            max_per_img=3000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=[            
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.5,
                    neg_iou_thr=0.5,
                    min_pos_iou=0.5,
                    match_low_quality=False,
                    iou_calculator=dict(type='RBboxOverlaps2D'),
                    ignore_iof_thr=-1,
                    gpu_assign_thr=256),
                sampler=dict(
                    type='RRandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                pos_weight=-1,
                debug=False),
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.6,
                    neg_iou_thr=0.6,
                    min_pos_iou=0.6,
                    match_low_quality=False,
                    iou_calculator=dict(type='RBboxOverlaps2D'),
                    ignore_iof_thr=-1,
                    gpu_assign_thr=256),
                sampler=dict(
                    type='RRandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                pos_weight=-1,
                debug=False),
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.7,
                    neg_iou_thr=0.7,
                    min_pos_iou=0.7,
                    match_low_quality=False,
                    iou_calculator=dict(type='RBboxOverlaps2D'),
                    ignore_iof_thr=-1,
                    gpu_assign_thr=256),
                sampler=dict(
                    type='RRandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                pos_weight=-1,
                debug=False)
        ]),
    test_cfg=dict(
        rpn=dict(
            nms_pre=3000,
            max_per_img=3000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            nms_pre=2000,
            min_bbox_size=0,
            score_thr=0.05,
            nms=dict(iou_thr=0.1),
            max_per_img=2000),
        # rcnn=dict(
        #     score_thr=0.05,
        #     nms=dict(iou_thr=0.5),
        #     max_per_img=3000)
            ))


# evaluation = dict(interval=1, metric='mAP')
# optimizer = dict(type='AdamW' ,lr=0.000125, weight_decay=0.0001)
# optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# lr_config = dict(
#     policy='step',
#     warmup='linear',
#     warmup_iters=500,
#     warmup_ratio=1.0 / 3,
#     step=[48, 66])
# runner = dict(type='EpochBasedRunner', max_epochs=72)
# checkpoint_config = dict(interval=1)
optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=5000,
    warmup_ratio=0.001,
    step=[32, 44])
checkpoint_config = dict(interval=1)

# evaluation
load_from = "/workstation1/fyy1/mm_runs/oriented_rcnn_TransRCNN_dual_branch_r50_fpn_sen1ship_le90_old/best_29.pth"
# load_from = None
# load_from = "/mnt/data0/Garmin/nwd-rka/mmdet-nwdrka/work_dirs/pretrain/base_24.pth"
# load_from = "/mnt/data0/Garmin/nwd-rka/mmdet-nwdrka/work_dirs/RS_cl_two_stage/e12_mAP251.pth"

# resume_from = "/home/hoiliu/Desktop/DNTR/mmdet-dntr/work_dirs/aitod_DNTR_mask/latest.pth"
# resume_from = None
# resume_from = "/mnt/data0/Garmin/nwd-rka/mmdet-nwdrka/work_dirs/RS_cl_two_stage/e12_mAP251.pth"
