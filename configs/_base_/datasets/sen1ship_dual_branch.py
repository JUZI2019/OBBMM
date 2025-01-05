# dataset settings
dataset_type = 'Sen1shipDualBranchDataset'
data_root = '/workstation/fyy/sen1ship_dota_vhbg_608_single_2/vh'
data_root_bg = '/workstation/fyy/sen1ship_dota_vhbg_608_single_2/vhbg'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile', key='img_branch_1'),
    dict(type='LoadImageFromFile', key='img_branch_2'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RResize', img_scale=(608, 608)),
    dict(type='RRandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img_branch_1', 'img_branch_2', 'gt_bboxes', 'gt_labels'])
]
test_pipeline = [
    dict(type='LoadImageFromFile', key='img_branch_1'),
    dict(type='LoadImageFromFile', key='img_branch_2'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(608, 608),
        flip=False,
        transforms=[
            dict(type='RResize'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img_branch_1', 'img_branch_2',])
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file = data_root + 'train/labelTxt/',
        img_branch_1_prefix = data_root + 'train/images/',
        img_branch_2_prefix = data_root_bg + 'images/',        
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'test/labelTxt/',
        img_branch_1_prefix = data_root + 'test/images/',
        img_branch_2_prefix = data_root_bg + 'images/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'test/labelTxt/',
        img_branch_1_prefix = data_root + 'test/images/',
        img_branch_2_prefix = data_root_bg + 'images/',
        pipeline=test_pipeline))
