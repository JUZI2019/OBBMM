# dataset settings
dataset_type = 'Sen1shipDualBranchDataset'
data_root = '/workstation1/fyy1/sen2ship_dota/split/608_1x_2x/'
data_root_bg = '/workstation1/fyy1/sen2ship_dota/split/608_1x_2x/bg/'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadTwoBranchImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RResize', img_scale=(608, 608)),
    dict(type='RRandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'img_bg', 'gt_bboxes', 'gt_labels'])
]
test_pipeline = [
    dict(type='LoadTwoBranchImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(608, 608),
        flip=False,
        transforms=[
            dict(type='RResize'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'img_bg'])
        ])
]
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file = data_root + 'trainval/labelTxt/',
        img_prefix = data_root + 'trainval/images/',
        img_branch_2_prefix = data_root_bg ,        
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'test/labelTxt/',
        img_prefix = data_root + 'test/images/',
        img_branch_2_prefix = data_root_bg,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'test/labelTxt/',
        img_prefix = data_root + 'test/images/',
        img_branch_2_prefix = data_root_bg,
        pipeline=test_pipeline))
