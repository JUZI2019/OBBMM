# dataset settings
dataset_type = 'Sen1shipDataset'
data_root = '/workstation/fyy/multi_scale/sen1ship_dota_vhbg_608_ms_train/vh/'
# data_root = '/workstation/fyy/NEW_SEN1SHIP_dataset/split/608_ms/'
data_root_test = '/workstation/fyy/sen1ship_new_dota_vhbg_608_single_2/vh/'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RResize', img_scale=(608, 608)),
    dict(type='RRandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(608, 608),
        flip=False,
        transforms=[
            dict(type='RResize'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=4,
    train=dict(
        type = dataset_type,
        ann_file = data_root + 'train/labelTxt/',
        img_prefix = data_root + 'train/images/',
        pipeline = train_pipeline),
    val=dict(
        type = dataset_type,
        ann_file = data_root_test + 'test/labelTxt/',
        img_prefix = data_root_test + 'test/images/',
        pipeline = test_pipeline),
    test=dict(
        type = dataset_type,
        ann_file = data_root_test + 'test/labelTxt/',
        img_prefix = data_root_test + 'test/images/',
        pipeline = test_pipeline))
