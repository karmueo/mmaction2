_base_ = ['../../_base_/models/tsm_r50.py', '../../_base_/default_runtime.py']


# model settings
model = dict(backbone=dict(num_segments=32), cls_head=dict(num_segments=32))

# dataset settings
dataset_type = 'RawframeDataset'
data_root_train = '/home/tl/data/datasets/mmaction2/no_keep_ratio/train'
data_root_val = '/home/tl/data/datasets/mmaction2/no_keep_ratio/val'
data_root_test = '/home/tl/data/datasets/mmaction2/no_keep_ratio/val'
ann_file_train = '/home/tl/data/datasets/mmaction2/no_keep_ratio/train/110_video_train_annotation_file.txt'
ann_file_val = '/home/tl/data/datasets/mmaction2/no_keep_ratio/val/110_video_train_annotation_file.txt'
ann_file_test = '/home/tl/data/datasets/mmaction2/no_keep_ratio/val/110_video_train_annotation_file.txt'
filename_tmpl = '{:d}.jpg'

file_client_args = dict(io_backend='disk')
train_pipeline = [
    dict(type='SampleFrames', clip_len=1, frame_interval=5, num_clips=32),
    dict(type='RawFrameDecode', **file_client_args),
    dict(type='Resize', scale=(224, 224), keep_ratio=True),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='PackActionInputs')
]

val_pipeline = [
    dict(
        type='SampleFrames',
        clip_len=1,
        frame_interval=5,
        num_clips=32,
        test_mode=True),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(224, 224), keep_ratio=True),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='PackActionInputs')
]

test_pipeline = [
    dict(
        type='SampleFrames',
        clip_len=3,
        frame_interval=1,
        num_clips=32,
        test_mode=True),
    dict(type='RawFrameDecode'),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='PackActionInputs')
]

train_dataloader = dict(
    batch_size=12,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        start_index=0,
        ann_file=ann_file_train,
        data_prefix=dict(img=data_root_train),
        filename_tmpl=filename_tmpl,
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=8,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        start_index=0,
        ann_file=ann_file_val,
        data_prefix=dict(img=data_root_val),
        filename_tmpl=filename_tmpl,
        pipeline=val_pipeline,
        test_mode=True))

test_dataloader = dict(
    batch_size=1,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        start_index=0,
        ann_file=ann_file_test,
        data_prefix=dict(img=data_root_test),
        filename_tmpl=filename_tmpl,
        pipeline=test_pipeline,
        test_mode=True))

val_evaluator = dict(type='AccMetric')
test_evaluator = val_evaluator

default_hooks = dict(checkpoint=dict(interval=3, max_keep_ckpts=3))

train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=50, val_begin=1, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

param_scheduler = [
    dict(type='LinearLR', start_factor=0.1, by_epoch=True, begin=0, end=5),
    dict(
        type='MultiStepLR',
        begin=0,
        end=50,
        by_epoch=True,
        milestones=[25, 45],
        gamma=0.1)
]

optim_wrapper = dict(
    constructor='TSMOptimWrapperConstructor',
    paramwise_cfg=dict(fc_lr5=True),
    optimizer=dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001),
    clip_grad=dict(max_norm=20, norm_type=2))

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (16 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=128)
