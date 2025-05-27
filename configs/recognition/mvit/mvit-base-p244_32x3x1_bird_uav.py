_base_ = [
    '../../_base_/models/mvit_small.py', '../../_base_/default_runtime.py'
]

model = dict(
    backbone=dict(
        arch='base',
        temporal_size=32,
        drop_path_rate=0.3,
    ),
    data_preprocessor=dict(
        type='ActionDataPreprocessor',
        mean=[114.75, 114.75, 114.75],
        std=[57.375, 57.375, 57.375],
        blending=dict(
            type='RandomBatchAugment',
            augments=[
                dict(type='MixupBlending', alpha=0.8, num_classes=2),
                dict(type='CutmixBlending', alpha=1, num_classes=2)
            ]),
        format_shape='NCTHW'),
)

# dataset settings
dataset_type = 'RawframeDataset'
data_root_train = '/home/tl/data/datasets/mmaction2/110_video_frames/train'
data_root_val = '/home/tl/data/datasets/mmaction2/110_video_frames/val'
data_root_test = '/home/tl/data/datasets/mmaction2/110_video_frames/test'
ann_file_train = '/home/tl/data/datasets/mmaction2/110_video_frames/train/110_video_train_annotation_file.txt'
ann_file_val = '/home/tl/data/datasets/mmaction2/110_video_frames/val/110_video_val_annotation_file.txt'
ann_file_test = '/home/tl/data/datasets/mmaction2/110_video_frames/test/110_video_test_annotation_file.txt'
filename_tmpl = '{:d}.jpg'

file_client_args = dict(io_backend='disk')
train_pipeline = [
    dict(type='SampleFrames', clip_len=32, frame_interval=3, num_clips=1),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(224, 224), keep_ratio=True),
    dict(type='Flip', flip_ratio=0.5),
    # dict(type='RandomErasing', erase_prob=0.25, mode='rand'),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]
val_pipeline = [
    dict(
        type='SampleFrames',
        clip_len=32,
        frame_interval=3,
        num_clips=1,
        test_mode=True),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(224, 224), keep_ratio=True),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]
test_pipeline = [
    dict(
        type='SampleFrames',
        clip_len=32,
        frame_interval=2,
        num_clips=1,
        test_mode=True),
    dict(type='RawFrameDecode'),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]

train_dataloader = dict(
    batch_size=2,
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
    batch_size=2,
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

train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=200, val_begin=1, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

base_lr = 1.6e-3
optim_wrapper = dict(
    optimizer=dict(
        type='AdamW', lr=base_lr, betas=(0.9, 0.999), weight_decay=0.05),
    paramwise_cfg=dict(norm_decay_mult=0.0, bias_decay_mult=0.0),
    clip_grad=dict(max_norm=1, norm_type=2))

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.01,
        by_epoch=True,
        begin=0,
        end=30,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        T_max=200,
        eta_min=base_lr / 100,
        by_epoch=True,
        begin=30,
        end=200,
        convert_to_iter_based=True)
]

default_hooks = dict(
    checkpoint=dict(interval=3, max_keep_ckpts=5), logger=dict(interval=100))

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (8 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=128)
