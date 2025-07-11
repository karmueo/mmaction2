_base_ = ['../../_base_/default_runtime.py']

load_from = '/home/tl/work/mmaction2/output/uniformerv2/best_epoch_84.pth'


# model settings
num_frames = 32
model = dict(
    type='Recognizer3D',
    backbone=dict(
        type='UniFormerV2',
        input_resolution=224,
        patch_size=16,
        width=768,
        layers=12,
        heads=12,
        t_size=num_frames,
        dw_reduction=1.5,
        temporal_downsample=False,
        no_lmhra=True,
        double_lmhra=True,
        return_list=[8, 9, 10, 11],
        n_layers=4,
        n_dim=768,
        n_head=12,
        mlp_factor=4.,
        drop_path_rate=0.,
        mlp_dropout=[0., 0., 0., 0.],
        clip_pretrained=True,
        pretrained='ViT-B/16'),
    cls_head=dict(
        type='UniFormerHead',
        dropout_ratio=0.,
        num_classes=2,
        in_channels=768,
        average_clips='prob'),
    data_preprocessor=dict(
        type='ActionDataPreprocessor',
        mean=[114.75, 114.75, 114.75],
        std=[57.375, 57.375, 57.375],
        format_shape='NCTHW'))

# dataset settings
dataset_type = 'RawframeDataset'
data_root_train = '/home/tl/data/datasets/mmaction2/110_video_frames_60/train_all'
data_root_val = '/home/tl/data/datasets/mmaction2/110_video_frames/val'
data_root_test = '/home/tl/data/datasets/mmaction2/110_video_frames/test/test_data'
ann_file_train = '/home/tl/data/datasets/mmaction2/110_video_frames_60/train_all/110_video_train_annotation_file.txt'
ann_file_val = '/home/tl/data/datasets/mmaction2/110_video_frames/val/110_video_val_annotation_file.txt'
ann_file_test = '/home/tl/data/datasets/mmaction2/110_video_frames/test/test_data/110_video_test_annotation_file.txt'
filename_tmpl = '{:d}.jpg'

file_client_args = dict(io_backend='disk')

train_pipeline = [
    dict(type='UniformSample', clip_len=num_frames, num_clips=1),
    dict(type='RawFrameDecode', **file_client_args),
    dict(type='Resize', scale=(224, 224), keep_ratio=True),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]

val_pipeline = [
    dict(
        type='UniformSample', clip_len=num_frames, num_clips=1,
        test_mode=True),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(224, 224), keep_ratio=True),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]

test_pipeline = [
    dict(
        type='UniformSample', clip_len=num_frames, num_clips=4,
        test_mode=True),
    dict(type='RawFrameDecode'),
    dict(type='FormatShape', input_format='NCTHW'),
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
    batch_size=12,
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
test_evaluator = dict(type='AccMetric')
train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=50, val_begin=1, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

optim_wrapper = dict(
    optimizer=dict(
        type='SGD',
        lr=0.005,
        momentum=0.9,
        weight_decay=0.0001),
    clip_grad=dict(max_norm=40, norm_type=2))

param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=50,
        by_epoch=True,
        milestones=[20, 40],
        gamma=0.1)
]

default_hooks = dict(
    checkpoint=dict(interval=3, max_keep_ckpts=5), logger=dict(interval=100))

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (8 samples per GPU).
auto_scale_lr = dict(enable=True, base_batch_size=128)
