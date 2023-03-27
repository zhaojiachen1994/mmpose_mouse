_base_ = [
    '../_base_/default_runtime.py',
    '../_base_/mouse_datasets/mouse_p5.py'
]

evaluation = dict(interval=5, metric='mAP', save_best='AP')

total_epochs = 50

optimizer = dict(
    type='Adam',
    lr=5e-4,
)
optimizer_config = dict(grad_clip=None)

# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[170, 200])

log_config = dict(
    interval=1,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])

# model config
model = dict(
    type='TopDown',
    pretrained='D:/Pycharm Projects-win/mm_mouse/mmpose/official_checkpoint/hrnet_w48-8ef0771d.pth',
    backbone=dict(
        type='HRNet',
        in_channels=3,
        extra=dict(
            stage1=dict(
                num_modules=1,
                num_branches=1,
                block='BOTTLENECK',
                num_blocks=(4,),
                num_channels=(64,)),
            stage2=dict(
                num_modules=1,
                num_branches=2,
                block='BASIC',
                num_blocks=(4, 4),
                num_channels=(48, 96)),
            stage3=dict(
                num_modules=4,
                num_branches=3,
                block='BASIC',
                num_blocks=(4, 4, 4),
                num_channels=(48, 96, 192)),
            stage4=dict(
                num_modules=3,
                num_branches=4,
                block='BASIC',
                num_blocks=(4, 4, 4, 4),
                num_channels=(48, 96, 192, 384))),
    ),
    keypoint_head=dict(
        type='TopdownHeatmapSimpleHead',
        in_channels=48,
        out_channels=5,
        num_deconv_layers=0,
        extra=dict(final_conv_kernel=1, ),
        loss_keypoint=dict(type='JointsMSELoss', use_target_weight=True)),
    train_cfg=dict(),
    test_cfg=dict(
        flip_test=True,
        post_process='default',
        shift_heatmap=True,
        modulate_kernel=11))

# pipeline config
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='TopDownGetBboxCenterScale', padding=1.25),
    dict(type='TopDownRandomShiftBboxCenter', shift_factor=0.16, prob=0.3),
    dict(type='TopDownRandomFlip', flip_prob=0.5),
    dict(
        type='TopDownHalfBodyTransform',
        num_joints_half_body=8,
        prob_half_body=0.3),
    dict(
        type='TopDownGetRandomScaleRotation', rot_factor=40, scale_factor=0.5),
    dict(type='TopDownAffine'),
    dict(type='ToTensor'),
    dict(
        type='NormalizeTensor',
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]),
    dict(type='TopDownGenerateTarget', sigma=2),
    dict(
        type='Collect',
        keys=['img', 'target', 'target_weight'],
        meta_keys=[
            'image_file', 'joints_3d', 'joints_3d_visible',
            'center', 'scale', 'rotation', 'bbox_score', 'flip_pairs'
        ]),
]

val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='TopDownGetBboxCenterScale', padding=1.25),
    dict(type='TopDownAffine'),
    dict(type='ToTensor'),
    dict(
        type='NormalizeTensor',
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]),
    dict(
        type='Collect',
        keys=['img'],
        meta_keys=[
            'image_file', 'center', 'scale', 'rotation', 'bbox_score',
            'flip_pairs', 'bbox'
        ]),
]

test_pipeline = val_pipeline

# mars dataset config
mars_data_root = "D:/Datasets/MARS-PoseAnnotationData"
mars_channel_cfg = dict(
    num_output_channels=5,
    dataset_joints=5,
    dataset_channel=[0, 1, 2, 3, 6],
    inference_channel=[0, 1, 2, 3, 6])

mars_data_cfg = dict(
    image_size=[256, 256],
    heatmap_size=[64, 64],
    num_output_channels=mars_channel_cfg['num_output_channels'],
    num_joints=mars_channel_cfg['dataset_joints'],
    dataset_channel=mars_channel_cfg['dataset_channel'],
    inference_channel=mars_channel_cfg['inference_channel'],
    soft_nms=False,
    nms_thr=1.0,
    oks_thr=0.9,
    vis_thr=0.2,
    use_gt_bbox=True,
    det_bbox_thr=0.0,
    bbox_file='',
    # space_size=[0.3, 0.3, 0.3],
    # space_center=[0, 0, 0.15],
    # cube_size=[0.1, 0.1, 0.1],
    # num_cameras=6,
    use_different_joint_weights=False
)

top_dataset_cfg = dict(
    type='MouseMars2dDataset',
    ann_file=f'{mars_data_root}/MARS_keypoints_top_black_hq.json',
    img_prefix=f'{mars_data_root}/raw_images_top/',
    data_cfg=mars_data_cfg,
    pipeline=train_pipeline,
    dataset_info={{_base_.dataset_info}})

front_dataset_cfg = dict(
    type='MouseMars2dDataset',
    ann_file=f'{mars_data_root}/MARS_keypoints_front_black_hq.json',
    img_prefix=f'{mars_data_root}/raw_images_front/',
    data_cfg=mars_data_cfg,
    pipeline=train_pipeline,
    dataset_info={{_base_.dataset_info}})

# dannce dataset config
dannce_data_root = "D:/Datasets/transfer_mouse/dannce_20230130"
dannce_channel_cfg = dict(
    num_output_channels=5,
    dataset_joints=5,
    dataset_channel=[2, 0, 1, 3, 5],
    inference_channel=[2, 0, 1, 3, 5])

data_cfg = dict(
    image_size=[256, 256],
    heatmap_size=[64, 64],
    num_output_channels=dannce_channel_cfg['num_output_channels'],
    num_joints=dannce_channel_cfg['dataset_joints'],
    dataset_channel=dannce_channel_cfg['dataset_channel'],
    inference_channel=dannce_channel_cfg['inference_channel'],
    soft_nms=False,
    nms_thr=1.0,
    oks_thr=0.9,
    vis_thr=0.2,
    use_gt_bbox=True,
    det_bbox_thr=0.0,
    bbox_file='',
    space_size=[0.3, 0.3, 0.3],
    space_center=[0, 0, 0.15],
    cube_size=[0.1, 0.1, 0.1],
    num_cameras=6,
    use_different_joint_weights=False
)

data = dict(
    samples_per_gpu=16,
    workers_per_gpu=2,
    val_dataloader=dict(samples_per_gpu=16),
    test_dataloader=dict(samples_per_gpu=16),
    train=dict(
        type='ConcatDataset',
        datasets=[top_dataset_cfg, front_dataset_cfg]),
    eval=dict(
        type='MouseDannce2dDatasetSview',
        ann_file=f'{dannce_data_root}/annotations_visible_eval930_new.json',
        img_prefix=f'{dannce_data_root}/images_gray/',
        data_cfg=data_cfg,
        pipeline=val_pipeline,
        dataset_info={{_base_.dataset_info}}),
    test=dict(
        type='MouseDannce2dDatasetSview',
        ann_file=f'{dannce_data_root}/annotations_visible_eval930_new.json',
        img_prefix=f'{dannce_data_root}/images_gray/',
        data_cfg=data_cfg,
        pipeline=val_pipeline,
        dataset_info={{_base_.dataset_info}}),
)
