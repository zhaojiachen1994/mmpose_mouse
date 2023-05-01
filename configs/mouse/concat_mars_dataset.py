_base_ = [
    '../_base_/default_runtime.py',
    '../_base_/mouse_datasets/mouse_mars_p5.py'
]

channel_cfg = dict(
    num_output_channels=5,
    dataset_joints=5,
    dataset_channel=[
        0, 1, 2, 3, 6
    ],
    inference_channel=[
        0, 1, 2, 3, 6
    ])

data_cfg = dict(
    image_size=[256, 256],
    heatmap_size=[64, 64],
    num_output_channels=channel_cfg['num_output_channels'],
    num_joints=channel_cfg['dataset_joints'],
    dataset_channel=channel_cfg['dataset_channel'],
    inference_channel=channel_cfg['inference_channel'],
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

# train pipeline config
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
            'flip_pairs'
        ]),
]

test_pipeline = val_pipeline

data_root = "D:/Datasets/MARS-PoseAnnotationData"

top_dataset_cfg = dict(
    type='MouseMars2dDataset',
    ann_file=f'{data_root}/MARS_keypoints_top_black.json',
    img_prefix=f'{data_root}/raw_images_top/',
    data_cfg=data_cfg,
    pipeline=train_pipeline,
    dataset_info={{_base_.dataset_info}})

front_dataset_cfg = dict(
    type='MouseMars2dDataset',
    ann_file=f'{data_root}/MARS_keypoints_front_black.json',
    img_prefix=f'{data_root}/raw_images_front/',
    data_cfg=data_cfg,
    pipeline=train_pipeline,
    dataset_info={{_base_.dataset_info}})

data = dict(
    samples_per_gpu=16,
    workers_per_gpu=2,
    val_dataloader=dict(samples_per_gpu=32),
    test_dataloader=dict(samples_per_gpu=32),
    train=dict(
        type='ConcatDataset',
        datasets=[top_dataset_cfg, front_dataset_cfg]
    ),
)