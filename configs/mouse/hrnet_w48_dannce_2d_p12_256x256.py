_base_ = [
    '../_base_/default_runtime.py',
    '../_base_/mouse_datasets/mouse_dannce_p12.py'
]

evaluation = dict(interval=2, metric='mAP', save_best='AP')

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
total_epochs = 5
log_config = dict(
    interval=1,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])

channel_cfg = dict(
    num_output_channels=12,
    dataset_joints=12,
    dataset_channel=[
        0, 1, 2, 3, 4, 5, 6, 7, 9, 13, 16, 19
    ],
    inference_channel=[
        0, 1, 2, 3, 4, 5, 6, 7, 9, 13, 16, 19
    ])

# model settings
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
        out_channels=channel_cfg['num_output_channels'],
        num_deconv_layers=0,
        extra=dict(final_conv_kernel=1, ),
        loss_keypoint=dict(type='JointsMSELoss', use_target_weight=True)),
    train_cfg=dict(),
    test_cfg=dict(
        flip_test=True,
        post_process='default',
        shift_heatmap=True,
        modulate_kernel=11))

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
            # 'center', 'scale', 'rotation', 'bbox_score', 'flip_pairs'
        ]),
]

eval_pipeline = [
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

test_pipeline = eval_pipeline

data_root = 'D:/Datasets/transfer_mouse/dannce_20230130'
data = dict(
    samples_per_gpu=32,
    workers_per_gpu=2,
    val_dataloader=dict(samples_per_gpu=32),
    test_dataloader=dict(samples_per_gpu=32),
    train=dict(
        type='MouseDannce2dDatasetSview',
        ann_file=f'{data_root}/annotations_visible_train930_new.json',
        # ann_3d_file=f'{data_root}/joints_3d.json',
        # cam_file=f'{data_root}/cams.pkl',
        img_prefix=f'{data_root}/images_gray/',
        data_cfg=data_cfg,
        pipeline=train_pipeline,
        dataset_info={{_base_.dataset_info}}),

    eval=dict(
        type='MouseDannce2dDatasetSview',
        ann_file=f'{data_root}/annotations_visible_eval930_new.json',
        # ann_3d_file=f'{data_root}/joints_3d.json',
        # cam_file=f'{data_root}/cams.pkl',
        img_prefix=f'{data_root}/images_gray/',
        data_cfg=data_cfg,
        pipeline=eval_pipeline,
        dataset_info={{_base_.dataset_info}}),

    test=dict(
        type='MouseDannce2dDatasetSview',
        ann_file=f'{data_root}/annotations_visible_eval930_new.json',
        # ann_3d_file=f'{data_root}/joints_3d.json',
        # cam_file=f'{data_root}/cams.pkl',
        img_prefix=f'{data_root}/images_gray/',
        data_cfg=data_cfg,
        pipeline=eval_pipeline,
        dataset_info={{_base_.dataset_info}}),
)
