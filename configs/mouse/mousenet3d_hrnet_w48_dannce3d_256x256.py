_base_ = [
    '../_base_/default_runtime.py',
    '../_base_/mouse_datasets/mouse_dannce_3d.py'
]

# evaluation config
evaluation = dict(interval=10, metric='mAP', save_best='AP')

# optimizer config
optimizer = dict(
    type='Adam',
    lr=5e-4,
)
optimizer_config = dict(grad_clip=None)

# learning policy config
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[170, 200])
total_epochs = 5

# log config
log_config = dict(
    interval=1,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])

# joint channel config
channel_cfg = dict(
    num_output_channels=22,
    dataset_joints=22,
    dataset_channel=[
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
    ],
    inference_channel=[
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21
    ])

"""total model pretrain weights"""
# load_from = "D:/Pycharm Projects-win/mmpose/work_dirs/my_hrnet_w48_dannce_256x256/latest.pth"

"""model config"""
model = dict(
    type='MouseNet_3d',
    # pretrained="D:\Pycharm Projects-win\mmpose\checkpoints\hrnet_w48-8ef0771d.pth",
    # pretrained="work_dirs/hrnet_gray/hrnet_w48_mouse_dannce_256x256/best_AP_epoch_190.pth",
    # pretrained setting is used for initializing backbone
    # load_from setting is used for total model
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
        type='TopdownHeatmapConvHead',
        in_channels=48,
        out_channels=channel_cfg['num_output_channels'],
        num_conv_layers=3,
        num_conv_filters=(64, 128, 256),
        num_conv_kernels=(3, 3, 3),
        num_deconv_layers=3,
        num_deconv_filters=(256, 128, 64),
        num_deconv_kernels=(3, 3, 3),

        final_conv_kernel=1,
        # extra=dict(final_conv_kernel=1, num_conv_layers=3, ),
        loss_keypoint=dict(type='JointsMSELoss', use_target_weight=True)),

    feature_head=dict(
        type='TopdownFeatureHead',
        in_channels=48,
        out_channels=channel_cfg['num_output_channels'],
        num_conv_layers=3,
        num_conv_filters=(64, 128, 256),
        num_conv_kernels=(3, 3, 3),
        num_deconv_layers=3,
        num_deconv_filters=(256, 128, 64),
        num_deconv_kernels=(3, 3, 3),
        loss_feature=dict(type='SupConLoss',
                          temperature=0.07,
                          contrast_mode='all',
                          base_temperature=0.01)),
    keypoint_loss=True,
    feature_loss=True,
    train_cfg=dict(),
    test_cfg=dict(
        flip_test=False,
        post_process='default',
        shift_heatmap=True,
        modulate_kernel=11)
)

"""data config"""
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
    space_size=[0, 0, 0],
    space_center=[0, 0, 0],
    cube_size=[0, 0, 0],
    num_cameras=6,
    use_different_joint_weights=False
)

# train pipeline setting for model with triangulate_head
train_pipeline = [
    dict(
        type='MultiItemProcess',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='SquareBbox'),
            dict(type='CropImage',
                 update_camera=True),
            dict(type='ResizeImage',
                 update_camera=False),
            dict(type='ToTensor'),
            dict(
                type='NormalizeTensor',
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
            dict(type='TopDownGenerateTarget', sigma=2)
        ]),
    dict(
        type='DiscardDuplicatedItems',
        keys_list=[
            'dataset', 'ann_info', 'roots_3d', 'joints_4d', 'joints_4d_visible', 'flip_pairs'
        ]),
    dict(
        type='GroupCams',
        keys=['img', 'target', 'target_weight']
    ),
    dict(
        type='Collect',
        keys=['img', 'target', 'target_weight'],
        meta_keys=[
            'image_file', 'joints_3d', 'joints_3d_visible', 'bbox',
            # 'center', 'scale', 'rotation', 'bbox_score', 'flip_pairs',
            'joints_4d', 'joints_4d_visible', 'camera',
        ]),
]

# evaluate pipeline setting for model with triangulate_head
eval_pipeline = [
    dict(
        type='MultiItemProcess',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='SquareBbox'),
            dict(type='CropImage',
                 update_camera=True),
            dict(type='ResizeImage',
                 update_camera=False),
            dict(type='ComputeProjMatric'),
            dict(type='ToTensor'),
            dict(
                type='NormalizeTensor',
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
            dict(type='TopDownGenerateTarget', sigma=2),
        ]),
    dict(
        type='DiscardDuplicatedItems',
        keys_list=[
            'dataset', 'ann_info', 'roots_3d', 'joints_world', 'joints_world_visible'
        ]),
    dict(
        type='GroupCams',
        keys=['img', 'proj_metric']
    ),
    # dict(
    #     type='ComputeProjMatrics'
    # ),

    dict(
        type='Collect',
        keys=['img', 'proj_metric'],
        meta_keys=[
            'image_file', 'bbox'
            # 'center', 'scale', 'rotation', 'bbox_score', 'flip_pairs',
            # 'camera',# 'proj_metrices'
        ]),
]

test_pipeline = eval_pipeline

data_root = 'D:/Datasets/transfer_mouse/dannce_20230130'
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=2,
    val_dataloader=dict(samples_per_gpu=4),
    test_dataloader=dict(samples_per_gpu=4),
    train=dict(
        type='MouseDannce3dDataset',
        ann_file=f'{data_root}/annotations_visible_train930_new.json',
        ann_3d_file=f'{data_root}/joints_3d.json',
        cam_file=f'{data_root}/cams.pkl',
        img_prefix=f'{data_root}/images_gray/',
        data_cfg=data_cfg,
        pipeline=train_pipeline,
        dataset_info={{_base_.dataset_info}}),

    eval=dict(
        type='MouseDannce2dDatasetSview',
        ann_file=f'{data_root}/annotations_visible_eval930_new.json',
        ann_3d_file=f'{data_root}/joints_3d.json',
        cam_file=f'{data_root}/cams.pkl',
        img_prefix=f'{data_root}/images_gray/',
        data_cfg=data_cfg,
        pipeline=eval_pipeline,
        dataset_info={{_base_.dataset_info}}),

    test=dict(
        type='MouseDannce2dDatasetSview',
        ann_file=f'{data_root}/annotations_visible_eval930_new.json',
        ann_3d_file=f'{data_root}/joints_3d.json',
        cam_file=f'{data_root}/cams.pkl',
        img_prefix=f'{data_root}/images_gray/',
        data_cfg=data_cfg,
        pipeline=eval_pipeline,
        dataset_info={{_base_.dataset_info}}),
)
