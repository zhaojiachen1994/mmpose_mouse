_base_ = [
    '../_base_/default_runtime.py',
    '../_base_/mouse_datasets/mouse_p9.py'
]

total_epochs = 50

evaluation = dict(interval=1, metric='mpjpe', by_epoch=True, save_best='MPJPE')

optimizer = dict(
    backbone=dict(type='Adam', lr=5e-5, betas=(0.0, 0.99)),
    keypoint_head=dict(type='Adam', lr=5e-5, betas=(0.0, 0.99)),
    score_head=dict(type='Adam', lr=5e-5, betas=(0.0, 0.99))
)
optimizer_config = None
lr_config = None

log_config = dict(
    interval=1,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])

num_joints = 9

# load_from = "D:/Pycharm Projects-win/mm_mouse/mmpose/work_dirs/hrnet_w48_mars_p9_256x256/best_AP_epoch_95.pth"
load_from = "D:/Pycharm Projects-win/mm_mouse/mmpose/experiments/work_dirs/" \
            "Triangnet_mars_to_dannce_p9_un3d_epoch/epoch_10.pth"

model = dict(
    type='AdvTriangNet',
    pretrained=None,
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
        out_channels=num_joints,
        num_deconv_layers=0,
        extra=dict(final_conv_kernel=1, ),
        loss_keypoint=dict(type='JointsMSELoss', use_target_weight=True)),

    score_head=dict(
        type='GlobalAveragePoolingHead',
        in_channels=48,
        n_classes=num_joints
    ),
    triangulate_head=dict(
        type='TriangulateHead',
        num_cams=6,
        img_shape=[256, 256],
        heatmap_shape=[64, 64],
        softmax_heatmap=True,
        loss_3d_sup=dict(type='MSELoss',
                         use_target_weight=False,
                         loss_weight=1.),
        det_conf_thr=0.5,
    ),
    train_cfg=None,
    test_cfg=dict(
        flip_test=False,
        post_process='default',
        shift_heatmap=True,
        modulate_kernel=11)
)


train_pipeline = [
    dict(
        type="MultiItemProcess",
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='SquareBbox'),
            dict(type='CropImage',
                 update_camera=True),
            dict(type='ResizeImage',
                 update_camera=True),
            dict(type='ComputeProjMatric'),
            dict(type='ToTensor'),
            dict(
                type='NormalizeTensor',
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
            dict(type='TopDownGenerateTarget', sigma=2)
        ]),
    dict(
        type='DiscardDuplicatedItems',
        keys_list=['dataset', 'ann_info', 'joints_4d', 'joints_4d_visible']
    ),
    dict(
        type='GroupCams',
        keys=['img', 'target', 'target_weight', 'proj_mat', 'joints_3d', 'bbox']
    ),
    dict(
        type="Collect",
        keys=['img', 'target', 'target_weight', 'joints_4d', 'joints_4d_visible', 'proj_mat', 'joints_3d'],
        meta_keys=['image_file', 'bbox_offset', 'resize_ratio']
    )
]

val_pipeline = [
    dict(
        type="MultiItemProcess",
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='SquareBbox'),
            dict(type='CropImage',
                 update_camera=True),
            dict(type='ResizeImage',
                 update_camera=True),
            dict(type='ComputeProjMatric'),
            dict(type='ToTensor'),
            dict(
                type='NormalizeTensor',
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
        ]),
    dict(
        type='DiscardDuplicatedItems',
        keys_list=['dataset', 'ann_info', 'joints_4d', 'joints_4d_visible']
    ),
    dict(
        type='GroupCams',
        keys=['img', 'proj_mat']
    ),
    dict(
        type="Collect",
        keys=['img', 'proj_mat'],
        meta_keys=['image_file', 'bbox_offset', 'resize_ratio',
                   'joints_4d', 'joints_4d_visible']
    )
]
test_pipeline = val_pipeline

dannce_channel_cfg = dict(
    num_output_channels=9,
    dataset_joints=9,
    dataset_channel=[2, 0, 1, 3, 5, 8, 9, 10, 11],
    inference_channel=[2, 0, 1, 3, 5, 8, 9, 10, 11])

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

data_root = "D:/Datasets/transfer_mouse/dannce_20230130"
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=2,
    val_dataloader=dict(samples_per_gpu=4),
    test_dataloader=dict(samples_per_gpu=4),
    train=dict(
        type='MouseDannce3dDataset',
        ann_file=f'{data_root}/annotations_visible_new2.json',
        ann_3d_file=f'{data_root}/joints_3d.json',
        cam_file=f'{data_root}/cams.pkl',
        img_prefix=f'{data_root}/images_gray/',
        data_cfg=data_cfg,
        pipeline=train_pipeline,
        dataset_info={{_base_.dataset_info}}),

    val=dict(
        type='MouseDannce3dDataset',
        ann_file=f'{data_root}/annotations_visible_new2.json',
        ann_3d_file=f'{data_root}/joints_3d.json',
        cam_file=f'{data_root}/cams.pkl',
        img_prefix=f'{data_root}/images_gray/',
        data_cfg=data_cfg,
        pipeline=val_pipeline,
        dataset_info={{_base_.dataset_info}}),

    test=dict(
        type='MouseDannce3dDataset',
        ann_file=f'{data_root}/annotations_visible_new2.json',
        ann_3d_file=f'{data_root}/joints_3d.json',
        cam_file=f'{data_root}/cams.pkl',
        img_prefix=f'{data_root}/images_gray/',
        data_cfg=data_cfg,
        pipeline=test_pipeline,
        dataset_info={{_base_.dataset_info}})
)
















