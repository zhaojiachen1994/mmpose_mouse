_base_ = [
    '../_base_/default_runtime.py',
    '../_base_/mouse_datasets/mouse_p9.py'
]


total_epochs = 50

evaluation = dict(interval=1, metric='mpjpe', by_epoch=True, save_best='MPJPE')
optimizer = dict(
    type='Adam',
    lr=1e-4,
)
optimizer_config = dict(grad_clip=None)

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

num_joints = 9

# load_from = "D:/Pycharm Projects-win/mm_mouse/mmpose/work_dirs/hrnet_w48_mars_p9_256x256/best_AP_epoch_95.pth"
load_from = "D:/Pycharm Projects-win/mm_mouse/mmpose/experiments/work_dirs/Triangnet_mars_to_dannce_p9_un3d_epoch/best_MPJPE_epoch_40.pth"

model = dict(
    type='CDTriangNet',
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
        det_conf_thr=0.0,
    ),
    train_cfg=dict(
        source_2d_sup_loss=True,  # use the 2d ground truth to train keypoint_head
        target_3d_unsup_loss=True,  # use the triangulation residual loss to train triangulate_head
    ),
    test_cfg=dict(
        flip_test=False,
        post_process='default',
        shift_heatmap=True,
        modulate_kernel=11)
)

source_pipeline = [
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
            'image_file', 'joints_3d', 'joints_3d_visible', 'center', 'scale',
            'rotation', 'bbox_score', 'flip_pairs'
        ]),
]

target_pipeline = [
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
        keys=['img', 'target', 'target_weight',  'proj_mat', 'joints_3d'],
        meta_keys=['image_file', 'bbox_offset', 'joints_4d', 'joints_4d_visible', 'resize_ratio']
    )
]

mars_data_root = "D:/Datasets/MARS-PoseAnnotationData"
mars_channel_cfg = dict(
    num_output_channels=9,
    dataset_joints=9,
    dataset_channel=[0, 1, 2, 3, 6, 7, 8, 9, 10],
    inference_channel=[0, 1, 2, 3, 6, 7, 8, 9, 10])
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
    space_size=[0.3, 0.3, 0.3],
    space_center=[0, 0, 0.15],
    cube_size=[0.1, 0.1, 0.1],
    num_cameras=6,
    use_different_joint_weights=False
)

dannce_data_root = 'D:/Datasets/transfer_mouse/dannce_20230130'
dannce_channel_cfg = dict(
    num_output_channels=9,
    dataset_joints=9,
    dataset_channel=[2, 0, 1, 3, 5, 8, 12, 16, 19],
    inference_channel=[2, 0, 1, 3, 5, 8, 12, 16, 19])
dannce_data_cfg = dict(
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
data_cfg = dannce_data_cfg


data = dict(
    samples_per_gpu=4,
    workers_per_gpu=2,
    val_dataloader=dict(samples_per_gpu=4),
    test_dataloader=dict(samples_per_gpu=4),

    train=dict(
        type='CrossDomain2d3dDataset',
        source_data=dict(
            type='MouseMars2dDataset',
            ann_file=f'{mars_data_root}/MARS_keypoints_front_black_hq.json',
            img_prefix=f'{mars_data_root}/raw_images_front/',
            data_cfg=mars_data_cfg,
            pipeline=source_pipeline,
            dataset_info={{_base_.dataset_info}}),
        target_data=dict(
            type='MouseDannce3dDataset',
            ann_file=f"{dannce_data_root}/annotations_visible_new2.json",
            ann_3d_file=f"{dannce_data_root}/joints_3d.json",
            cam_file=f"{dannce_data_root}/cams.pkl",
            img_prefix=f'{dannce_data_root}/images_gray/',
            data_cfg=dannce_data_cfg,
            pipeline=target_pipeline,
            dataset_info={{_base_.dataset_info}})
    ),

    val=dict(
        type='CrossDomain2d3dDataset',
        source_data=None,
        target_data=dict(
            type='MouseDannce3dDataset',
            ann_file=f"{dannce_data_root}/annotations_visible_new2.json",
            ann_3d_file=f"{dannce_data_root}/joints_3d.json",
            cam_file=f"{dannce_data_root}/cams.pkl",
            img_prefix=f'{dannce_data_root}/images_gray/',
            data_cfg=dannce_data_cfg,
            pipeline=target_pipeline,
            dataset_info={{_base_.dataset_info}})
    ),
    test=dict(
        type='CrossDomain2d3dDataset',
        source_data=None,
        target_data=dict(
            type='MouseDannce3dDataset',
            ann_file=f"{dannce_data_root}/annotations_visible_new2.json",
            ann_3d_file=f"{dannce_data_root}/joints_3d.json",
            cam_file=f"{dannce_data_root}/cams.pkl",
            img_prefix=f'{dannce_data_root}/images_gray/',
            data_cfg=dannce_data_cfg,
            pipeline=target_pipeline,
            dataset_info={{_base_.dataset_info}})
    )
)


