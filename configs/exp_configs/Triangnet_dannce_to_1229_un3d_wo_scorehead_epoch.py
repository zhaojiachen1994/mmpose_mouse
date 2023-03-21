_base_ = ['../_base_/default_runtime.py',
          '../_base_/mouse_datasets/mouse_one_1229_p12_s.py'
          ]

total_epochs = 50

evaluation = dict(interval=5, metric='mpjpe', by_epoch=True, save_best='MPJPE')
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

# model settings
load_from = "D:/Pycharm Projects-win/mm_mouse/mmpose/work_dirs/" \
            "hrnet_w48_dannce_2d_p12_256x256/best_AP_epoch_100.pth"

num_joints = 12
model = dict(
    type='TriangNet',
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

    # score_head=dict(
    #     type='GlobalAveragePoolingHead',
    #     in_channels=48,
    #     n_classes=num_joints
    # ),

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
    train_cfg=dict(
        use_2d_sup=False,  # use the 2d ground truth to train keypoint_head
        use_3d_sup=False,  # use the 3d ground truth to train triangulate_head
        use_3d_unsup=True,  # use the triangulation residual loss to train triangulate_head
    ),
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

data_root = "D:/Datasets/transfer_mouse/onemouse1229"
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=2,
    val_dataloader=dict(samples_per_gpu=5),
    test_dataloader=dict(samples_per_gpu=5),
    train=dict(
        type="Mouse12293dDatasetMview",
        ann_file=f"{data_root}/anno_20221229-1-012345.json",
        ann_3d_file=f"{data_root}/anno_20221229_joints_3d.json",
        cam_file=f"{data_root}/calibration_adjusted.json",
        img_prefix=f'{data_root}/',
        data_cfg={{_base_.mouse1229_data_cfg}},
        pipeline=train_pipeline,
        dataset_info={{_base_.mouse1229_dataset_info}}),
    val=dict(
        type="Mouse12293dDatasetMview",
        ann_file=f"{data_root}/anno_20221229-1-012345.json",
        ann_3d_file=f"{data_root}/anno_20221229_joints_3d.json",
        cam_file=f"{data_root}/calibration_adjusted.json",
        img_prefix=f'{data_root}/',
        data_cfg={{_base_.mouse1229_data_cfg}},
        pipeline=val_pipeline,
        dataset_info={{_base_.mouse1229_dataset_info}}),
    test=dict(
        type="Mouse12293dDatasetMview",
        ann_file=f"{data_root}/anno_20221229-1-012345.json",
        ann_3d_file=f"{data_root}/anno_20221229_joints_3d.json",
        cam_file=f"{data_root}/calibration_adjusted.json",
        img_prefix=f'{data_root}/',
        data_cfg={{_base_.mouse1229_data_cfg}},
        pipeline=val_pipeline,
        dataset_info={{_base_.mouse1229_dataset_info}})
)
