_base_ = ['../_base_/default_runtime.py',
          '../_base_/mouse_datasets/mouse_dannce_p5_s.py'
          ]

load_from = "D:/Pycharm Projects-win/mm_mouse/mmpose/work_dirs/" \
            "hrnet_w48_concat_mars_p5_256x256/best_AP_epoch_5.pth"

num_joints = 5

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
        det_conf_thr=0.0,
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

data_root = "D:/Datasets/transfer_mouse/dannce_20230130"
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=2,
    val_dataloader=dict(samples_per_gpu=5),
    test_dataloader=dict(samples_per_gpu=5),
    # train=dict(
    #     type="Mouse12293dDatasetMview",
    #     ann_file=f"{data_root}/anno_20221229-1-012345.json",
    #     ann_3d_file=f"{data_root}/anno_20221229_joints_3d.json",
    #     cam_file=f"{data_root}/calibration_adjusted.json",
    #     img_prefix=f'{data_root}/',
    #     data_cfg={{_base_.mouse1229_data_cfg}},
    #     pipeline=train_pipeline,
    #     dataset_info={{_base_.mouse1229_dataset_info}}),
    # val=dict(
    #     type="Mouse12293dDatasetMview",
    #     ann_file=f"{data_root}/anno_20221229-1-012345.json",
    #     ann_3d_file=f"{data_root}/anno_20221229_joints_3d.json",
    #     cam_file=f"{data_root}/calibration_adjusted.json",
    #     img_prefix=f'{data_root}/',
    #     data_cfg={{_base_.mouse1229_data_cfg}},
    #     pipeline=val_pipeline,
    #     dataset_info={{_base_.mouse1229_dataset_info}}),
    test=dict(
        type='MouseDannce3dDataset',
        ann_file=f'{data_root}/annotations_visible_eval930_new2.json',
        ann_3d_file=f'{data_root}/joints_3d.json',
        cam_file=f'{data_root}/cams.pkl',
        img_prefix=f'{data_root}/images_gray/',
        data_cfg={{_base_.dannce_data_cfg}},
        pipeline=test_pipeline,
        dataset_info={{_base_.dannce_dataset_info}}),
)
