_base_ = ['../_base_/default_runtime.py',
          '../_base_/mouse_datasets/mouse_dannce_3d.py']
# source_data_info = ['../_base_/mouse_datasets/mouse_dannce_3d.py']
# target_data_info = ['../_base_/mouse_datasets/mouse_one_1229.py']


# joint channel config
source_channel_cfg = dict(
    num_output_channels=22,
    dataset_joints=22,
    dataset_channel=[
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
    ],
    inference_channel=[
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21
    ])

target_channel_cfg = dict(
    num_output_channels=16,
    dataset_joints=16,
    dataset_channel=[
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    ],
    inference_channel=[
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])

"""model config"""

train_cfg = dict(
    supervised_2d=True,  # use the 2d ground truth to train keypoint_head
    contrastive_feature=True,  # use the sup_con_loss to train feature_head
    supervised_3d=True,  # use the 3d ground truth to train triangulate_head
    unSupervised_3d=False,  # use the triangulation residual loss to train triangulate_head
),

model = dict(
    type='TriangNet',
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
        out_channels=source_channel_cfg['num_output_channels'],
        num_deconv_layers=0,
        extra=dict(final_conv_kernel=1, ),
        loss_keypoint=dict(type='JointsMSELoss', use_target_weight=True)),

    triangulate_head=dict(
        type='TriangulateHead',
        num_cams=6,
        img_shape=[256, 256],
        heatmap_shape=[64, 64],
        softmax_heatmap=True,
        loss_3d_super=dict(type='MSELoss',
                           use_target_weight=True,
                           loss_weight=1.),
        train_cfg=train_cfg,
    ),
    test_cfg=dict(
        flip_test=False,
        post_process='default',
        shift_heatmap=True,
        modulate_kernel=11)
)

"""data config"""
source_data_cfg = dict(
    image_size=[256, 256],
    heatmap_size=[64, 64],
    num_output_channels=source_channel_cfg['num_output_channels'],
    num_joints=source_channel_cfg['dataset_joints'],
    dataset_channel=source_channel_cfg['dataset_channel'],
    inference_channel=source_channel_cfg['inference_channel'],
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

target_data_cfg = dict(
    image_size=[256, 256],
    heatmap_size=[64, 64],
    num_output_channels=target_channel_cfg['num_output_channels'],
    num_joints=target_channel_cfg['dataset_joints'],
    dataset_channel=target_channel_cfg['dataset_channel'],
    inference_channel=target_channel_cfg['inference_channel'],
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
        keys=['img', 'target', 'target_weight', 'joints_4d', 'proj_mat', 'joints_3d'],
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
        meta_keys=['image_file', 'bbox_offset', 'resize_ratio']
    )
]
test_pipeline = val_pipeline

source_data_root = 'D:/Datasets/transfer_mouse/dannce_20230130'
target_data_root = 'D:/Datasets/transfer_mouse/onemouse1229'

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=2,
    val_dataloader=dict(samples_per_gpu=4),
    test_dataloader=dict(samples_per_gpu=4),
    source=dict(
        type='MouseDannce3dDataset',
        ann_file=f'{source_data_root}/annotations_visible_train930_new.json',
        ann_3d_file=f'{source_data_root}/joints_3d.json',
        cam_file=f'{source_data_root}/cams.pkl',
        img_prefix=f'{source_data_root}/images_gray/',
        data_cfg=source_data_cfg,
        pipeline=train_pipeline,
        dataset_info={{_base_.dataset_info}}),

    target=dict(
        type="Mouse12293dDatasetMview",
        ann_file=f"{target_data_root}/anno_20221229-1-012345.json",
        ann_3d_file=f"{target_data_root}/anno_20221229_joints_3d.json",
        cam_file=f"{target_data_root}/calibration_adjusted.json",
        img_prefix=f'{target_data_root}/',
        data_cfg=target_data_cfg,
        pipeline=val_pipeline,
        dataset_info={{_base_.dataset_info}})
)
