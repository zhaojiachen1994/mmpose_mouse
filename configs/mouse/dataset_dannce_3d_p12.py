_base_ = ['../_base_/default_runtime.py',
          '../_base_/mouse_datasets/mouse_dannce_p12.py']

channel_cfg = dict(
    num_output_channels=12,
    dataset_joints=12,
    dataset_channel=[
        0, 1, 2, 3, 4, 5, 6, 7, 9, 13, 16, 19
    ],
    inference_channel=[
        0, 1, 2, 3, 4, 5, 6, 7, 9, 13, 16, 19
    ])

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
eval_pipeline = train_pipeline
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
