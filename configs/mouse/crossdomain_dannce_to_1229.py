_base_ = [
    '../_base_/default_runtime.py',
    '../_base_/mouse_datasets/mouse_one_1229_p12_s.py',
    '../_base_/mouse_datasets/mouse_dannce_p12_s.py'
]

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
        keys=['img', 'target', 'target_weight', 'joints_4d', 'proj_mat', 'joints_3d'],
        meta_keys=['image_file', 'bbox_offset', 'resize_ratio']
    )
]

dannce_data_root = 'D:/Datasets/transfer_mouse/dannce_20230130'
mouse1229_data_root = 'D:/Datasets/transfer_mouse/onemouse1229'

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=2,
    val_dataloader=dict(samples_per_gpu=4),
    test_dataloader=dict(samples_per_gpu=4),
    train=dict(
        type='CrossDomain2d3dDataset',
        source_data=dict(
            type='MouseDannce2dDatasetSview',
            ann_file=f'{dannce_data_root}/annotations_visible_train930_new.json',
            # ann_3d_file=f'{_base_.daccne_data_root}/joints_3d.json',
            # cam_file=f'{_base_.daccne_data_root}/cams.pkl',
            img_prefix=f'{dannce_data_root}/images_gray/',
            data_cfg={{_base_.dannce_data_cfg}},
            pipeline=source_pipeline,
            dataset_info={{_base_.dannce_dataset_info}}),
        target_data=dict(
            type='Mouse12293dDatasetMview',
            ann_file=f"{mouse1229_data_root}/anno_20221229-1-012345.json",
            ann_3d_file=f"{mouse1229_data_root}/anno_20221229_joints_3d.json",
            cam_file=f"{mouse1229_data_root}/calibration_adjusted.json",
            img_prefix=f'{mouse1229_data_root}/',
            data_cfg={{_base_.mouse1229_data_cfg}},
            pipeline=target_pipeline,
            dataset_info={{_base_.mouse1229_dataset_info}})

    ),
)
