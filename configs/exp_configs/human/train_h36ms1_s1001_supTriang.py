_base_ = ['../_base_/default_runtime.py',
          '../../_base_/human_datasets/human_p17.py',
          'data_h36m_p17.py',
          'model_hrnet_score_p17_sup3d.py',
          ]

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

subject = 1
h36m_root = "D:/Datasets/h36m_dataset/human3.6m_parse"
load_from = "D:/Pycharm Projects-win/mm_mouse/mmpose/work_dirs/hrnet_w48_h36m_s01_01_p17_256x256/best_PCK_epoch_120.pth"

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=2,
    val_dataloader=dict(samples_per_gpu=4),
    test_dataloader=dict(samples_per_gpu=4),
    train=dict(
        type="Body3DH36MMviewDataset",
        ann_file=f"{h36m_root}/annotations/Human36M_subject{subject}_data_001.json",
        ann_3d_file=f"{h36m_root}/annotations/Human36M_subject{subject}_joint_3d.json",
        cam_file=f"{h36m_root}/annotations/Human36M_subject{subject}_camera.json",
        img_prefix=f"{h36m_root}/images/",
        data_cfg={{_base_.data_cfg}},
        pipeline=train_pipeline,
        dataset_info={{_base_.dataset_info}}),

    test=dict(
        type="Body3DH36MMviewDataset",
        ann_file=f"{h36m_root}/annotations/Human36M_subject{subject}_data_01.json",
        ann_3d_file=f"{h36m_root}/annotations/Human36M_subject{subject}_joint_3d.json",
        cam_file=f"{h36m_root}/annotations/Human36M_subject{subject}_camera.json",
        img_prefix=f"{h36m_root}/images/",
        data_cfg={{_base_.data_cfg}},
        pipeline=test_pipeline,
        dataset_info={{_base_.dataset_info}}),
)
