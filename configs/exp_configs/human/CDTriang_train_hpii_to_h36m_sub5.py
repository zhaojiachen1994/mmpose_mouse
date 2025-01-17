_base_ = [
    '../../_base_/default_runtime.py',
    '../../_base_/human_datasets/human_p16.py',
    'CDTriang_base.py',
    'hpii_to_h36m_base.py'
]
subject=5
mpii_root = "D:/Datasets/mpii_human_pose_v1"
h36m_root = "D:/Datasets/h36m_dataset/human3.6m_parse"
load_from = "D:\Pycharm Projects-win\mm_mouse\mmpose\work_dirs\hrnet_w48_mpii_256x256\epoch_60.pth"

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    val_dataloader=dict(samples_per_gpu=4),
    test_dataloader=dict(samples_per_gpu=4),
    train=dict(
        type='CrossDomain2d3dDataset',
        source_data=dict(
            type="TopDownMpiiDataset",
            ann_file=f'{mpii_root}/mpii_annotations/mpii_train.json',
            img_prefix=f'{mpii_root}/images/',
            data_cfg={{_base_.mpii_data_cfg}},
            pipeline={{_base_.source_pipeline}},
            dataset_info={{_base_.dataset_info}}),

        target_data=dict(
            type="Body3DH36MMviewDataset",
            ann_file=f"{h36m_root}/annotations/Human36M_subject{subject}_data_reorder_ds.json",
            ann_3d_file=f"{h36m_root}/annotations/Human36M_subject{subject}_joint_3d.json",
            cam_file=f"{h36m_root}/annotations/Human36M_subject{subject}_camera.json",
            img_prefix=f"{h36m_root}/images/",
            data_cfg={{_base_.h36m_data_cfg}},
            pipeline={{_base_.target_pipeline}},
            dataset_info={{_base_.dataset_info}}),
        ),
    val=dict(
        type="CrossDomain2d3dDataset",
        source_data=None,
        target_data=dict(
            type="Body3DH36MMviewDataset",
            ann_file=f"{h36m_root}/annotations/Human36M_subject{subject}_data_reorder_ds.json",
            ann_3d_file=f"{h36m_root}/annotations/Human36M_subject{subject}_joint_3d.json",
            cam_file=f"{h36m_root}/annotations/Human36M_subject{subject}_camera.json",
            img_prefix=f"{h36m_root}/images/",
            data_cfg={{_base_.h36m_data_cfg}},
            pipeline={{_base_.target_pipeline}},
            dataset_info={{_base_.dataset_info}}),
        ),
    test=dict(
        type="CrossDomain2d3dDataset",
        source_data=None,
        target_data=dict(
            type="Body3DH36MMviewDataset",
            ann_file=f"{h36m_root}/annotations/Human36M_subject{subject}_data_reorder_ds.json",
            ann_3d_file=f"{h36m_root}/annotations/Human36M_subject{subject}_joint_3d.json",
            cam_file=f"{h36m_root}/annotations/Human36M_subject{subject}_camera.json",
            img_prefix=f"{h36m_root}/images/",
            data_cfg={{_base_.h36m_data_cfg}},
            pipeline={{_base_.target_pipeline}},
            dataset_info={{_base_.dataset_info}}),
        ),
    )
