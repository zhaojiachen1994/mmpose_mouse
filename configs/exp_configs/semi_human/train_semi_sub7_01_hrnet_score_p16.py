_base_ = [
    '../../_base_/default_runtime.py',
    '../../_base_/human_datasets/human_p16.py',
    'model_cd_hrnet_score_p16.py',
    'semi_h36m_base.py'
]

total_epochs = 210

evaluation = dict(interval=1, metric='mpjpe', by_epoch=True, save_best='MPJPE')
optimizer = dict(type='Adam', lr=1e-5, )
optimizer_config = dict(grad_clip=None)

lr_config = dict(policy='step', warmup='linear', warmup_iters=50, warmup_ratio=0.001, step=[17, 20])
log_config = dict(interval=1, hooks=[dict(type='TextLoggerHook'), ])

subject = 7
ratio = "01"
h36m_root = "D:/Datasets/h36m_dataset/human3.6m_parse"
load_from = "D:/Pycharm Projects-win/mm_mouse/mmpose/work_dirs/hrnet_w48_h36m_01_p16_256x256/best_PCK_epoch_120.pth"

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    val_dataloader=dict(samples_per_gpu=4),
    test_dataloader=dict(samples_per_gpu=4),
    train=dict(
        type='CrossDomain2d3dDataset',
        source_data=dict(
            type='TopDownH36MDataset',
            ann_file=f'{h36m_root}/annotations/Human36M_subjects_156_{ratio}.json',
            img_prefix=f'{h36m_root}/images/',
            data_cfg={{_base_.source_data_cfg}},
            pipeline={{_base_.source_pipeline}},
            dataset_info={{_base_.dataset_info}}),

        target_data=dict(
            type="Body3DH36MMviewDataset",
            ann_file=f"{h36m_root}/annotations/Human36M_subject{subject}_data_01.json",
            ann_3d_file=f"{h36m_root}/annotations/Human36M_subject{subject}_joint_3d.json",
            cam_file=f"{h36m_root}/annotations/Human36M_subject{subject}_camera.json",
            img_prefix=f"{h36m_root}/images/",
            data_cfg={{_base_.target_data_cfg}},
            pipeline={{_base_.target_pipeline}},
            dataset_info={{_base_.dataset_info}}),
    ),
    val=dict(
        type="CrossDomain2d3dDataset",
        source_data=None,
        target_data=dict(
            type="Body3DH36MMviewDataset",
            ann_file=f"{h36m_root}/annotations/Human36M_subject{subject}_data_01.json",
            ann_3d_file=f"{h36m_root}/annotations/Human36M_subject{subject}_joint_3d.json",
            cam_file=f"{h36m_root}/annotations/Human36M_subject{subject}_camera.json",
            img_prefix=f"{h36m_root}/images/",
            data_cfg={{_base_.target_data_cfg}},
            pipeline={{_base_.target_pipeline}},
            dataset_info={{_base_.dataset_info}}),
    ),
    test=dict(
        type="CrossDomain2d3dDataset",
        source_data=None,
        target_data=dict(
            type="Body3DH36MMviewDataset",
            ann_file=f"{h36m_root}/annotations/Human36M_subject{subject}_data_01.json",
            ann_3d_file=f"{h36m_root}/annotations/Human36M_subject{subject}_joint_3d.json",
            cam_file=f"{h36m_root}/annotations/Human36M_subject{subject}_camera.json",
            img_prefix=f"{h36m_root}/images/",
            data_cfg={{_base_.target_data_cfg}},
            pipeline={{_base_.target_pipeline}},
            dataset_info={{_base_.dataset_info}}),
    ),
)
