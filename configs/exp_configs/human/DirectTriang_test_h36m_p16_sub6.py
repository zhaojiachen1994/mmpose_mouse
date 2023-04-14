_base_ = [
    '../../_base_/default_runtime.py',
    '../../_base_/human_datasets/human_p16.py',
    'DirectTriang_test_h36m_p16_base.py'
]
subject = 6

data_root = "D:/Datasets/h36m_dataset/human3.6m_parse"

data=dict(
    samples_per_gpu=6,
    workers_per_gpu=2,
    val_dataloader=dict(samples_per_gpu=7, workers_per_gpu=7),
    test_dataloader=dict(samples_per_gpu=7, workers_per_gpu=7),
    test=dict(
        type="Body3DH36MMviewDataset",
        ann_file=f"{data_root}/annotations/Human36M_subject{subject}_data_reorder_ds.json",
        ann_3d_file=f"{data_root}/annotations/Human36M_subject{subject}_joint_3d.json",
        cam_file=f"{data_root}/annotations/Human36M_subject{subject}_camera.json",
        img_prefix=f"{data_root}/images/",
        data_cfg={{_base_.data_cfg}},
        pipeline={{_base_.test_pipeline}},
        dataset_info={{_base_.dataset_info}}),
)





















