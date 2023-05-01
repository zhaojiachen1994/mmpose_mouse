_base_ = [
    '../../_base_/default_runtime.py',
    '../../_base_/human_datasets/human_p17.py',
    'data_h36m_p17.py',
    'model_hrnet_score_p17.py',
    'pipeline_target.py'
]

data_root = "D:/Datasets/h36m_dataset/human3.6m_parse"

subject = 9
data = dict(
    samples_per_gpu=6,
    workers_per_gpu=2,
    val_dataloader=dict(samples_per_gpu=7, workers_per_gpu=7),
    test_dataloader=dict(samples_per_gpu=7, workers_per_gpu=7),
    test=dict(
        type="Body3DH36MMviewDataset",
        ann_file=f"{data_root}/annotations/Human36M_subject{subject}_data_.json",
        ann_3d_file=f"{data_root}/annotations/Human36M_subject{subject}_joint_3d.json",
        cam_file=f"{data_root}/annotations/Human36M_subject{subject}_camera.json",
        img_prefix=f"{data_root}/images/",
        data_cfg={{_base_.data_cfg}},
        pipeline={{_base_.target_pipeline}},
        dataset_info={{_base_.dataset_info}}),
)
