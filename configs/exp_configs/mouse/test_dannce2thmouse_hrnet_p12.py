_base_ = [
    '../../_base_/default_runtime.py',
    '../../_base_/mouse_datasets/mouse_p12.py',
    'data_thmouse_3d_p12.py',
    'model_hrnet_p12.py',
    'pipeline_target.py'
]

thmouse_root = "D:/Datasets/transfer_mouse/onemouse1229"
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=2,
    val_dataloader=dict(samples_per_gpu=5),
    test_dataloader=dict(samples_per_gpu=5),
    test=dict(
        type="Mouse12293dDatasetMview",
        ann_file=f"{thmouse_root}/anno_20221229-1-012345.json",
        ann_3d_file=f"{thmouse_root}/anno_20221229_joints_3d.json",
        cam_file=f"{thmouse_root}/calibration_adjusted.json",
        img_prefix=f'{thmouse_root}/',
        data_cfg={{_base_.thmouse_data_cfg}},
        pipeline={{_base_.target_pipeline}},
        dataset_info={{_base_.dataset_info}})
)