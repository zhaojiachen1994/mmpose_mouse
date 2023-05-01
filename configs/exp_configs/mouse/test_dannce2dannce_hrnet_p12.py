_base_ = [
    '../../_base_/default_runtime.py',
    '../../_base_/mouse_datasets/mouse_p12.py',
    'data_dannce_p12.py',
    'model_hrnet_p12.py',
    'pipeline_target.py'
]

dannce_root = 'D:/Datasets/transfer_mouse/dannce_20230130'
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=2,
    val_dataloader=dict(samples_per_gpu=5),
    test_dataloader=dict(samples_per_gpu=5),
    test=dict(
        type='MouseDannce3dDataset',
        ann_file=f"{dannce_root}/annotations_visible_new2.json",
        ann_3d_file=f"{dannce_root}/joints_3d.json",
        cam_file=f"{dannce_root}/cams.pkl",
        img_prefix=f'{dannce_root}/images_gray/',
        data_cfg={{_base_.dannce_data_cfg}},
        pipeline={{_base_.target_pipeline}},
        dataset_info={{_base_.dataset_info}})
)
