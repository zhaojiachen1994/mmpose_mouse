_base_ = [
    '../../_base_/default_runtime.py',
    '../../_base_/mouse_datasets/mouse_p9.py',
    'data_calms_p9.py',
    'data_thmouse_p9.py',
    'pipeline_source.py',
    'pipeline_target.py',
    'model_cd_hrnet_score_dd_p9.py',
]

total_epochs = 120

evaluation = dict(interval=5, metric='mpjpe', by_epoch=True, save_best='MPJPE')
optimizer = dict(type='Adam', lr=1e-5, )
optimizer_config = dict(grad_clip=None)

lr_config = dict(policy='step', warmup='linear', warmup_iters=50, warmup_ratio=0.001, step=[17, 20])

log_config = dict(interval=1, hooks=[dict(type='TextLoggerHook'), ])

load_from = "D:/Pycharm Projects-win/mm_mouse/mmpose/work_dirs/hrnet_w48_mars_p9_256x256/best_AP_epoch_95.pth"

thmouse_root = "D:/Datasets/transfer_mouse/onemouse1229"
calms_root = "D:/Datasets/MARS-PoseAnnotationData"

data = dict(
    samples_per_gpu=3,
    workers_per_gpu=3,
    val_dataloader=dict(samples_per_gpu=4),
    test_dataloader=dict(samples_per_gpu=4),

    train=dict(
        type='CrossDomain2d3dDataset',
        source_data=dict(
            type='MouseMars2dDataset',
            ann_file=f'{calms_root}/MARS_keypoints_front_black_hq.json',
            img_prefix=f'{calms_root}/raw_images_front/',
            data_cfg={{_base_.calms_data_cfg}},
            pipeline={{_base_.source_pipeline}},
            dataset_info={{_base_.dataset_info}}),
        target_data=dict(
            type='Mouse12293dDatasetMview',
            ann_file=f"{thmouse_root}/anno_20221229-1-012345.json",
            ann_3d_file=f"{thmouse_root}/anno_20221229_joints_3d.json",
            cam_file=f"{thmouse_root}/calibration_adjusted.json",
            img_prefix=f'{thmouse_root}/',
            data_cfg={{_base_.thmouse_data_cfg}},
            pipeline={{_base_.target_pipeline}},
            dataset_info={{_base_.dataset_info}}),
    ),

    val=dict(
        type='CrossDomain2d3dDataset',
        source_data=None,
        target_data=dict(
            type='Mouse12293dDatasetMview',
            ann_file=f"{thmouse_root}/anno_20221229-1-012345.json",
            ann_3d_file=f"{thmouse_root}/anno_20221229_joints_3d.json",
            cam_file=f"{thmouse_root}/calibration_adjusted.json",
            img_prefix=f'{thmouse_root}/',
            data_cfg={{_base_.thmouse_data_cfg}},
            pipeline={{_base_.target_pipeline}},
            dataset_info={{_base_.dataset_info}}),
    ),
    test=dict(
        type='CrossDomain2d3dDataset',
        source_data=None,
        target_data=dict(
            type='Mouse12293dDatasetMview',
            ann_file=f"{thmouse_root}/anno_20221229-1-012345.json",
            ann_3d_file=f"{thmouse_root}/anno_20221229_joints_3d.json",
            cam_file=f"{thmouse_root}/calibration_adjusted.json",
            img_prefix=f'{thmouse_root}/',
            data_cfg={{_base_.thmouse_data_cfg}},
            pipeline={{_base_.target_pipeline}},
            dataset_info={{_base_.dataset_info}}),
    )
)
