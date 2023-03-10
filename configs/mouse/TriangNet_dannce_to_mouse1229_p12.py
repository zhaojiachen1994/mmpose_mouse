_base_ = ['../_base_/default_runtime.py']
# source_data_info = ['../_base_/mouse_datasets/mouse_dannce_3d.py']
# target_data_info = ['../_base_/mouse_datasets/mouse_one_1229.py']

# source data config
dataset_info = dict(
    dataset_name='mouse_dannce_3d_p12',
    paper_info=dict(
        author='Dunn', title='DANNCE', container='Nature Methods', year='2021', homepage='',
    ),
    keypoint_info={
        0:
            dict(
                name='left_ear_tip', id=0, color=[92, 94, 170], type='upper', swap='right_ear_tip'),
        1:
            dict(
                name='right_ear_tip',
                id=1,
                color=[92, 94, 170],
                type='upper',
                swap='left_ear_tip'),
        2:
            dict(
                name='nose',
                id=2,
                color=[92, 94, 170],
                type='upper',
                swap=''),
        3:
            dict(
                name='neck',
                id=3,
                color=[221, 94, 86],
                type='upper',
                swap=''),
        4:
            dict(name='body_middle', id=4, color=[221, 94, 86], type='upper', swap=''),
        5:
            dict(name='tail_root', id=5, color=[221, 94, 86], type='upper', swap=''),
        6:
            dict(
                name='tail_middle', id=6, color=[221, 94, 86], type='lower',
                swap=''),
        7:
            dict(
                name='tail_end', id=7, color=[221, 94, 86], type='upper', swap=''),
        # 8:
        # dict(
        #     name='left_paw',
        #     id=8,
        #     color=[187,97,166],
        #     type='upper',
        #     swap='right_paw'),
        9:
            dict(
                name='left_paw_end',
                id=9,
                color=[187, 97, 166],
                type='upper',
                swap='right_paw_end'),
        # 10:
        # dict(
        #     name='left_elbow',
        #     id=10,
        #     color=[187,97,166],
        #     type='lower',
        #     swap='right_elbow'),
        # 11:
        # dict(
        #     name='left_shoulder',
        #     id=11,
        #     color=[187,97,166],
        #     type='lower',
        #     swap='right_shoulder'),
        # 12:
        # dict(
        #     name='right_paw',
        #     id=12,
        #     color=[109, 192, 91],
        #     type='upper',
        #     swap='left_paw'),
        13:
            dict(
                name='right_paw_end',
                id=13,
                color=[109, 192, 91],
                type='upper',
                swap='left_paw_end'),
        # 14:
        # dict(
        #     name='right_elbow',
        #     id=14,
        #     color=[109, 192, 91],
        #     type='lower',
        #     swap='left_elbow'),
        # 15:
        # dict(
        #     name='right_shoulder',
        #     id=15,
        #     color=[109, 192, 91],
        #     type='lower',
        #     swap='left_shoulder'),
        16:
            dict(
                name='left_foot',
                id=16,
                color=[210, 220, 88],
                type='upper',
                swap='right_foot'),
        # 17:
        # dict(
        #     name='left_knee',
        #     id=17,
        #     color=[210, 220, 88],
        #     type='upper',
        #     swap='right_knee'),
        # 18:
        # dict(
        #     name='left_hip',
        #     id=18,
        #     color=[210, 220, 88],
        #     type='lower',
        #     swap='right_hip'),
        19:
            dict(
                name='right_foot',
                id=19,
                color=[98, 201, 211],
                type='lower',
                swap='left_foot'),
        # 20:
        # dict(
        #     name ="right_knee",
        #     id= 20,
        #     color=[98,201,211],
        #     type='lower',
        #     swap="left_knee"
        # ),
        # 21:
        # dict(name="right_hip",
        #     id=21,
        #     color=[98,201,211],
        #     type='lower',
        #     swap='left_hip'
        # )
    },
    skeleton_info={
        0: dict(link=('tail_root', 'tail_middle'), id=0, color=[221, 94, 86]),
        1: dict(link=('tail_middle', 'tail_end'), id=1, color=[221, 94, 86]),
        2: dict(link=('tail_root', 'left_foot'), id=2, color=[210, 220, 88]),
        3: dict(link=('body_middle', 'left_foot'), id=3, color=[210, 220, 88]),
        4: dict(link=('tail_root', 'right_foot'), id=4, color=[98, 201, 211]),
        5: dict(link=('body_middle', 'right_foot'), id=5, color=[98, 201, 211]),
        6: dict(link=('tail_root', 'body_middle'), id=6, color=[221, 94, 86]),
        7: dict(link=('body_middle', 'neck'), id=7, color=[221, 94, 86]),
        8: dict(link=('neck', 'nose'), id=8, color=[221, 94, 86]),
        9: dict(link=('nose', 'left_ear_tip'), id=9, color=[92, 94, 170]),
        10: dict(link=('nose', 'right_ear_tip'), id=10, color=[92, 94, 170]),
        11: dict(link=('neck', 'left_paw_end'), id=11, color=[187, 97, 166]),
        12: dict(link=('body_middle', 'left_paw_end'), id=12, color=[187, 97, 166]),
        13: dict(link=('neck', 'right_paw_end'), id=13, color=[109, 192, 91]),
        14: dict(link=('body_middle', 'right_paw_end'), id=14, color=[109, 192, 91]),

        # 3: dict(link=('left_hip', 'left_knee'), id=3, color=[210, 220, 88]),
        # 4: dict(link=('left_knee', 'left_foot'), id=4, color=[210, 220, 88]),
        # 5: dict(link=('tail_root', 'right_hip'), id=5, color=[98,201,211]),
        # 6: dict(link=('right_hip', 'right_knee'), id=6, color=[98,201,211]),
        # 7: dict(link=('right_knee', 'right_foot'), id=7, color=[98,201,211]),
        # 8: dict(link=('tail_root', 'body_middle'), id=8, color=[221,94,86]),
        # 9: dict(link=('body_middle', 'neck'), id=9, color=[221,94,86]),
        # 10: dict(link=('neck', 'nose'), id=10, color=[221,94,86]),
        # 11: dict(link=('nose', 'left_ear_tip'), id=11, color=[92,94,170]),
        # 12: dict(link=('nose', 'right_ear_tip'), id=12, color=[92,94,170]),
        # 13: dict(link=('neck', 'left_shoulder'), id=13, color=[187,97,166]),
        # 14: dict(link=('left_shoulder', 'left_elbow'), id=14, color=[187,97,166]),
        # 15: dict(link=('left_elbow', 'left_paw_end'), id=15, color=[187,97,166]),
        # 16: dict(link=('left_paw_end', 'left_paw'), id=16, color=[187,97,166]),
        # 17: dict(link=('neck', 'right_shoulder'), id=17, color=[109, 192, 91]),
        # 18: dict(link=('right_shoulder', 'right_elbow'), id=18, color=[109, 192, 91]),
        # 19: dict(link=('right_elbow', 'right_paw_end'), id=19, color=[109, 192, 91]),
        # 20: dict(link=('right_paw_end', 'right_paw'), id=20, color=[109, 192, 91])
    },
    joint_weights=[
        1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
        1., 1., 1., 1., 1., 1.
    ],
    # Note: The original paper did not provide enough information about
    # the sigmas. We modified from 'https://github.com/cocodataset/'
    # 'cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py#L523'
    sigmas=[
        0.025, 0.025, 0.025,
        0.1, 0.12, 0.035, 0.035, 0.025,
        0.03, 0.03, 0.03, 0.1,
        0.03, 0.03, 0.03, 0.1,
        0.025, 0.025, 0.1,
        0.025, 0.025, 0.1
    ])

source_channel_cfg = dict(
    num_output_channels=12,
    dataset_joints=12,
    dataset_channel=[
        0, 1, 2, 3, 4, 5, 6, 7, 9, 13, 16, 19
    ],
    inference_channel=[
        0, 1, 2, 3, 4, 5, 6, 7, 9, 13, 16, 19
    ])

source_data_cfg = dict(
    image_size=[256, 256],
    heatmap_size=[64, 64],
    num_output_channels=source_channel_cfg['num_output_channels'],
    num_joints=source_channel_cfg['dataset_joints'],
    dataset_channel=source_channel_cfg['dataset_channel'],
    inference_channel=source_channel_cfg['inference_channel'],
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
    use_different_joint_weights=False)

# target data config
dataset_info = dict(
    dataset_name="mouse_one_1229",
    paper_info=dict(
        author='zjc'),
    keypoint_info={
        0:
            dict(name='left_ear_tip', id=0, color=[92, 94, 170], type='upper', swap='right_ear_tip'),
        1: dict(name='right_ear_tip',
                id=1,
                color=[92, 94, 170],
                type='upper',
                swap='left_ear_tip'),
        2: dict(name='nose',
                id=2,
                color=[92, 94, 170],
                type='upper',
                swap=''),
        3: dict(name='neck',
                id=3,
                color=[221, 94, 86],
                type='upper',
                swap=''),
        4: dict(name='body_middle', id=4, color=[221, 94, 86], type='upper', swap=''),
        5: dict(name='tail_root', id=5, color=[221, 94, 86], type='upper', swap=''),
        6:
            dict(
                name='tail_middle', id=6, color=[221, 94, 86], type='lower',
                swap=''),
        7:
            dict(
                name='tail_end', id=7, color=[221, 94, 86], type='upper', swap=''),
        8:
            dict(
                name='left_paw',
                id=8,
                color=[187, 97, 166],
                type='upper',
                swap='right_paw'),
        # 9: dict(
        #     name='left_shoulder',
        #     id=9,
        #     color=[187, 97, 166],
        #     type='lower',
        #     swap='right_shoulder'),
        10: dict(
            name='right_paw',
            id=10,
            color=[109, 192, 91],
            type='upper',
            swap='left_paw'),

        # 11: dict(
        #     name='right_shoulder',
        #     id=11,
        #     color=[109, 192, 91],
        #     type='lower',
        #     swap='left_shoulder'),
        12: dict(
            name='left_foot',
            id=12,
            color=[210, 220, 88],
            type='upper',
            swap='right_foot'),
        # 13: dict(
        #     name='left_hip',
        #     id=13,
        #     color=[210, 220, 88],
        #     type='lower',
        #     swap='right_hip'),
        14: dict(
            name='right_foot',
            id=14,
            color=[98, 201, 211],
            type='lower',
            swap='left_foot'),
        # 15: dict(name="right_hip",
        #          id=25,
        #          color=[98, 201, 211],
        #          type='lower',
        #          swap='left_hip'
        #          )
    },
    skeleton_info={
        0: dict(link=('tail_root', 'tail_middle'), id=0, color=[221, 94, 86]),
        1: dict(link=('tail_middle', 'tail_end'), id=1, color=[221, 94, 86]),
        2: dict(link=('tail_root', 'left_foot'), id=2, color=[210, 220, 88]),
        3: dict(link=('body_middle', 'left_foot'), id=3, color=[210, 220, 88]),
        4: dict(link=('tail_root', 'right_foot'), id=4, color=[98, 201, 211]),
        5: dict(link=('body_middle', 'right_foot'), id=5, color=[98, 201, 211]),
        6: dict(link=('tail_root', 'body_middle'), id=6, color=[221, 94, 86]),
        7: dict(link=('body_middle', 'neck'), id=7, color=[221, 94, 86]),
        8: dict(link=('neck', 'nose'), id=8, color=[221, 94, 86]),
        9: dict(link=('nose', 'left_ear_tip'), id=9, color=[92, 94, 170]),
        10: dict(link=('nose', 'right_ear_tip'), id=10, color=[92, 94, 170]),
        11: dict(link=('neck', 'left_paw'), id=11, color=[187, 97, 166]),
        12: dict(link=('body_middle', 'left_paw'), id=12, color=[187, 97, 166]),
        13: dict(link=('neck', 'right_paw'), id=13, color=[109, 192, 91]),
        14: dict(link=('body_middle', 'right_paw'), id=14, color=[109, 192, 91]),
    },
    joint_weights=[
        1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.
    ],
    sigmas=[0.1, 0.1, 0.1,
            0.1, 0.1, 0.1,
            0.1, 0.1, 0.1,
            0.1, 0.1, 0.1,
            0.1, 0.1, 0.1, 0.1]
)

target_channel_cfg = dict(
    num_output_channels=12,
    dataset_joints=12,
    dataset_channel=[0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14],
    inference_channel=[0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14])

target_data_cfg = dict(
    image_size=[256, 256],
    heatmap_size=[64, 64],
    num_output_channels=target_channel_cfg['num_output_channels'],
    num_joints=target_channel_cfg['dataset_joints'],
    dataset_channel=target_channel_cfg['dataset_channel'],
    inference_channel=target_channel_cfg['inference_channel'],
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
    use_different_joint_weights=False)

"""data config"""
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

source_data_root = 'D:/Datasets/transfer_mouse/dannce_20230130'
target_data_root = 'D:/Datasets/transfer_mouse/onemouse1229'

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=2,
    val_dataloader=dict(samples_per_gpu=4),
    test_dataloader=dict(samples_per_gpu=4),
    source=dict(
        type='MouseDannce3dDataset',
        ann_file=f'{source_data_root}/annotations_visible_train930_new.json',
        ann_3d_file=f'{source_data_root}/joints_3d.json',
        cam_file=f'{source_data_root}/cams.pkl',
        img_prefix=f'{source_data_root}/images_gray/',
        data_cfg=source_data_cfg,
        pipeline=train_pipeline,
        dataset_info={{_base_.dataset_info}}),

    target=dict(
        type="Mouse12293dDatasetMview",
        ann_file=f"{target_data_root}/anno_20221229-1-012345.json",
        ann_3d_file=f"{target_data_root}/anno_20221229_joints_3d.json",
        cam_file=f"{target_data_root}/calibration_adjusted.json",
        img_prefix=f'{target_data_root}/',
        data_cfg=target_data_cfg,
        pipeline=val_pipeline,
        dataset_info={{_base_.dataset_info}})
)

"""model config"""

train_cfg = dict(
    supervised_2d=True,  # use the 2d ground truth to train keypoint_head
    contrastive_feature=True,  # use the sup_con_loss to train feature_head
    supervised_3d=True,  # use the 3d ground truth to train triangulate_head
    unSupervised_3d=False,  # use the triangulation residual loss to train triangulate_head
),

model = dict(
    type='TriangNet',
    backbone=dict(
        type='HRNet',
        in_channels=3,
        extra=dict(
            stage1=dict(
                num_modules=1,
                num_branches=1,
                block='BOTTLENECK',
                num_blocks=(4,),
                num_channels=(64,)),
            stage2=dict(
                num_modules=1,
                num_branches=2,
                block='BASIC',
                num_blocks=(4, 4),
                num_channels=(48, 96)),
            stage3=dict(
                num_modules=4,
                num_branches=3,
                block='BASIC',
                num_blocks=(4, 4, 4),
                num_channels=(48, 96, 192)),
            stage4=dict(
                num_modules=3,
                num_branches=4,
                block='BASIC',
                num_blocks=(4, 4, 4, 4),
                num_channels=(48, 96, 192, 384))),
    ),
    keypoint_head=dict(
        type='TopdownHeatmapSimpleHead',
        in_channels=48,
        out_channels=source_channel_cfg['num_output_channels'],
        num_deconv_layers=0,
        extra=dict(final_conv_kernel=1, ),
        loss_keypoint=dict(type='JointsMSELoss', use_target_weight=True)),

    triangulate_head=dict(
        type='TriangulateHead',
        num_cams=6,
        img_shape=[256, 256],
        heatmap_shape=[64, 64],
        softmax_heatmap=True,
        loss_3d_super=dict(type='MSELoss',
                           use_target_weight=True,
                           loss_weight=1.),
        train_cfg=train_cfg,
    ),
    test_cfg=dict(
        flip_test=False,
        post_process='default',
        shift_heatmap=True,
        modulate_kernel=11)
)
