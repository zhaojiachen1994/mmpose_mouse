_base_ = [
    '../_base_/default_runtime.py',
    '../_base_/datasets/h36m.py'
]

channel_cfg = dict(
    num_output_channels=17,
    dataset_joints=17,
    dataset_channel=[
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
    ],
    inference_channel=[
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16
    ])

"""total model pretrain weights"""
load_from = "D:/Pycharm Projects-win/mm_mouse/mmpose/official_checkpoint/hrnet_w48_h36m_256x256-78e88d08_20210621.pth"

train_cfg = dict(
    supervised_2d=True,  # use the 2d ground truth to train keypoint_head
    contrastive_feature=True,  # use the sup_con_loss to train feature_head
    supervised_3d=True,  # use the 3d ground truth to train triangulate_head
    unSupervised_3d=False,  # use the triangulation residual loss to train triangulate_head
),

"""model config"""
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
        out_channels=channel_cfg['num_output_channels'],
        num_deconv_layers=0,
        extra=dict(final_conv_kernel=1, ),
        loss_keypoint=dict(type='JointsMSELoss', use_target_weight=True)),

    triangulate_head=dict(
        type='TriangulateHead',
        num_cams=4,
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

# data settings
data_root = "D:/Datasets/h36m_dataset/human3.6m_parse"
data_cfg = dict(
    image_size=[256, 256],
    heatmap_size=[64, 64],
    num_output_channels=channel_cfg['num_output_channels'],
    num_joints=channel_cfg['dataset_joints'],
    dataset_channel=channel_cfg['dataset_channel'],
    inference_channel=channel_cfg['inference_channel'],
    soft_nms=False,
    nms_thr=1.0,
    oks_thr=0.9,
    vis_thr=0.2,
    use_gt_bbox=True,
    det_bbox_thr=0.0,
    bbox_file='',
    space_size=[0, 0, 0],
    space_center=[0, 0, 0],
    cube_size=[0, 0, 0],
    num_cameras=6,
    use_different_joint_weights=False

)

# 3D joint normalization parameters
# From file: '{data_root}/annotation_body3d/fps50/joint3d_rel_stats.pkl'
joint_3d_normalize_param = dict(
    mean=[[-2.55652589e-04, -7.11960570e-03, -9.81433052e-04],
          [-5.65463051e-03, 3.19636009e-01, 7.19329269e-02],
          [-1.01705840e-02, 6.91147892e-01, 1.55352986e-01],
          [2.55651315e-04, 7.11954606e-03, 9.81423866e-04],
          [-5.09729780e-03, 3.27040413e-01, 7.22258095e-02],
          [-9.99656606e-03, 7.08277383e-01, 1.58016408e-01],
          [2.90583676e-03, -2.11363307e-01, -4.74210915e-02],
          [5.67537804e-03, -4.35088906e-01, -9.76974016e-02],
          [5.93884964e-03, -4.91891970e-01, -1.10666618e-01],
          [7.37352083e-03, -5.83948619e-01, -1.31171400e-01],
          [5.41920653e-03, -3.83931702e-01, -8.68145417e-02],
          [2.95964662e-03, -1.87567488e-01, -4.34536934e-02],
          [1.26585822e-03, -1.20170579e-01, -2.82526049e-02],
          [4.67186639e-03, -3.83644089e-01, -8.55125784e-02],
          [1.67648571e-03, -1.97007177e-01, -4.31368364e-02],
          [8.70569015e-04, -1.68664569e-01, -3.73902498e-02]],
    std=[[0.11072244, 0.02238818, 0.07246294],
         [0.15856311, 0.18933832, 0.20880479],
         [0.19179935, 0.24320062, 0.24756193],
         [0.11072181, 0.02238805, 0.07246253],
         [0.15880454, 0.19977188, 0.2147063],
         [0.18001944, 0.25052739, 0.24853247],
         [0.05210694, 0.05211406, 0.06908241],
         [0.09515367, 0.10133032, 0.12899733],
         [0.11742458, 0.12648469, 0.16465091],
         [0.12360297, 0.13085539, 0.16433336],
         [0.14602232, 0.09707956, 0.13952731],
         [0.24347532, 0.12982249, 0.20230181],
         [0.2446877, 0.21501816, 0.23938235],
         [0.13876084, 0.1008926, 0.1424411],
         [0.23687529, 0.14491219, 0.20980829],
         [0.24400695, 0.23975028, 0.25520584]])

# 2D joint normalization parameters
# From file: '{data_root}/annotation_body3d/fps50/joint2d_stats.pkl'
joint_2d_normalize_param = dict(
    mean=[[532.08351635, 419.74137558], [531.80953144, 418.2607141],
          [530.68456967, 493.54259285], [529.36968722, 575.96448516],
          [532.29767646, 421.28483336], [531.93946631, 494.72186795],
          [529.71984447, 578.96110365], [532.93699382, 370.65225054],
          [534.1101856, 317.90342311], [534.55416813, 304.24143901],
          [534.86955004, 282.31030885], [534.11308566, 330.11296796],
          [533.53637525, 376.2742511], [533.49380107, 391.72324565],
          [533.52579142, 330.09494668], [532.50804964, 374.190479],
          [532.72786934, 380.61615716]],
    std=[[107.73640054, 63.35908715], [119.00836213, 64.1215443],
         [119.12412107, 50.53806215], [120.61688045, 56.38444891],
         [101.95735275, 62.89636486], [106.24832897, 48.41178119],
         [108.46734966, 54.58177071], [109.07369806, 68.70443672],
         [111.20130351, 74.87287863], [111.63203838, 77.80542514],
         [113.22330788, 79.90670556], [105.7145833, 73.27049436],
         [107.05804267, 73.93175781], [107.97449418, 83.30391802],
         [121.60675105, 74.25691526], [134.34378973, 77.48125087],
         [131.79990652, 89.86721124]])

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
        keys_list=[
            'dataset', 'ann_info', 'joints_4d',
            'subject', 'action_idx', 'subaction_idx', 'frame_idx'
        ]),
    dict(
        type='GroupCams',
        keys=['img', 'target', 'target_weight', 'proj_mat', 'joints_3d']
    ),
    dict(
        type="Collect",
        keys=['img', 'target', 'target_weight', 'joints_4d', 'proj_mat', 'joints_3d'],
        meta_keys=['image_file', 'subject', 'action_idx', 'subaction_idx', 'frame_idx']
    )
]

data = dict(
    samples_per_gpu=5,
    workers_per_gpu=2,
    val_dataloader=dict(samples_per_gpu=64),
    test_dataloader=dict(samples_per_gpu=64),
    train=dict(
        type="Body3DH36MMviewDataset",
        ann_file=f"{data_root}/annotations/Human36M_subject1_data_reorder.json",
        ann_3d_file=f"{data_root}/annotations/Human36M_subject1_joint_3d.json",
        cam_file=f"{data_root}/annotations/Human36M_subject1_camera.json",
        img_prefix=f"{data_root}/images/",
        data_cfg=data_cfg,
        pipeline=train_pipeline,
        dataset_info={{_base_.dataset_info}}),
)
