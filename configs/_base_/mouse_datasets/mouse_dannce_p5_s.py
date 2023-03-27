dannce_dataset_info = dict(
    dataset_name='mouse_dannce_3d_p12',
    paper_info=dict(
        author='Dunn',
        title='DANNCE',
        container='Nature Methods',
        year='2021',
        homepage='',
    ),
    keypoint_info={
        0: dict(
            name='nose',
            id=0,
            color=[92, 94, 170],
            type='upper',
            swap=''),
        1: dict(
            name='left_ear', id=1, color=[92, 94, 170], type='upper', swap='right_ear'),
        2: dict(
            name='right_ear',
            id=2,
            color=[92, 94, 170],
            type='upper',
            swap='left_ear'),
        3: dict(name='neck', id=3, color=[221, 94, 86], type='upper', swap=''),
        4: dict(name='tail_root', id=4, color=[221, 94, 86], type='upper', swap=''),
    },
    skeleton_info={
        0: dict(link=('neck', 'nose'), id=0, color=[221, 94, 86]),
        1: dict(link=('nose', 'left_ear'), id=1, color=[92, 94, 170]),
        2: dict(link=('nose', 'right_ear'), id=2, color=[92, 94, 170]),
        3: dict(link=('tail_root', 'neck'), id=3, color=[221, 94, 86]),
    },
    joint_weights=[1., 1., 1., 1., 1.],
    sigmas=[0.5, 0.5, 0.5, 0.5, 0.5]
)

dannce_channel_cfg = dict(
    num_output_channels=5,
    dataset_joints=5,
    dataset_channel=[2, 0, 1, 3, 5],
    inference_channel=[2, 0, 1, 3, 5])

dannce_data_cfg = dict(
    image_size=[256, 256],
    heatmap_size=[64, 64],
    num_output_channels=dannce_channel_cfg['num_output_channels'],
    num_joints=dannce_channel_cfg['dataset_joints'],
    dataset_channel=dannce_channel_cfg['dataset_channel'],
    inference_channel=dannce_channel_cfg['inference_channel'],
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
    use_different_joint_weights=False
)
