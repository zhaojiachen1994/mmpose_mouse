mouse1229_dataset_info = dict(
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
        9: dict(
            name='right_paw',
            id=9,
            color=[109, 192, 91],
            type='upper',
            swap='left_paw'),
        10: dict(
            name='left_foot',
            id=10,
            color=[210, 220, 88],
            type='upper',
            swap='right_foot'),
        11: dict(
            name='right_foot',
            id=11,
            color=[98, 201, 211],
            type='lower',
            swap='left_foot'),
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
        1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.
    ],
    sigmas=[0.1, 0.1, 0.1,
            0.1, 0.1, 0.1,
            0.1, 0.1, 0.1,
            0.1, 0.1, 0.1]
)

mouse1229_channel_cfg = dict(
    num_output_channels=12,
    dataset_joints=12,
    dataset_channel=[0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14],
    inference_channel=[0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14])

mouse1229_data_cfg = dict(
    image_size=[256, 256],
    heatmap_size=[64, 64],
    num_output_channels=mouse1229_channel_cfg['num_output_channels'],
    num_joints=mouse1229_channel_cfg['dataset_joints'],
    dataset_channel=mouse1229_channel_cfg['dataset_channel'],
    inference_channel=mouse1229_channel_cfg['inference_channel'],
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
