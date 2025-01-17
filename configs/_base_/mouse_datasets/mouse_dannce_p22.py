dataset_info = dict(
    dataset_name='mouse_dannce_3d',
    paper_info=dict(
        author='Dunn',
        title='DANNCE',
        container='Nature Methods',
        year='2021',
        homepage='',
    ),
    keypoint_info={
        0:
        dict(
            name='left_ear_tip', id=0, color=[92,94,170], type='upper', swap='right_ear_tip'),
        1:
        dict(
            name='right_ear_tip',
            id=1,
            color=[92,94,170],
            type='upper',
            swap='left_ear_tip'),
        2:
        dict(
            name='nose',
            id=2,
            color=[92,94,170],
            type='upper',
            swap=''),
        3:
        dict(
            name='neck',
            id=3,
            color=[221,94,86],
            type='upper',
            swap=''),
        4:
        dict(name='body_middle', id=4, color=[221,94,86], type='upper', swap=''),
        5:
        dict(name='tail_root', id=5, color=[221,94,86], type='upper', swap=''),
        6:
        dict(
            name='tail_middle', id=6, color=[221,94,86], type='lower',
            swap=''),
        7:
        dict(
            name='tail_end', id=7, color=[221,94,86], type='upper', swap=''),
        8:
        dict(
            name='left_paw',
            id=8,
            color=[187,97,166],
            type='upper',
            swap='right_paw'),
        9:
        dict(
            name='left_paw_end',
            id=9,
            color=[187,97,166],
            type='upper',
            swap='right_paw_end'),
        10:
        dict(
            name='left_elbow',
            id=10,
            color=[187,97,166],
            type='lower',
            swap='right_elbow'),
        11:
        dict(
            name='left_shoulder',
            id=11,
            color=[187,97,166],
            type='lower',
            swap='right_shoulder'),
        12:
        dict(
            name='right_paw',
            id=12,
            color=[109, 192, 91],
            type='upper',
            swap='left_paw'),
        13:
        dict(
            name='right_paw_end',
            id=13,
            color=[109, 192, 91],
            type='upper',
            swap='left_paw_end'),
        14:
        dict(
            name='right_elbow',
            id=14,
            color=[109, 192, 91],
            type='lower',
            swap='left_elbow'),
        15:
        dict(
            name='right_shoulder',
            id=15,
            color=[109, 192, 91],
            type='lower',
            swap='left_shoulder'),
        16:
        dict(
            name='left_foot',
            id=16,
            color=[210, 220, 88],
            type='upper',
            swap='right_foot'),
        17:
        dict(
            name='left_knee',
            id=17,
            color=[210, 220, 88],
            type='upper',
            swap='right_knee'),
        18:
        dict(
            name='left_hip',
            id=18,
            color=[210, 220, 88],
            type='lower',
            swap='right_hip'),
        19:
        dict(
            name='right_foot',
            id=19,
            color=[98,201,211],
            type='lower',
            swap='left_foot'),
        20: 
        dict(
            name ="right_knee", 
            id= 20,
            color=[98,201,211],
            type='lower',
            swap="left_knee"
        ), 
        21: 
        dict(name="right_hip", 
            id=21, 
            color=[98,201,211], 
            type='lower',
            swap='left_hip'
        )
    },
    skeleton_info={
        0: dict(link=('tail_root', 'tail_middle'), id=0, color=[221,94,86]),
        1: dict(link=('tail_middle', 'tail_end'), id=1, color=[221,94,86]),
        2: dict(link=('tail_root', 'left_hip'), id=2, color=[210, 220, 88]),
        3: dict(link=('left_hip', 'left_knee'), id=3, color=[210, 220, 88]),
        4: dict(link=('left_knee', 'left_foot'), id=4, color=[210, 220, 88]),
        5: dict(link=('tail_root', 'right_hip'), id=5, color=[98,201,211]),
        6: dict(link=('right_hip', 'right_knee'), id=6, color=[98,201,211]),
        7: dict(link=('right_knee', 'right_foot'), id=7, color=[98,201,211]),
        8: dict(link=('tail_root', 'body_middle'), id=8, color=[221,94,86]),
        9: dict(link=('body_middle', 'neck'), id=9, color=[221,94,86]),
        10: dict(link=('neck', 'nose'), id=10, color=[221,94,86]),
        11: dict(link=('nose', 'left_ear_tip'), id=11, color=[92,94,170]),
        12: dict(link=('nose', 'right_ear_tip'), id=12, color=[92,94,170]),
        13: dict(link=('neck', 'left_shoulder'), id=13, color=[187,97,166]),
        14: dict(link=('left_shoulder', 'left_elbow'), id=14, color=[187,97,166]),
        15: dict(link=('left_elbow', 'left_paw_end'), id=15, color=[187,97,166]),
        16: dict(link=('left_paw_end', 'left_paw'), id=16, color=[187,97,166]),
        17: dict(link=('neck', 'right_shoulder'), id=17, color=[109, 192, 91]),
        18: dict(link=('right_shoulder', 'right_elbow'), id=18, color=[109, 192, 91]),
        19: dict(link=('right_elbow', 'right_paw_end'), id=19, color=[109, 192, 91]),
        20: dict(link=('right_paw_end', 'right_paw'), id=20, color=[109, 192, 91])
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
