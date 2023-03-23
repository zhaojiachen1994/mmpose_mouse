dataset_info = dict(
    dataset_name='mars_p5',
    paper_info=dict(
        author='aa',
        title='aa',
        container='aa',
        year='aa',
        homepage='',
    ),
    keypoint_info={
        0: dict(
            name='nose',
            id=2,
            color=[92, 94, 170],
            type='upper',
            swap=''),
        1: dict(
            name='left_ear', id=0, color=[92, 94, 170], type='upper', swap='right_ear'),
        2: dict(
            name='right_ear',
            id=1,
            color=[92, 94, 170],
            type='upper',
            swap='left_ear'),
        3: dict(name='neck', id=3, color=[221, 94, 86], type='upper', swap=''),
        4: dict(name='tail_root', id=5, color=[221, 94, 86], type='upper', swap=''),
    },
    skeleton_info={
        0: dict(link=('neck', 'nose'), id=0, color=[221, 94, 86]),
        1: dict(link=('nose', 'left_ear'), id=1, color=[92, 94, 170]),
        2: dict(link=('nose', 'right_ear'), id=10, color=[92, 94, 170]),
        3: dict(link=('tail_root', 'neck'), id=7, color=[221, 94, 86]),
    },
    joint_weights=[1., 1., 1., 1., 1.],
    sigmas=[0.02, 0.02, 0.02, 0.02, 0.02]
)
