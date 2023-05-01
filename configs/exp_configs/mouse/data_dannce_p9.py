dannce_channel_cfg = dict(
    num_output_channels=9,
    dataset_joints=9,
    dataset_channel=[2, 0, 1, 3, 5, 8, 12, 16, 19],
    inference_channel=[2, 0, 1, 3, 5, 8, 12, 16, 19])

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
