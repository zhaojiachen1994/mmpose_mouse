mpii_channel_cfg = dict(
    num_output_channels=16,
    dataset_joints=16,
    dataset_channel=[6, 2, 1, 0, 3, 4, 5, 7, 8, 9, 13, 14, 15, 12, 11, 10],
    inference_channel=[6, 2, 1, 0, 3, 4, 5, 7, 8, 9, 13, 14, 15, 12, 11, 10])

mpii_data_cfg = dict(
    image_size=[256, 256],
    heatmap_size=[64, 64],
    num_output_channels=mpii_channel_cfg['num_output_channels'],
    num_joints=mpii_channel_cfg['dataset_joints'],
    dataset_channel=mpii_channel_cfg['dataset_channel'],
    inference_channel=mpii_channel_cfg['inference_channel'],
    use_gt_bbox=True,
    bbox_file=None,
)
