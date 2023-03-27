import warnings
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import mmcv
import numpy as np
import torch

warnings.filterwarnings("ignore")

from mmpose.datasets import DatasetInfo
from mmpose.datasets import build_dataset
from mmpose.apis import (init_pose_model)
from mmpose.core import imshow_keypoints, imshow_keypoints_3d

has_mmdet = True


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--pose_config', type=str, help='the config file.')
    parser.add_argument('--pose_checkpoint', type=str, help='the checkpoint file')
    parser.add_argument('--output_path', type=str, help='path to save images')
    parser.add_argument(
        '--show',
        action='store_true',
        default=False,
        help='whether to show img')
    parser.add_argument(
        '--out-img-root',
        type=str,
        default='',
        help='root of the output img file. '
             'Default not saving the visualization images.')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--det-cat-id',
        type=int,
        default=1,
        help='Category id for bounding box detection model')
    parser.add_argument(
        '--bbox-thr',
        type=float,
        default=0.3,
        help='Bounding box score threshold')
    parser.add_argument(
        '--kpt-thr', type=float, default=0.2, help='Keypoint score threshold')
    parser.add_argument(
        '--radius',
        type=int,
        default=4,
        help='Keypoint radius for visualization')
    parser.add_argument(
        '--thickness',
        type=int,
        default=1,
        help='Link thickness for visualization')

    assert has_mmdet, 'Please install mmdet to run the demo.'

    args = parser.parse_args()
    return args


def inference(args):
    config = mmcv.Config.fromfile(args.pose_config)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    pose_model = init_pose_model(args.pose_config, checkpoint=args.pose_checkpoint, device=device)
    dataset = config.data['test']['type']
    dataset_info = DatasetInfo(config.dataset_info)

    dataset = build_dataset(config.data.test)
    num_cams = config.data_cfg['num_cameras']
    for i in range(15):
        data = dataset.__getitem__(i)
        imgs = data['img']
        proj_matrices = data['proj_mat']
        imgs = torch.unsqueeze(imgs, 0)
        proj_matrices = torch.from_numpy(np.expand_dims(proj_matrices, 0))
        imgs = imgs.to(device)
        proj_matrices = proj_matrices.to(device)
        result = pose_model.forward(imgs,
                                    proj_mat=proj_matrices,
                                    img_metas=None,
                                    return_loss=False,
                                    return_heatmap=False)
        kpt_3d_pred = result['preds']  # [bs, num_joints, 3]
        kpt_2d_pred = result['kp_2d_preds']  # [bs, cams, 16, 3]
        res_triang = result['res_triang']  # [bs, num_joints]
        kpt_2d_reproject = result['kp_2d_reproject']
        res_thr = 500
        kpt_3d_score = np.where((res_triang > 0.001) & (res_triang < res_thr), True, False) * 1
        kpt_3d_score = np.expand_dims(kpt_3d_score, -1)  # [bs, num_joints, 1]

        """plot the 3d results"""
        kpt_3d_pred = np.concatenate([kpt_3d_pred, kpt_3d_score], axis=-1)

        img = imshow_keypoints_3d(
            [{"keypoints_3d": kpt_3d_pred[0]}],
            img=None,
            skeleton=dataset_info.skeleton,
            pose_kpt_color=dataset_info.pose_kpt_color,
            pose_link_color=dataset_info.pose_link_color,
            vis_height=400,
            kpt_score_thr=0.2,
            axis_azimuth=70,
            axis_limit=100,
            axis_dist=10.0,
            axis_elev=15.0,

        )

        # img = imshow_multiview_keypoints_3d(
        #     [kpt_3d_pred[0]],
        #     skeleton=dataset_info.skeleton,
        #     pose_kpt_color=dataset_info.pose_kpt_color,
        #     pose_link_color=dataset_info.pose_link_color,
        #     space_size=[300, 300, 300],
        #     space_center=[0, 0, 150],
        #     kpt_score_thr=0.1,
        # )

        # plot the 2d predictions
        rr = np.array(data['img_metas'].data['resize_ratio'])
        offset = np.array(data['img_metas'].data['bbox_offset'])  # [num_cams, 2]
        # kpt_2d_pred_original_img = kpt_2d_pred
        # kpt_2d_pred_original_img[..., :-1] = kpt_2d_pred[..., :-1] / rr[None, :, None, None] \
        #                                      + offset[None, :, None, :]  # [bs, num_cams, num_joints, 2]
        kpt_2d_pred_original_img = kpt_2d_reproject
        kpt_2d_pred_original_img = kpt_2d_reproject / rr[None, :, None, None] \
                                   + offset[None, :, None, :]  # [bs, num_cams, num_joints, 2]

        f, axarr = plt.subplots(nrows=1, ncols=7, figsize=[140, 30])
        axarr[0].imshow(img, interpolation='nearest')

        for c in range(num_cams):
            image_file = data['img_metas'].data['image_file'][c]
            img = mmcv.imread(image_file, channel_order='rgb')
            pose_result = kpt_2d_pred_original_img[0, c, ...]
            # ic(pose_result)
            if pose_result.shape[1] == 2:
                pose_result = np.concatenate([pose_result, np.ones([pose_result.shape[0], 1])], axis=1)
            img_vis_2d = imshow_keypoints(
                image_file,
                [pose_result],
                skeleton=dataset_info.skeleton,
                kpt_score_thr=0.2,
                radius=10,
                thickness=10,
                pose_kpt_color=dataset_info.pose_kpt_color,
                pose_link_color=dataset_info.pose_link_color,
            )
            axarr[c + 1].imshow(img_vis_2d, interpolation='nearest')

        plt.savefig(f"{args.output_path}/mars2dannce_{i}_repro_un3d.png")


if __name__ == "__main__":
    args = parse_args()

    # args.pose_config = "../configs/exp_configs/Triangnet_infer_pretrain_mars_to_dannce.py"
    # args.pose_checkpoint = "../work_dirs/hrnet_w48_concat_mars_p5_256x256/best_AP_epoch_5.pth"

    args.pose_config = "../configs/exp_configs/Triangnet_mars_to_dannce_p5_un3d_epoch.py"
    args.pose_checkpoint = "best_MPJPE_epoch_5.pth"

    args.output_path = "../work_dirs/temp"
    inference(args)
