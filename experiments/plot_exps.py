import os.path as osp
import warnings
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import mmcv
import numpy as np
import torch
from icecream import ic

warnings.filterwarnings("ignore")

from mmpose.datasets import DatasetInfo
from mmpose.datasets import build_dataset
from mmpose.apis import (init_pose_model)
from mmpose.core import imshow_keypoints, imshow_keypoints_3d

has_mmdet = True


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, help='the config file.')
    parser.add_argument('--checkpoint', type=str, help='the checkpoint file')
    parser.add_argument('--work_dir', type=str, help='path to save images')
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
    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        work_dir = args.work_dir
    else:
        # use config filename as default work_dir if cfg.work_dir is None
        work_dir = osp.join('./work_dirs', osp.splitext(osp.basename(args.config))[0], 'visualize_results')
    ic(work_dir)
    mmcv.mkdir_or_exist(osp.abspath(work_dir))

    config = mmcv.Config.fromfile(args.config)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    pose_model = init_pose_model(args.config, checkpoint=args.checkpoint, device=device)
    dataset = config.data['test']['type']
    dataset_info = DatasetInfo(config.dataset_info)

    dataset = build_dataset(config.data.test)
    num_cams = config.data_cfg['num_cameras']
    for i in range(20):
        data = dataset.__getitem__(i)
        ic(i, data.keys())
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


        f, axarr = plt.subplots(nrows=2, ncols=7, figsize=[140, 60])
        axarr[0, 0].imshow(img, interpolation='nearest')

        # plot the 2d predictions
        rr = np.array(data['img_metas'].data['resize_ratio'])
        offset = np.array(data['img_metas'].data['bbox_offset'])  # [num_cams, 2]
        kpt_2d_pred_original_img = kpt_2d_pred
        kpt_2d_pred_original_img[..., :-1] = kpt_2d_pred[..., :-1] / rr[None, :, None, None] \
                                             + offset[None, :, None, :]  # [bs, num_cams, num_joints, 2]

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
            axarr[0, c + 1].imshow(img_vis_2d, interpolation='nearest')

        # plot the reprojected data
        kpt_2d_reproject = kpt_2d_reproject / rr[None, :, None, None] + offset[None, :, None, :]  # [bs, num_cams, num_joints, 2]
        for c in range(num_cams):
            image_file = data['img_metas'].data['image_file'][c]
            img = mmcv.imread(image_file, channel_order='rgb')
            pose_result = kpt_2d_reproject[0, c, ...]
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
            axarr[1, c + 1].imshow(img_vis_2d, interpolation='nearest')

        plt.savefig(f"{work_dir}/scene_{i}.png")


if __name__ == "__main__":
    args = parse_args()

    # args.config = "../configs/exp_configs/Triangnet_infer_pretrain_mars_to_dannce_p5.py"
    # args.checkpoint = "../work_dirs/hrnet_w48_concat_mars_p5_256x256/best_AP_epoch_5.pth"

    # args.config = "../configs/exp_configs/Triangnet_mars_to_dannce_p5_un3d_epoch.py"
    # args.checkpoint = "best_MPJPE_epoch_5.pth"

    "========================================"
    """experiment mars to dannce p9"""
    "========================================"

    # args.config = "../configs/exp_configs/Triangnet_mars_to_dannce_p9_un3d_epoch.py"
    # args.checkpoint = "work_dirs/Triangnet_mars_to_dannce_p9_un3d_epoch/best_MPJPE_epoch_40.pth"

    ##### DirectTriang #####
    # args.config = "../configs/exp_configs/DirectTriang_mars_to_dannce_p9.py"
    # args.checkpoint = "../work_dirs/hrnet_w48_mars_p9_256x256/best_AP_epoch_95.pth"

    # args.config = "../configs/exp_configs/DirectTriang_test_dannce_p9.py"
    # args.checkpoint = "../work_dirs/hrnet_w48_dannce_p9_256x256/best_AP_epoch_200.pth"

    # args.config = "../configs/exp_configs/CDTriangnet_mars_to_dannce_p9_infer.py"
    # args.checkpoint = "../experiments/work_dirs/CDTriangnet_mars_to_dannce_p9/best_MPJPE_epoch_49.pth"

    "========================================="
    """====experiment dannce to 1229 p12===="""
    "========================================="
    # 上界
    # args.config = "../configs/exp_configs/DirectTriang_test_1229_p12.py"
    # args.checkpoint = "../work_dirs/hrnet_w48_mouse1229_2d_p12_256x256/best_AP_epoch_300.pth"
    # 下界
    # args.config = "../configs/exp_configs/DirectTriang_test_1229_p12.py"
    # args.checkpoint = "../work_dirs/hrnet_w48_dannce_2d_p12_256x256/best_AP_epoch_100.pth"

    # DATriangnet
    # args.config = "../configs/exp_configs/DATriangnet_dannce_to_1229_p12_infer.py"
    # args.checkpoint = "../experiments/work_dirs/DATriangnet_dannce_to_1229_p12/best_MPJPE_epoch_120.pth"

    args.config = "../configs/exp_configs/human/DirectTriang_test_h36m_p16_sub1.py"
    # args.checkpoint = "../experiments/work_dirs/CDTriang_train_hpii_to_h36m_sub1/best_MPJPE_epoch_3.pth"
    # args.checkpoint = "../work_dirs/hrnet_w48_mpii_256x256/best_PCKh_epoch_10.pth"
    args.checkpoint = "../work_dirs/hrnet_w48_h36m_small_p16_256x256/latest.pth"
    args.work_dir = osp.join('./work_dirs', f"{osp.splitext(osp.basename(args.config))[0]}", 'visualize_results')
    inference(args)
