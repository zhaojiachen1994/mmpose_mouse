import mmcv
import numpy as np
import torch
from PIL import Image
from icecream import ic

from mmpose.apis import init_pose_model
from mmpose.core import imshow_multiview_keypoints_3d
from mmpose.datasets import DatasetInfo
from mmpose.datasets import build_dataset

if __name__ == "__main__":
    config_file = "D:/Pycharm Projects-win/mm_mouse/mmpose/configs/mouse/dataset_1229_3d_mview.py"
    results_path = "D:/Pycharm Projects-win/mm_mouse/mmpose/work_dirs/hrnet_w48_mouse_1229_256x256/results/try_1229_mview"
    config = mmcv.Config.fromfile(config_file)
    dataset_info = DatasetInfo(config._cfg_dict['dataset_info'])
    dataset = build_dataset(config.data.train)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    checkpoint = "D:/Pycharm Projects-win/mm_mouse/mmpose/work_dirs/hrnet_w48_mouse_1229_256x256/latest.pth"
    model = init_pose_model(config_file, checkpoint=checkpoint, device=device)
    num_cams = 6
    res_thr = 10

    for i in range(20):
        data = dataset.__getitem__(i)
        imgs = data['img']
        proj_matrices = data['proj_mat']
        imgs = torch.unsqueeze(imgs, 0)
        proj_matrices = torch.from_numpy(np.expand_dims(proj_matrices, 0))
        imgs = imgs.to(device)
        proj_matrices = proj_matrices.to(device)
        result = model.forward(imgs,
                               proj_matrices=proj_matrices,
                               img_metas=None,
                               return_loss=False,
                               return_heatmap=False)
        kpt_3d_pred = result['preds']  # [bs, num_joints, 3]
        kpt_2d_pred = result['kp_2d_preds']  # [bs, cams, 16, 3]
        res_triang = result['res_triang']  # [bs, num_joints]
        kpt_3d_score = np.array(res_triang < res_thr, dtype=float)
        kpt_3d_score = np.expand_dims(kpt_3d_score, -1)  # [bs, num_joints, 1]
        ic(res_triang.shape)
        # kpt_3d_reset = np.concatenate([kpt_3d_reset, scores.reshape([-1, 1])], axis=-1)

        """plot the 3d results"""
        kpt_3d_pred = np.concatenate([kpt_3d_pred, kpt_3d_score], axis=-1)
        ic(kpt_2d_pred.shape)
        img = imshow_multiview_keypoints_3d(
            [kpt_3d_pred[0]],
            skeleton=dataset_info.skeleton,
            pose_kpt_color=dataset_info.pose_kpt_color,
            pose_link_color=dataset_info.pose_link_color,
            space_size=[0.2, 0.2, 0.2],
            space_center=[0, 0, 0.1],
            kpt_score_thr=0.5,
        )
        # plt.imshow(img, interpolation='nearest')
        # plt.show()
        im = Image.fromarray(img)
        im.save(f"{results_path}/scene{i}_3d.jpeg")

        # plot the 2d predictions
        # rr = np.array(data['img_metas'].data['resize_ratio'])
        # offset = np.array(data['img_metas'].data['bbox_offset'])  # [num_cams, 2]
        # kpt_2d_pred_original_img = kpt_2d_pred
        # kpt_2d_pred_original_img[..., :-1] = kpt_2d_pred[..., :-1] / rr[None, :, None, None] \
        #                                      + offset[None, :, None, :]  # [bs, num_cams, num_joints, 2]

        # for c in range(num_cams):
        #     image_file = data['img_metas'].data['image_file'][c]
        #     img = mmcv.imread(image_file, channel_order='rgb')
        #     pose_result = kpt_2d_pred_original_img[0, c, ...]
        #     # ic(pose_result)
        #     if pose_result.shape[1] == 2:
        #         pose_result = np.concatenate([pose_result, np.ones([pose_result.shape[0], 1])], axis=1)
        #     img_vis_2d = imshow_keypoints(
        #         image_file,
        #         [pose_result],
        #         skeleton=dataset_info.skeleton,
        #         kpt_score_thr=0.9,
        #         pose_kpt_color=dataset_info.pose_kpt_color,
        #         pose_link_color=dataset_info.pose_link_color,
        #     )
        #     img_vis_2d = Image.fromarray(img_vis_2d[:, :, ::-1])
        #     img_vis_2d.save(f"{results_path}/scene{i}_cam{c}.jpeg")
