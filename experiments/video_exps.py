import copy
import json
import os
import os.path as osp

import cv2
import mmcv
import numpy as np
import torch

from mmpose.apis import (init_pose_model)
from mmpose.core import imshow_multiview_keypoints_3d
from mmpose.datasets import DatasetInfo
from mmpose.datasets.pipelines import Compose

try:
    from mmdet.apis import inference_detector, init_detector
    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False

# from moviepy.editor import VideoFileClip
# clip = VideoFileClip("sample.mp4").subclip(start, end)
# clip.to_videofile(outputfile, codec="libx264", temp_audiofile='temp-audio.m4a', remove_temp=True, audio_codec='aac')

def read_cams(dataset):
    # read the camera parameters
    if dataset == "Mouse12293dDatasetMview":
        calib = f"D:/Datasets/transfer_mouse/onemouse1229/calibration_adjusted.json"
        with open(calib, 'r') as f:
            cameras_params = json.load(f)
        new_cameras_params = {}
        for i, k in enumerate(cameras_params.keys()):
            new_cameras_params[i] = {}
            new_cameras_params[i]['K'] = np.array(cameras_params[k]['K']).reshape([3, 3])
            new_cameras_params[i]['R'] = np.array(cameras_params[k]['R'])
            new_cameras_params[i]['T'] = np.array(cameras_params[k]['T'])
        return new_cameras_params


def main(det_config, det_checkpoint, pose_config,
         pose_checkpoint, video_paths,
         save_video=True, show=True, save_out_image=True):
    work_dir = osp.join('./work_dirs', osp.splitext(osp.basename(pose_config))[0], 'video_inference')
    mmcv.mkdir_or_exist(osp.abspath(work_dir))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # initializing the model
    det_model = init_detector(
        det_config, det_checkpoint, device=device)
    # build the pose model from a config file and a checkpoint file
    pose_model = init_pose_model(
        pose_config, pose_checkpoint, device=device)

    dataset = pose_model.cfg.data['test']['type']
    dataset_info = pose_model.cfg.data['test'].get('dataset_info', None)
    dataset_info = DatasetInfo(dataset_info)

    cam_params = read_cams(dataset)

    # read videos
    videos = [mmcv.VideoReader(video) for video in video_paths]

    if save_video:
        fps = videos[0].fps
        size = (640, 480)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        videoWriter = cv2.VideoWriter(os.path.join(work_dir, f'video.mp4'), fourcc, fps, size)

    for scene_id in range(100):
        scene = [next(videos[i]) for i in range(len(videos))]
        mmdet_results = inference_detector(det_model, scene)

        mmpose_inputs = dict()
        for cam_idx in range(len(videos)):
            mmpose_inputs[cam_idx] = {'img': scene[cam_idx],
                                      'bbox': mmdet_results[cam_idx][0][0][:-1].tolist(),   # in xyxy format without score
                                      'camera_0': cam_params[cam_idx],
                                      'camera': copy.deepcopy(cam_params[cam_idx]),
                                      'ann_info': {'image_size': [256, 256]}
                                      }
        test_pipeline = Compose(pose_model.cfg.video_pipeline)
        data = test_pipeline(mmpose_inputs)
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
        img = imshow_multiview_keypoints_3d(
            [kpt_3d_pred[0]],
            skeleton=dataset_info.skeleton,
            pose_kpt_color=dataset_info.pose_kpt_color,
            pose_link_color=dataset_info.pose_link_color,
            space_size=[300, 300, 300],
            space_center=[0, 0, 150],
            kpt_score_thr=0.1,
        )
        # img = cv2.resize(img, [500, 500])
        if save_video:
            videoWriter.write(cv2.resize(img, size))

        if save_out_image:
            image_step = 5
            if scene_id % image_step == 0:
                cv2.imwrite(f"{work_dir}/img{scene_id}.png", img)

        if show:
            cv2.imshow('Frame', img)

        if show and cv2.waitKey(1) & 0xFF == ord('q'):
            break

    if save_video:
        print("(((")
        videoWriter.release()
    if show:
        cv2.destroyAllWindows()


if __name__ == '__main__':
    det_config = "D:/Pycharm Projects-win/mm_mouse/mmdetection/work_dirs/" \
                 "faster_rcnn_r50_fpn_1x_1229/faster_rcnn_r50_fpn_1x_1229.py"
    det_checkpoint = "D:/Pycharm Projects-win/mm_mouse/mmdetection/work_dirs/" \
                     "faster_rcnn_r50_fpn_1x_1229/latest.pth"
    pose_config = "../configs/exp_configs/DATriangnet_dannce_to_1229_p12_infer.py"
    pose_checkpoint = "../experiments/work_dirs/DATriangnet_dannce_to_1229_p12/best_MPJPE_epoch_120.pth"
    video_paths = [f'D:/Datasets/transfer_mouse/onemouse1229/videos/20221229-1-cam{i}.mp4' for i in range(6)]


    main(det_config, det_checkpoint, pose_config, pose_checkpoint, video_paths,
         save_video=True, show=True,
         )





