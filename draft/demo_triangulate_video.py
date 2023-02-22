# demo_mmdet_mmpose_video.py
# demo_triangulate_image.py


import warnings
from argparse import ArgumentParser

import cv2
import mmcv
import numpy as np
from icecream import ic

from mmpose.apis import (inference_top_down_pose_model, init_pose_model,
                         process_mmdet_results)
from mmpose.core import imshow_multiview_keypoints_3d
from mmpose.datasets import DatasetInfo

try:
    from mmdet.apis import inference_detector, init_detector

    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False

from demo_triangulate_image import triangulate_joints, read_my_calibration_full, make_projection_matrix


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        '--show',
        action='store_true',
        default=True,
        help='whether to show visualizations.')
    parser.add_argument(
        '--out-video-root',
        default='',
        help='Root of the output video file. '
             'Default not saving the visualization video.')
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
        default=0.6,
        help='Bounding box score threshold')
    parser.add_argument(
        '--kpt-thr', type=float, default=0.6, help='Keypoint score threshold')
    parser.add_argument(
        '--use-oks-tracking', action='store_true', help='Using OKS tracking')
    parser.add_argument(
        '--tracking-thr', type=float, default=0.3, help='Tracking threshold')
    parser.add_argument(
        '--euro',
        action='store_true',
        help='(Deprecated, please use --smooth and --smooth-filter-cfg) '
             'Using One_Euro_Filter for smoothing.')
    parser.add_argument(
        '--smooth',
        action='store_true',
        help='Apply a temporal filter to smooth the pose estimation results. '
             'See also --smooth-filter-cfg.')
    parser.add_argument(
        '--smooth-filter-cfg',
        type=str,
        default='configs/_base_/filters/one_euro.py',
        help='Config file of the filter to smooth the pose estimation '
             'results. See also --smooth.')
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

    parser.add_argument(
        '--use-multi-frames',
        action='store_true',
        default=False,
        help='whether to use multi frames for inference in the pose'
             'estimation stage. Default: False.')
    parser.add_argument(
        '--online',
        action='store_true',
        default=False,
        help='inference mode. If set to True, can not use future frame'
             'information when using multi frames for inference in the pose'
             'estimation stage. Default: False.')

    assert has_mmdet, 'Please install mmdet to run the demo.'

    args = parser.parse_args()
    return args


def init_models_datasets(args):
    det_config = "D:/Pycharm Projects-win/mm_mouse/mmdetection/work_dirs/" \
                 "faster_rcnn_r50_fpn_1x_1229/faster_rcnn_r50_fpn_1x_1229.py"
    det_checkpoint = "D:/Pycharm Projects-win/mm_mouse/mmdetection/work_dirs/" \
                     "faster_rcnn_r50_fpn_1x_1229/latest.pth"
    pose_config = "D:/Pycharm Projects-win/mm_mouse/mmpose/configs/mouse/" \
                  "hrnet_w48_mouse_1229_256x256.py"
    pose_checkpoint = "D:/Pycharm Projects-win/mm_mouse/mmpose/work_dirs/hrnet_w48_mouse_1229_256x256/" \
                      "best_AP_epoch_90.pth"  # "latest.pth"
    det_model = init_detector(
        det_config, det_checkpoint, device=args.device.lower())
    # build the pose model from a config file and a checkpoint file
    pose_model = init_pose_model(
        pose_config, pose_checkpoint, device=args.device.lower())
    dataset = pose_model.cfg.data['test']['type']
    dataset_info = pose_model.cfg.data['test'].get('dataset_info', None)
    if dataset_info is None:
        warnings.warn(
            'Please set `dataset_info` in the config.'
            'Check https://github.com/open-mmlab/mmpose/pull/663 for details.',
            DeprecationWarning)
    else:
        dataset_info = DatasetInfo(dataset_info)
    return det_model, pose_model, dataset, dataset_info


def det_pose_2d(det_model, pose_model, dataset, dataset_info, images):
    """
    perform triangulation on onemouse1229 data
    returns:
    # [[{'bbox': array[5,], 'keypoints': array[1, num_joint, 3]}], ...] length = num_cams
    # bbox in [x1, y1, x2, y2, score]
    # estimated keypoint coordinates in full image [num_obj, num_joint, 3], [x, y, score]
    """

    keypoint_mview = []
    # ic(images)
    #
    # mmdet_results = inference_detector(det_model, images)
    # ic(mmdet_results)
    # # mouse_results = process_mmdet_results(mmdet_results, args.det_cat_id)
    # mmdet_results = [{'bbox': mmdet_results[i][0][0]} for i in range(len(mmdet_results))]
    # ic(mmdet_results)
    for i, image in enumerate(images):
        mmdet_results = inference_detector(det_model, image)
        # ic([mmdet_results[0][0]])
        # keep the person class bounding boxes.
        mouse_results = process_mmdet_results(mmdet_results, args.det_cat_id)
        # mouse_results = [mouse_results[0]]
        pose_results, returned_outputs = inference_top_down_pose_model(
            pose_model,
            image,
            mouse_results,
            bbox_thr=args.bbox_thr,
            format='xyxy',
            dataset=dataset,
            dataset_info=dataset_info,
            return_heatmap=False,
            outputs=None)
        keypoint_mview.append(pose_results)

        # # show the results
        # vis_pose_result(
        #     pose_model,
        #     img_file,
        #     pose_results,
        #     dataset=dataset,
        #     dataset_info=dataset_info,
        #     kpt_score_thr=args.kpt_thr,
        #     radius=args.radius,
        #     thickness=args.thickness,
        #     show=args.show,
        #     out_file=f"{results_path}/{img_file[-11:-4]}-{i}.png")
    return keypoint_mview


def main(args):
    cams = ["cam0", "cam1", "cam2", "cam3", "cam4", "cam5"]
    videos_path = "D:/Datasets/transfer_mouse/onemouse1229/videos"
    video_files = [f"{videos_path}/20221229-1-{cam}.mp4" for cam in cams]
    videos = [mmcv.VideoReader(video_file) for video_file in video_files]

    det_model, pose_model, dataset, dataset_info = init_models_datasets(args)
    num_joint = dataset_info.keypoint_num

    """===========Reading calibration data=========="""
    calibration_path = f"D:/Datasets/transfer_mouse/onemouse1229/calibration_full.json"
    calibration_data = read_my_calibration_full(calibration_path)

    """===========Computing projection matrix for each camera============="""
    projection_matrices = make_projection_matrix(calibration_data, cams)

    for i_frame in range(1000):

        images = [videos[i_cam].read() for i_cam in range(len(cams))]
        keypoints_mview = det_pose_2d(det_model, pose_model, dataset, dataset_info, images)
        # ic(keypoint_mview)
        keypoints_mview = np.array([keypoints_mview[i][0]['keypoints'] for i in range(len(cams))])

        """===========triangulate all joint================"""
        keypoints_3d = triangulate_joints(keypoints_mview, projection_matrices, dataset_info, args)
        keypoints_3d = np.concatenate([keypoints_3d, np.ones([num_joint, 1])], axis=1)
        ic(i_frame, keypoints_3d.shape)

        # res = [{"keypoints_3d": keypoints_3d}]
        # vis_frame = imshow_keypoints_3d(res,
        #                           skeleton=dataset_info.skeleton,
        #                           pose_kpt_color=dataset_info.pose_kpt_color,
        #                           pose_link_color=dataset_info.pose_link_color,
        #                           vis_height=400,
        #                           axis_azimuth=70,
        #                           axis_limit=0.3,
        #                           axis_dist=10.0,
        #                           axis_elev=120,
        #                           )
        res = [keypoints_3d]
        vis_frame = imshow_multiview_keypoints_3d(
            res,
            skeleton=dataset_info.skeleton,
            pose_kpt_color=dataset_info.pose_kpt_color,
            pose_link_color=dataset_info.pose_link_color,
            space_size=[0.3, 0.3, 0.3],
            space_center=[0, 0, 0],
            kpt_score_thr=0.0,
        )
        if args.show:
            cv2.imshow('Frame', vis_frame)

        if args.show and cv2.waitKey(1) & 0xFF == ord('q'):
            break

    if args.show:
        cv2.destroyAllWindows()


if __name__ == '__main__':
    args = parse_args()
    main(args)
