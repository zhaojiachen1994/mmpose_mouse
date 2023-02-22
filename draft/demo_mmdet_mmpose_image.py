# demo/top_down_img_demo_with_mmdet.py

import warnings
from argparse import ArgumentParser

from icecream import ic

from mmpose.apis import (inference_top_down_pose_model, init_pose_model,
                         process_mmdet_results, vis_pose_result)
from mmpose.datasets import DatasetInfo

try:
    from mmdet.apis import inference_detector, init_detector

    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False


def parse_args():
    parser = ArgumentParser()
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
        '--kpt-thr', type=float, default=0.6, help='Keypoint score threshold')
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


def main():
    args = parse_args()
    det_config = "D:/Pycharm Projects-win/mm_mouse/mmdetection/work_dirs/" \
                 "faster_rcnn_r50_fpn_1x_1229/faster_rcnn_r50_fpn_1x_1229.py"
    det_checkpoint = "D:/Pycharm Projects-win/mm_mouse/mmdetection/work_dirs/" \
                     "faster_rcnn_r50_fpn_1x_1229/latest.pth"
    pose_config = "D:/Pycharm Projects-win/mm_mouse/mmpose/configs/mouse/" \
                  "hrnet_w48_mouse_1229_256x256.py"
    pose_checkpoint = "D:/Pycharm Projects-win/mm_mouse/mmpose/work_dirs/hrnet_w48_mouse_1229_256x256/" \
                      "best_AP_epoch_90.pth"  # "latest.pth"

    img_file = "D:/Datasets/transfer_mouse/onemouse1229/20221229-1-cam0/img0125.png"
    # img_file = "D:/Datasets/transfer_mouse/dannce_20230130/images_gray/mouse0_003118_0.png"
    out_file = "D:/Pycharm Projects-win/mm_mouse/mmpose/work_dirs/hrnet_w48_mouse_1229_256x256/results/a.png"

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

    # test a single image, the resulting box is (x1, y1, x2, y2)
    mmdet_results = inference_detector(det_model, img_file)
    ic(mmdet_results)

    # keep the person class bounding boxes.
    mouse_results = process_mmdet_results(mmdet_results, args.det_cat_id)
    ic(mouse_results)

    # test a single image, with a list of bboxes.

    # optional
    return_heatmap = False

    # e.g. use ('backbone', ) to return backbone feature
    output_layer_names = None

    pose_results, returned_outputs = inference_top_down_pose_model(
        pose_model,
        img_file,
        mouse_results,
        bbox_thr=args.bbox_thr,
        format='xyxy',
        dataset=dataset,
        dataset_info=dataset_info,
        return_heatmap=return_heatmap,
        outputs=output_layer_names)

    # ic(pose_results[0]['keypoints'][:4])
    # pose_results[0]['keypoints'] = pose_results[0]['keypoints'][:4]
    ic(pose_results)

    # show the results
    vis_pose_result(
        pose_model,
        img_file,
        pose_results,
        dataset=dataset,
        dataset_info=dataset_info,
        kpt_score_thr=args.kpt_thr,
        radius=args.radius,
        thickness=args.thickness,
        show=args.show,
        out_file=out_file)


if __name__ == '__main__':
    main()
