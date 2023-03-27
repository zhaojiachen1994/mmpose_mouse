import warnings
from argparse import ArgumentParser

import mmcv
from icecream import ic

warnings.filterwarnings("ignore")

from mmpose.datasets import DatasetInfo
from mmpose.datasets import build_dataset, build_dataloader
from mmpose.apis import (inference_top_down_pose_model, init_pose_model,
                         vis_pose_result)

has_mmdet = True


def try_mars_top_p5():
    config_file = "../configs/mouse/dataset_mars_p5_top.py"
    config = mmcv.Config.fromfile(config_file)
    dataset_info = DatasetInfo(mmcv.Config.fromfile(config_file)._cfg_dict['dataset_info'])
    dataset = build_dataset(config.data.train)
    i = 0
    data = dataset.__getitem__(i)
    ic(data.keys())
    ic(data['target'].shape)
    ic(data['img_metas'].data)


def try_concat_p5_dataset():
    config_file = "../configs/mouse/concat_mars_dataset.py"
    config = mmcv.Config.fromfile(config_file)
    dataset = build_dataset(config.data.train)
    dataloader = build_dataloader(dataset, samples_per_gpu=10, workers_per_gpu=2)
    _, a = next(enumerate(dataloader))
    ic(a['img_metas'].data)
    ic(a.keys())
    # ic(len(dataset))
    # i = 0
    # data = dataset.__getitem__(i)
    # ic(data.keys())


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


def visualize_p5_top_results():
    args = parse_args()
    # det_config = "D:/Pycharm Projects-win/mm_mouse/mmdetection/mouse_configs/faster_rcnn_r50_fpn_1x_mars_black.py"
    # det_checkpoint = "D:/Pycharm Projects-win/mm_mouse/mmdetection/work_dirs/faster_rcnn_r50_fpn_1x_mars_black/epoch_3.pth"
    #
    pose_config = "../configs/mouse/dataset_mars_p5_top.py"
    pose_checkpoint = "../work_dirs/dataset_mars_p5/epoch_40.pth"
    #
    # img_file = "D:/Datasets/MARS-PoseAnnotationData/raw_images_top/MARS_top_00001.jpg"
    #
    # det_model = init_detector(det_config, det_checkpoint, device=args.device.lower())
    # pose_model = init_pose_model(pose_config, pose_checkpoint, device=args.device.lower())
    # dataset = pose_model.cfg.data['test']['type']
    # dataset_info = pose_model.cfg.data['test'].get('dataset_info', None)
    # mmdet_results = inference_detector(det_model, img_file)
    # ic(mmdet_results)
    #
    # # keep the person class bounding boxes.
    # mouse_results = process_mmdet_results(mmdet_results, args.det_cat_id)
    # ic(mouse_results)

    config = mmcv.Config.fromfile(pose_config)
    dataset_info = DatasetInfo(config._cfg_dict['dataset_info'])
    pose_model = init_pose_model(pose_config, checkpoint=pose_checkpoint, device="cpu")
    dataset = config.data['test']['type']
    dataset_info = DatasetInfo(config.dataset_info)
    ic(dataset_info)

    dataset = build_dataset(config.data.test)
    i = 0
    aa = dataset.__getitem__(i)
    image_file = aa['img_metas'].data['image_file']
    out_file = f"../work_dirs/temp/MARS_result{i}.jpg"
    mouse_result = [{'bbox': aa['img_metas'].data['bbox']}]
    ic(aa.keys())
    ic(aa['img_metas'].data)
    pose_results, _ = inference_top_down_pose_model(pose_model,
                                                    image_file,
                                                    mouse_result,
                                                    dataset=config.data['test']['type'],
                                                    dataset_info=dataset_info
                                                    )
    ic(pose_results)
    vis_pose_result(
        pose_model,
        image_file,
        pose_results,
        dataset=config.data['test']['type'],
        dataset_info=dataset_info,
        kpt_score_thr=args.kpt_thr,
        radius=args.radius,
        thickness=args.thickness,
        show=args.show,
        out_file=out_file)


def demo_mars2dannce_2d():
    args = parse_args()
    pose_config = "../configs/mouse/hrnet_w48_mars_to_dannce_2d_p5.py"
    pose_checkpoint = "../work_dirs/hrnet_w48_concat_mars_p5_256x256/best_AP_epoch_5.pth"

    config = mmcv.Config.fromfile(pose_config)
    dataset_info = DatasetInfo(config._cfg_dict['dataset_info'])
    pose_model = init_pose_model(pose_config, checkpoint=pose_checkpoint, device="cpu")
    dataset = config.data['test']['type']
    dataset_info = DatasetInfo(config.dataset_info)
    ic(dataset_info)

    dataset = build_dataset(config.data.test)
    for i in range(60):
        aa = dataset.__getitem__(i)
        image_file = aa['img_metas'].data['image_file']
        out_file = f"../work_dirs/temp/MARS_dannce_result{i}.jpg"
        mouse_result = [{'bbox': aa['img_metas'].data['bbox']}]
        ic(aa.keys())
        ic(aa['img_metas'].data)
        pose_results, _ = inference_top_down_pose_model(pose_model,
                                                        image_file,
                                                        mouse_result,
                                                        dataset=config.data['test']['type'],
                                                        dataset_info=dataset_info
                                                        )
        ic(pose_results)
        vis_pose_result(
            pose_model,
            image_file,
            pose_results,
            dataset=config.data['test']['type'],
            dataset_info=dataset_info,
            kpt_score_thr=args.kpt_thr,
            radius=args.radius,
            thickness=args.thickness,
            show=args.show,
            out_file=out_file)


if __name__ == "__main__":
    ic("====")
    # try_concat_p5_dataset()
    # visualize_p5_top_results()
    demo_mars2dannce_2d()
