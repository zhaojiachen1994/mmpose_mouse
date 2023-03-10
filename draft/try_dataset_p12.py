import warnings

import mmcv
import torch
from icecream import ic

warnings.filterwarnings("ignore")

from mmpose.apis import init_pose_model
from mmpose.datasets import DatasetInfo
from mmpose.datasets import build_dataset, build_dataloader


def try_dataset_p12():
    config_file = "D:/Pycharm Projects-win/mm_mouse/mmpose/configs/mouse/dataset_dannce_3d_p12.py"
    config = mmcv.Config.fromfile(config_file)
    dataset_info = DatasetInfo(mmcv.Config.fromfile(config_file)._cfg_dict['dataset_info'])
    ic(dataset_info.__dir__())
    ic(len(dataset_info.pose_kpt_color))
    dataset = build_dataset(config.data.train)
    i = 0
    data = dataset.__getitem__(i)
    ic(data.keys())
    ic(data['joints_4d'])
    ic(data['joints_4d'].shape)


def try_3dDatato2Dmodel():
    config_file = "D:/Pycharm Projects-win/mm_mouse/mmpose/configs/mouse/hrnet_w48_dannce_3d_p12_256x256.py"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    config = mmcv.Config.fromfile(config_file)
    ic(config.keys())
    ic(config.dataset_info.keys())
    ic(config)
    dataset = build_dataset(config.data.test)
    ic(config.keys())
    i = 0
    # data = dataset.__getitem__(i)
    # ic(data.keys())
    # ic(data['img'].shape)
    # ic(data['img_metas'].data['image_file'])

    dataloader = build_dataloader(dataset, samples_per_gpu=2, workers_per_gpu=2)
    _, a = next(enumerate(dataloader))
    ic(a.keys())
    ic(a['img'].shape)
    # ic(a['img_metas'].data[0][0]['joints_3d_visible'])
    ic(a['img_metas'])
    ic(len(a['img_metas'].data[0]))
    ic(a['img_metas'].data[0][0].keys())
    ic(a['img_metas'].data[0][0]['center'])

    model = init_pose_model(config_file)
    # losses = model.forward(img=a['img'].to(device),
    #                        target=a['target'].to(device),
    #                        target_weight=a['target_weight'].to(device),
    #                        img_metas=a['img_metas'],
    #                        return_loss=True,
    #                        )
    # ic(losses)

    losses = model.forward(img=a['img'].to(device),
                           target=None,
                           target_weight=None,
                           img_metas=a['img_metas'],
                           return_loss=False,
                           )


if __name__ == "__main__":
    print("====")
    config_file = "D:/Pycharm Projects-win/mm_mouse/mmpose/configs/mouse/hrnet_w48_dannce_2d_p12_256x256.py"
    results_path = "D:/Pycharm Projects-win/mm_mouse/mmpose/work_dirs/temp"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    config = mmcv.Config.fromfile(config_file)
    dataset_info = DatasetInfo(config._cfg_dict['dataset_info'])
    dataset = build_dataset(config.data.train)
    dataloader = build_dataloader(dataset, samples_per_gpu=1, workers_per_gpu=1)
    checkpoint = "D:/Pycharm Projects-win/mmpose/work_dirs/hrnet_gray" \
                 "/hrnet_w48_mouse_dannce_256x256/best_AP_epoch_190.pth"
    model = init_pose_model(config_file, checkpoint=checkpoint, device=device)

    """dataset getitem"""
    for i in range(1):
        a = dataset.__getitem__(i)
        img = torch.unsqueeze(a['img'], 0)
        ic(a.keys())
        ic(a['img_metas'].data['joints_3d'])
        ic(a['img_metas'].data['image_file'])

        result = model.forward(img.to(device),
                               target=None,
                               target_weight=None,
                               img_metas=[a['img_metas'].data],
                               return_loss=False,
                               return_heatmap=False
                               )

    """dataloader"""
    # results = []
    # for i in range(2):
    #     _, a = next(enumerate(dataloader))
    #     result = model.forward(img=a['img'].to(device),
    #                            target=None,
    #                            target_weight=None,
    #                            img_metas=a['img_metas'].data[0],
    #                            return_loss=False,
    #                            return_heatmap=False
    #                            )
    #     ic(i, result.keys(), result['image_paths'], result['preds'], result['preds'].shape)
    #     img = imshow_keypoints(
    #         result['image_paths'][0],
    #         [result['preds'][0]],
    #         skeleton=dataset_info.skeleton,
    #         kpt_score_thr=0.6,
    #         pose_kpt_color=dataset_info.pose_kpt_color,
    #         pose_link_color=dataset_info.pose_link_color,
    #
    #     )
    #     ic(img.shape)
    #     img_vis_2d = Image.fromarray(img)
    #     img_vis_2d.save(f"{results_path}/{i}.jpeg")
    #
    #     results.append(result)
    #
    # evaluate_results = dataset.evaluate(results)
    # ic(evaluate_results)
