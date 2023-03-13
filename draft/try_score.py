import mmcv
import numpy as np
import torch
from icecream import ic

from mmpose.apis import init_pose_model
from mmpose.datasets import DatasetInfo
from mmpose.datasets import build_dataloader, build_dataset


def try_score_forward():
    # triangnet with score head
    config_file = "D:/Pycharm Projects-win/mm_mouse/mmpose/configs/mouse/try_score_head.py"
    # results_path = "D:/Pycharm Projects-win/mm_mouse/mmpose/work_dirs/hrnet_w48_mouse_1229_256x256/results/try_1229_mview"
    config = mmcv.Config.fromfile(config_file)
    dataset_info = DatasetInfo(config._cfg_dict['dataset_info'])
    dataset = build_dataset(config.data.train)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #
    checkpoint = "D:/Pycharm Projects-win/mm_mouse/mmpose/work_dirs/hrnet_w48_dannce_2d_p12_256x256/best_AP_epoch_100.pth"
    model = init_pose_model(config_file, checkpoint=checkpoint, device=device)

    # model = init_pose_model(config_file, device=device)
    num_cams = 6

    for i in range(2):
        data = dataset.__getitem__(i)
        ic(data.keys())
        imgs = data['img']
        imgs = torch.unsqueeze(imgs, 0)
        imgs = imgs.to(device)

        proj_matrices = data['proj_mat']
        proj_matrices = torch.from_numpy(np.expand_dims(proj_matrices, 0))
        proj_matrices = proj_matrices.to(device)

        target = torch.from_numpy(data['target'])
        target = torch.unsqueeze(target, 0).to(device)

        target_weight = torch.from_numpy(data['target_weight'])
        target_weight = torch.unsqueeze(target_weight, 0).to(device)

        kpt_3d_gt = torch.from_numpy(data['joints_4d'])
        kpt_3d_gt = torch.unsqueeze(kpt_3d_gt, 0).to(device)

        result = model.forward(imgs,
                               img_metas=None,
                               proj_matrices=proj_matrices,
                               target=target,
                               target_weight=target_weight,
                               kpt_3d_gt=kpt_3d_gt,
                               return_loss=True,
                               return_heatmap=False)
        ic("Train output", result)

        result = model.forward(imgs,
                               img_metas=None,
                               proj_matrices=proj_matrices,
                               target=None,
                               target_weight=None,
                               kpt_3d_gt=None,
                               return_loss=False,
                               return_heatmap=False)
        ic("Test output", result)


if __name__ == "__main__":
    print("===")
    # triangnet with score head
    config_file = "D:/Pycharm Projects-win/mm_mouse/mmpose/configs/mouse/try_score_head.py"
    # results_path = "D:/Pycharm Projects-win/mm_mouse/mmpose/work_dirs/hrnet_w48_mouse_1229_256x256/results/try_1229_mview"
    config = mmcv.Config.fromfile(config_file)
    dataset_info = DatasetInfo(config._cfg_dict['dataset_info'])
    dataset = build_dataset(config.data.train)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    checkpoint = "D:/Pycharm Projects-win/mm_mouse/mmpose/work_dirs/hrnet_w48_dannce_2d_p12_256x256/best_AP_epoch_100.pth"
    model = init_pose_model(config_file, checkpoint=checkpoint, device=device)

    # model = init_pose_model(config_file, device=device)
    num_cams = 6
    dataloader = build_dataloader(dataset, samples_per_gpu=5, workers_per_gpu=2)

    results = []
    for i in range(5):
        _, a = next(enumerate(dataloader))
        result = model.forward()
