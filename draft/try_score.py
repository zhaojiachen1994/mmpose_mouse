import mmcv
import torch

from mmpose.datasets import DatasetInfo
from mmpose.datasets import build_dataset

if __name__ == "__main__":
    """triangnet with score head,"""
    config_file = "D:/Pycharm Projects-win/mm_mouse/mmpose/configs/mouse/try_score_head.py"
    # results_path = "D:/Pycharm Projects-win/mm_mouse/mmpose/work_dirs/hrnet_w48_mouse_1229_256x256/results/try_1229_mview"
    config = mmcv.Config.fromfile(config_file)
    dataset_info = DatasetInfo(config._cfg_dict['dataset_info'])
    dataset = build_dataset(config.data.train)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #
    # checkpoint = "D:/Pycharm Projects-win/mm_mouse/mmpose/work_dirs/hrnet_w48_mouse_1229_256x256/best_AP_epoch_90.pth"
    # model = init_pose_model(config_file, checkpoint=checkpoint, device=device)
    # num_cams = 6
