import mmcv
import torch
from icecream import ic

from mmpose.apis import init_pose_model
from mmpose.datasets import build_dataloader, build_dataset

if __name__ == "__main__":
    ic("====")
    config_file = "../configs/mouse/cd_mars_to_dannce_p9.py"
    config = mmcv.Config.fromfile(config_file)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    checkpoint = "../work_dirs/hrnet_w48_mars_p9_256x256/best_AP_epoch_95.pth"
    model = init_pose_model(config_file, checkpoint=checkpoint, device="cpu")

    dataset = build_dataset(config.data.test)
    dataloader = build_dataloader(dataset, samples_per_gpu=2, workers_per_gpu=2)
    _, a = next(enumerate(dataloader))
    ic(a.keys())

    # losses = model.forward(**a, return_loss=True)
    # ic(losses.keys())

    result = model.forward(**a, return_loss=False)
    ic(result.keys())