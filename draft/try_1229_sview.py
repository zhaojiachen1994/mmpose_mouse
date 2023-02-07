import mmcv
from icecream import ic

from mmpose.datasets import build_dataloader, build_dataset

if __name__ == "__main__":
    print("===")
    config_file = "../configs/mouse/dataset_1229_sview.py"
    config = mmcv.Config.fromfile(config_file)
    dataset = build_dataset(config.data.train)
    ic(len(dataset))
    a = dataset.__getitem__(4)
    dataloader = build_dataloader(dataset, samples_per_gpu=2, workers_per_gpu=2)
    _, a = next(enumerate(dataloader))
    ic(a.keys())
