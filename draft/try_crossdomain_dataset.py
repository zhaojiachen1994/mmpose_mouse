import mmcv
from icecream import ic

from mmpose.datasets import build_dataset

if __name__ == "__main__":
    ic("===")
    config_file = "../configs/mouse/crossdomain_dannce_to_1229.py"
    # config_file = "../configs/mouse/TriangNet_w48_dannce3d_256x256.py"
    config = mmcv.Config.fromfile(config_file)
    ic(config)
    dataset = build_dataset(config.data.train)
    i = 1
    a = dataset.__getitem__(i)
    ic(a['source_data'].keys())
    ic(a['target_data'].keys())
