import warnings

import mmcv
from icecream import ic

warnings.filterwarnings("ignore")

from mmpose.datasets import DatasetInfo
from mmpose.datasets import build_dataset

if __name__ == "__main__":
    ic("====")
    config_file = "../configs/mouse/dataset_mars_p5.py"
    config = mmcv.Config.fromfile(config_file)
    dataset_info = DatasetInfo(mmcv.Config.fromfile(config_file)._cfg_dict['dataset_info'])
    dataset = build_dataset(config.data.train)
    i = 0
    data = dataset.__getitem__(i)
    ic(data.keys())
    ic(data['target'].shape)
    ic(data['img_metas'].data)
