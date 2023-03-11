import copy

from mmpose.datasets.builder import DATASETS
from .dannce_2d_dataset_sview import MouseDannce2dDatasetSview


@DATASETS.register_module()
class MouseDannce2dDatasetMview(MouseDannce2dDatasetSview):
    """multi-view 2d dannce mouse dataset
    """

    def __init__(self,
                 ann_file,
                 img_prefix,
                 data_cfg,
                 pipeline,
                 dataset_info=None,
                 test_mode=False,
                 num_cameras=6
                 ):
        if dataset_info is None:
            cfg = Config.fromfile('configs/_base_/datasets/mouse_datasets/mouse_dannce_p22.py')
            dataset_info = cfg._cfg_dict['dataset_info']

        super().__init__(
            ann_file,
            img_prefix,
            data_cfg,
            pipeline,
            dataset_info=dataset_info,
            test_mode=test_mode
        )
        self.num_cameras = num_cameras

    def __getitem__(self, idx):
        """Get the sample by a given index"""
        results = {}
        for c in range(self.num_cams):
            result = copy.deepcopy(self.db[self.num_cameras * idx + c])
            result['ann_info'] = self.ann_info
            results[c] = result
        return self.pipeline(results)
