import copy
import json
import os.path as osp

import numpy as np
from mmcv import Config
from xtcocotools.coco import COCO

from mmpose.datasets.builder import DATASETS
from mmpose.datasets.datasets.base import Kpt3dMviewRgbImgDirectDataset


@DATASETS.register_module()
class Mouse12293dDatasetMview(Kpt3dMviewRgbImgDirectDataset):
    # metric
    ALLOWED_METRICS = {'mpjpe', 'p-mpjpe', 'n-mpjpe'}

    def __init__(self,
                 ann_file,
                 ann_3d_file,
                 cam_file,
                 img_prefix,
                 data_cfg,
                 pipeline=None,
                 dataset_info=None,
                 test_mode=False,
                 ):
        if dataset_info is None:
            cfg = Config.fromfile("configs/_base_/mouse_datasets/mouse_one_1229.py")
            dataset_info = cfg._cfg_dict['dataset_info']
        super().__init__(
            ann_file,
            img_prefix,
            data_cfg,
            pipeline,
            dataset_info=dataset_info,
            test_mode=test_mode,
        )
        self.ann_3d_file = ann_3d_file
        self.ann_info['use_different_joint_weights'] = False
        self.use_nms = data_cfg.get('use_nms', True)
        self.soft_nms = data_cfg['soft_nms']
        self.nms_thr = data_cfg['nms_thr']
        self.oks_thr = data_cfg['oks_thr']
        self.vis_thr = data_cfg['vis_thr']
        self.num_joints = data_cfg['num_joints']
        self.data_cfg = data_cfg
        self.num_cameras = data_cfg['num_cameras']
        self.coco = COCO(ann_file)
        self.img_ids = self.coco.getImgIds()
        self.num_images = len(self.img_ids)
        self.id2name, self.name2id = self._get_mapping_id_name(
            self.coco.imgs)
        self.db = self._get_db(data_cfg)
        self.cams_params = self._get_cam_params(cam_file)
        # get the 3d keypoint ground truth, here written as joints_4d to match the mmpose denotation
        self.joints_4d, self.joints_4d_visible, _ = self._get_joints_3d(self.ann_3d_file, data_cfg)

    def __len__(self):
        return int(self.num_images / 4)

    def _get_db(self, data_cfg):
        """get the database"""
        gt_db = []
        for img_id in self.img_ids:
            img_ann = self.coco.loadImgs(img_id)[0]
            width = img_ann['width']
            height = img_ann['height']
            num_joints = self.ann_info['num_joints']
            ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=False)
            objs = self.coco.loadAnns(ann_ids)

            # only one mouse in this dataset
            obj = objs[0]
            x, y, w, h = obj['bbox']
            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(width - 1, x1 + max(0, w))
            y2 = min(height - 1, y1 + max(0, h))
            if ('area' not in obj or obj['area'] > 0) and x2 > x1 and y2 > y1:
                obj['clean_bbox'] = [x1, y1, x2 - x1, y2 - y1]

            # get the 2d keypoint ground truth, here written as joints_3d to match the mmpose denotation
            joints_3d = np.zeros((num_joints, 3), dtype=np.float32)
            joints_3d_visible = np.zeros((num_joints, 3), dtype=np.float32)
            keypoints = np.array(obj['keypoints']).reshape(-1, 3)
            joints_3d[:, :2] = keypoints[data_cfg['dataset_channel'], :2]
            joints_3d_visible[:, :2] = np.minimum(1, keypoints[data_cfg['dataset_channel'], 2:3])
            image_file = osp.join(self.img_prefix, self.id2name[img_id])

            rec = {
                'image_file': image_file,
                'cam_idx': image_file[-16:-12],
                'joints_3d': joints_3d,
                'joints_3d_visible': joints_3d_visible,
                'bbox': obj['clean_bbox'][:4],
                'rotation': 0,
                'dataset': self.dataset_name,
                'bbox_score': 1,
                'bbox_id': 0,
            }
            gt_db.append(rec)
        return gt_db

    def _get_cam_params(self, calib):
        with open(calib, 'r') as f:
            cameras_params = json.load(f)
        new_cameras_params = {}
        for k in cameras_params.keys():
            new_cameras_params[k] = {}
            new_cameras_params[k]['K'] = np.array(cameras_params[k]['K']).reshape([3, 3])
            new_cameras_params[k]['R'] = np.array(cameras_params[k]['R'])
            new_cameras_params[k]['T'] = np.array(cameras_params[k]['T'])
        return new_cameras_params

    def _get_joints_3d(self, ann_3d_file, data_cfg):
        """load the ground truth 3d keypoint, annoted as 4d in outer space"""
        with open(ann_3d_file, 'rb') as f:
            data = json.load(f)
        data = np.array(data['joint_3d'])
        data = data[:, data_cfg['dataset_channel'], :]

        [num_sample, num_joints, _] = data.shape
        data[np.isnan(data)] = 0.0

        # joints_3d
        joints_3d = np.zeros_like(data, dtype=np.float32)
        joints_3d[:] = data[:]

        # joints_3d_visible
        joints_3d_visible = np.ones_like(data, dtype=np.float32)
        joints_3d_visible[joints_3d == 0] = 0.0
        joints_3d_visible = joints_3d_visible.reshape([num_sample, num_joints, 3])

        roots_3d = data[:, 5, :]  # body_middle as root here
        return joints_3d, joints_3d_visible, roots_3d

    def __getitem__(self, idx):
        """Get the sample by a given index"""
        results = {}
        for c in range(self.num_cameras):
            result = copy.deepcopy(self.db[self.num_cameras * idx + c])
            result['ann_info'] = self.ann_info
            result['camera_0'] = self.cams_params[result['cam_idx']]  # the original camera
            result['camera'] = copy.deepcopy(self.cams_params[result['cam_idx']])  # updated camera in pipeline
            result['joints_4d'] = self.joints_4d[idx]
            result['joints_4d_visible'] = self.joints_4d_visible[idx]
            results[c] = result
        return self.pipeline(results)

    def evaluate(self, results, *args, **kwargs):
        pass
