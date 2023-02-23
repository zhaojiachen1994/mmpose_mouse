# my h36m dataset class

import copy
import json
import os.path as osp

import numpy as np
from icecream import ic
from mmcv import Config
from xtcocotools.coco import COCO

from mmpose.datasets.builder import DATASETS
from mmpose.datasets.datasets.base import Kpt3dMviewRgbImgDirectDataset


@DATASETS.register_module()
class Body3DH36MMviewDataset(Kpt3dMviewRgbImgDirectDataset):
    def __init__(self,
                 ann_file,
                 ann_3d_file,
                 cam_file,
                 img_prefix,
                 data_cfg,
                 pipeline=None,
                 dataset_info=None,
                 test_mode=False,
                 coco_style=True
                 ):
        if dataset_info is None:
            cfg = Config.fromfile("configs/_base_/datasets/h36m.py")
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
        self.num_cameras = 4
        self.coco = COCO(ann_file)
        self.img_ids = self.coco.getImgIds()
        self.num_images = len(self.img_ids)
        self.id2name, self.name2id = self._get_mapping_id_name(
            self.coco.imgs)
        self.db = self._get_db()
        self.cams_params = self._get_cam(cam_file)
        self.joints_global = self._get_joints_3d(ann_3d_file)

    def _get_db(self):
        """get the database"""
        gt_db = []
        for img_id in self.img_ids[:20]:
            img_ann = self.coco.loadImgs(img_id)[0]
            width = img_ann['width']
            height = img_ann['height']
            num_joints = self.ann_info['num_joints']
            ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=False)
            ic(img_ann)
            objs = self.coco.loadAnns(ann_ids)

            # only a person in h36m
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
            joints_3d[:, :2] = keypoints[:, :2]
            joints_3d_visible[:, :2] = np.minimum(1, keypoints[:, 2:3])
            image_file = osp.join(self.img_prefix, self.id2name[img_id])

            rec = {
                'image_file': image_file,
                'joints_3d': joints_3d,
                'joints_3d_visible': joints_3d_visible,
                'bbox': obj['clean_bbox'][:4],
                'rotation': 0,
                'dataset': self.dataset_name,
                'bbox_score': 1,
                'bbox_id': 0,
                'subject': img_ann['subject'],
                'action_idx': img_ann['action_idx'],
                'subaction_idx': img_ann['subaction_idx'],
                'cam_idx': img_ann['cam_idx'],
                'frame_idx': img_ann['frame_idx']
            }
            gt_db.append(rec)
        return gt_db

    def _get_cam(self, calib):
        with open(calib, 'r') as f:
            cameras_params = json.load(f)
        return cameras_params

    def _get_joints_3d(self, ann_3d_file):
        """load the ground truth 3d keypoint, annoted as 4d in outer space"""
        with open(ann_3d_file, 'rb') as f:
            data = json.load(f)
        return data

    def __getitem__(self, idx):
        """Get the sample by a given index"""
        results = {}
        for c in range(self.num_cameras):
            result = copy.deepcopy(self.db[self.num_cameras * idx + c])
            result['ann_info'] = self.ann_info
            result['cam_params'] = self.cams_params[str(result['cam_idx'])]
            # get the global 3d joint
            result['joints_4d'] = \
                self.joints_global[str(result['action_idx'])][str(result['subaction_idx'])][str(result['frame_idx'])]
            results[c] = result
        return self.pipeline(results)

    def evaluate(self, results, *args, **kwargs):
        pass
