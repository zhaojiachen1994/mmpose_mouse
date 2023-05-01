import json
import os.path as osp

import numpy as np
from icecream import ic
from mmcv import Config

from mmpose.core.evaluation import keypoint_mpjpe


def main(config_file):
    cfg = Config.fromfile(config_file)
    work_dir = osp.join('./work_dirs/human', osp.splitext(osp.basename(config_file))[0])
    result_file = f"{work_dir}/result_keypoints.json"
    with open(result_file, 'r') as f:
        keypoint_results = json.load(f)
    preds = []
    gts = []
    masks = []
    for idx, result in enumerate(keypoint_results):
        pred = result['keypoints']
        gt = result['joints_4d']
        mask = np.array(result['joints_4d_visible']) > 0

        gts.append(gt)
        preds.append(pred)
        masks.append(mask)
    preds = np.stack(preds)  # [num_samples, num_joints, 3]
    gts = np.stack(gts)  # [num_samples, ]
    masks = np.stack(masks) > 0  # [num_samples, num_joints]

    errors = np.linalg.norm(preds - gts, ord=2, axis=-1)[masks]
    ic(errors)

    error = keypoint_mpjpe(preds, gts, masks)
    ic(error)

    # print(preds.shape)
    preds_pelvis = preds - preds[:, 0:1, :]
    gts_pelvis = gts - gts[:, 0:1, :]
    # print(preds_pelvis)
    error = keypoint_mpjpe(preds_pelvis, gts_pelvis, masks)
    ic(error)

    # metrics = f"{}"


if __name__ == '__main__':
    # config_file = "../configs/exp_configs/human/test_h36ms9_fully_hrnet.py"
    config_file = "../configs/exp_configs/human/test_h36ms11_fully_hrnet_score.py"
    main(config_file)
