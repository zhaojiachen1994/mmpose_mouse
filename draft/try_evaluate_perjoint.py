import json

import numpy as np
from icecream import ic

from mmpose.core.evaluation import keypoint_mpjpe

if __name__ == '__main__':
    file = "../experiments/work_dirs/Triangnet_mars_to_dannce_p9_un3d_epoch/result_keypoints.json"
    with open(file, 'r') as f:
        keypoint_results = json.load(f)
    # ic(result[0].keys())
    # idx = 0
    # pred = keypoint_results[idx]['keypoints']
    # gt = keypoint_results[idx]['joints_4d']
    # visible = keypoint_results[idx]['joints_4d_visible']

    preds = []
    gts = []
    masks = []
    for idx, result in enumerate(keypoint_results):
        pred = result['keypoints']
        gt = result['joints_4d']
        ic(result['joints_4d_visible'])
        mask = np.array(result['joints_4d_visible']) > 0

        gts.append(gt)
        preds.append(pred)
        masks.append(mask)
    preds = np.stack(preds) # [num_samples, num_joints, 3]
    gts = np.stack(gts)  # [num_samples, ]
    masks = np.stack(masks) > 0  # [num_samples, num_joints]
    ic(preds.shape)
    ic(gts.shape)
    ic(masks.shape)

    matrics = []
    for j in range(preds.shape[1]):
        pred_joint = preds[:, j:j+1, :]
        gt_joint = gts[:, j:j+1, :]
        mask_joint = masks[:, j:j+1]
        ic(pred_joint.shape)
        matrics.append(keypoint_mpjpe(pred_joint,  gt_joint, mask_joint))
    ic(matrics)
    ic(np.mean(matrics))