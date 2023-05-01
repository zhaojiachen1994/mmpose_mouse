import json
import os.path as osp

import numpy as np
import pandas as pd
from icecream import ic
from mmcv import Config

from mmpose.core.evaluation import keypoint_mpjpe


def main(config_file):
    cfg = Config.fromfile(config_file)
    work_dir = osp.join('./work_dirs', osp.splitext(osp.basename(config_file))[0])
    result_file = f"{work_dir}/result_keypoints.json"
    with open(result_file, 'r') as f:
        keypoint_results = json.load(f)

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
    preds = np.stack(preds)  # [num_samples, num_joints, 3]
    gts = np.stack(gts)  # [num_samples, ]
    masks = np.stack(masks) > 0  # [num_samples, num_joints]
    ic(preds.shape)
    ic(gts.shape)
    ic(masks.shape)

    matrics = []
    for j in range(preds.shape[1]):
        pred_joint = preds[:, j:j + 1, :]
        gt_joint = gts[:, j:j + 1, :]
        # mask_joint = masks[:, j:j + 1]   # for THmouse
        mask_joint = masks[:, j:j + 1, 0]  # for dannce
        # mask_joint = np.ones([len(gt_joint),1])>0
        ic(pred_joint.shape)
        ic(gt_joint.shape)
        ic(mask_joint.shape)
        matrics.append(keypoint_mpjpe(pred_joint, gt_joint, mask_joint))
    matrics.append(np.mean(matrics))
    matrics = np.around(np.array(matrics), 2)
    np.savetxt(f"{work_dir}/matrics.txt", matrics, fmt='%.02f')  # use exponential notation

    matrics_file = f"{work_dir}/matrics.csv"
    df = pd.DataFrame(data={"col1": matrics})
    df.to_csv(matrics_file, sep=',', index=False)


if __name__ == '__main__':
    "========================================"
    "======experiment mars to dannce p9======"
    "========================================"
    # config_files = [
    #     "../configs/exp_configs/mouse/test_calms2dannce_hrnet_p9.py",
    #     "../configs/exp_configs/mouse/test_dannce2dannce_hrnet_p9.py",
    #     "../configs/exp_configs/mouse/test_calms2dannce_hrnet_score_p9.py",
    #     "../configs/exp_configs/mouse/test_calms2dannce_hrnet_score_dd_p9.py"
    # ]
    # for file in config_files:
    #     main(file)

    "========================================"
    "======experiment mars to thmouse p9======"
    "========================================"
    # config_files = [
    #     "../configs/exp_configs/mouse/test_calms2thmouse_hrnet_p9.py",
    #     "../configs/exp_configs/mouse/test_thmouse2thmouse_hrnet_p9.py",
    #     "../configs/exp_configs/mouse/test_calms2thmouse_hrnet_score_p9.py",
    #     "../configs/exp_configs/mouse/test_calms2thmouse_hrnet_score_dd_p9.py"
    # ]
    # for file in config_files:
    #     main(file)

    "========================================"
    "====experiment thmouse to dannce p12===="
    "========================================"
    # config_files = [
    #     "../configs/exp_configs/mouse/test_thmouse2dannce_hrnet_p12.py",
    #     "../configs/exp_configs/mouse/test_dannce2dannce_hrnet_p12.py",
    #     "../configs/exp_configs/mouse/test_thmouse2dannce_hrnet_score_p12.py",
    #     "../configs/exp_configs/mouse/test_thmouse2dannce_hrnet_score_dd_p12.py"]
    # for file in config_files:
    #     main(file)

    "========================================"
    "======experiment dannce to 1229 p12====="
    "========================================"
    config_files = [
        "../configs/exp_configs/mouse/test_dannce2thmouse_hrnet_p12.py",
        # "../configs/exp_configs/mouse/test_thmouse2thmouse_hrnet_p12.py",
        # "../configs/exp_configs/mouse/test_dannce2thmouse_hrnet_score_p12.py",
        # "../configs/exp_configs/mouse/test_dannce2thmouse_hrnet_score_dd_p12.py"
    ]
    for file in config_files:
        main(file)
