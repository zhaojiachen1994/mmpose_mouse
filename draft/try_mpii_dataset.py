import warnings

import matplotlib.pyplot as plt
import mmcv
import torch
from icecream import ic

warnings.filterwarnings("ignore")
from mmpose.core import imshow_keypoints
from mmpose.datasets import DatasetInfo
from mmpose.datasets import build_dataset
from mmpose.apis import (init_pose_model)


if __name__ == "__main__":
    config_file = "../configs/human/hrnet_w48_mpii_256x256.py"
    # pose_checkpoint = "../official_checkpoint/hrnet_w48_mpii_256x256-92cab7bd_20200812.pth"
    pose_checkpoint = "../work_dirs/hrnet_w48_mpii_256x256/latest.pth"

    config = mmcv.Config.fromfile(config_file)
    dataset_info = DatasetInfo(config.dataset_info)
    pose_model = init_pose_model(config_file, checkpoint=pose_checkpoint, device="cpu")

    # dataset = build_dataset(config.data.train)
    dataset = build_dataset(config.data.test)
    i = 3
    data = dataset.__getitem__(i)
    ic(data['img_metas'].data)
    imgs = data['img']
    imgs = torch.unsqueeze(imgs, 0)
    img_metas = [data['img_metas'].data]
    image_file = data['img_metas'].data['image_file']
    # ic(aa.keys())
    # ic(aa['target'].shape)
    # ic(aa['img_metas'].data['image_file'])
    # ic(aa['img_metas'].data['joints_3d'])
    result = pose_model.forward(imgs, img_metas=img_metas, return_loss=False)

    ic(result.keys())
    ic(result['preds'].shape)
    img_vis_2d = imshow_keypoints(
        image_file,
        [result['preds'][0]],
        skeleton=dataset_info.skeleton,
        kpt_score_thr=0.2,
        radius=10,
        thickness=10,
        pose_kpt_color=dataset_info.pose_kpt_color,
        pose_link_color=dataset_info.pose_link_color,
    )
    ic(img_vis_2d.shape)
    plt.imshow(img_vis_2d, interpolation='nearest')
    plt.show()




