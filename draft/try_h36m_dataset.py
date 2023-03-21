import mmcv
from icecream import ic

from mmpose.apis import init_pose_model
from mmpose.datasets import build_dataloader, build_dataset

if __name__ == "__main__":
    # data_root = "D:/Datasets/h36m_dataset/human3.6m_parse"
    # ann_file = f"{data_root}/annotations/Human36M_subject1_joint_2d.json"
    # cam_file = f"{data_root}/annotations/Human36M_subject1_camera.json"
    # img_prefix = f"{data_root}/images/"

    config_file = "D:/Pycharm Projects-win/mm_mouse/mmpose/configs/myconfigs/h36m.py"

    config = mmcv.Config.fromfile(config_file)
    dataset = build_dataset(config.data.train)

    checkpoint = "D:/Pycharm Projects-win/mm_mouse/mmpose/official_checkpoint/hrnet_w48_h36m_256x256-78e88d08_20210621.pth"
    model = init_pose_model(config_file, checkpoint=checkpoint, device="cpu")

    """compute the error for one epoch"""
    # dataloader = build_dataloader(dataset, samples_per_gpu=3, workers_per_gpu=2)
    # _, a = next(enumerate(dataloader))
    # ic(a.keys())
    # ic(a['img'].shape)
    # ic(len(a['img_metas'].data[0]))
    # # ic(a['proj_mat'])
    # ic(a['proj_mat'].shape)
    # # ic(a['img_metas'].data[0][1]['image_file'])
    # # ic(a['joints_4d'][1])
    # ic(a['joints_4d'].shape)
    # # ic(a['joints_3d'].shape)
    # ic(a['joints_3d'])
    # result = model.forward(a['img'], img_metas=a['img_metas'],
    #                        proj_matrices=a['proj_mat'],
    #                        return_loss=False, return_heatmap=False)
    # ic(result.keys())
    # # ic(result['preds'][1])
    # ic(result['preds'].shape)
    # mask = np.ones(result['preds'].shape[:2]) > 0
    # gt = a['joints_4d'].numpy()
    # pred = result['preds']
    # aa = np.linalg.norm(pred - gt, ord=2, axis=-1)
    # ic(aa.shape)
    # ic(aa)
    # error = keypoint_mpjpe(pred=result['preds'], gt=gt, mask=mask)
    # ic(error)

    """test dataset evaluate"""
    dataloader = build_dataloader(dataset, samples_per_gpu=5, workers_per_gpu=2)
    results = []
    for i in range(5):
        _, a = next(enumerate(dataloader))
        ic(a['proj_mat'].shape)
        result = model.forward(a['img'], img_metas=a['img_metas'],
                               proj_mat=a['proj_mat'],
                               return_loss=False, return_heatmap=False)
        ic(i, result.keys())
        results.append(result)

    evaluate_results = dataset.evaluate(results,
                                        res_folder="D:/Pycharm Projects-win/mm_mouse/mmpose/work_dirs/temp",
                                        metric='mpjpe'
                                        )

    ic(evaluate_results)
