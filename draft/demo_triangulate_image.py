import json
import os
import pickle
import warnings
from argparse import ArgumentParser

import numpy as np
from icecream import ic

from mmpose.apis import (inference_top_down_pose_model, init_pose_model,
                         process_mmdet_results, vis_pose_result)
from mmpose.datasets import DatasetInfo

try:
    from mmdet.apis import inference_detector, init_detector

    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        '--show',
        action='store_true',
        default=False,
        help='whether to show img')
    parser.add_argument(
        '--out-img-root',
        type=str,
        default='',
        help='root of the output img file. '
             'Default not saving the visualization images.')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--det-cat-id',
        type=int,
        default=1,
        help='Category id for bounding box detection model')
    parser.add_argument(
        '--bbox-thr',
        type=float,
        default=0.8,
        help='Bounding box score threshold')
    parser.add_argument(
        '--kpt-thr', type=float, default=0.5, help='Keypoint score threshold')
    parser.add_argument(
        '--radius',
        type=int,
        default=4,
        help='Keypoint radius for visualization')
    parser.add_argument(
        '--thickness',
        type=int,
        default=1,
        help='Link thickness for visualization')

    assert has_mmdet, 'Please install mmdet to run the demo.'

    args = parser.parse_args()
    return args


def load_images(i):
    root_path = "D:/Datasets/transfer_mouse/onemouse1229"
    file_names = [item for item in os.listdir(f"{root_path}/20221229-1-cam0") if item.endswith(".png")]
    num_images = len(file_names)
    cams = [0, 1, 2, 3, 4, 5]
    image_files = [f"{root_path}/20221229-1-cam{cam}/{file_names[i]}" for cam in cams]
    ic(image_files)
    return image_files


def init_models_datasets(args):
    det_config = "D:/Pycharm Projects-win/mm_mouse/mmdetection/work_dirs/" \
                 "faster_rcnn_r50_fpn_1x_1229/faster_rcnn_r50_fpn_1x_1229.py"
    det_checkpoint = "D:/Pycharm Projects-win/mm_mouse/mmdetection/work_dirs/" \
                     "faster_rcnn_r50_fpn_1x_1229/latest.pth"
    pose_config = "D:/Pycharm Projects-win/mm_mouse/mmpose/configs/mouse/" \
                  "hrnet_w48_mouse_1229_256x256.py"
    pose_checkpoint = "D:/Pycharm Projects-win/mm_mouse/mmpose/work_dirs/hrnet_w48_mouse_1229_256x256/" \
                      "best_AP_epoch_90.pth"  # "latest.pth"
    det_model = init_detector(
        det_config, det_checkpoint, device=args.device.lower())
    # build the pose model from a config file and a checkpoint file
    pose_model = init_pose_model(
        pose_config, pose_checkpoint, device=args.device.lower())
    dataset = pose_model.cfg.data['test']['type']
    dataset_info = pose_model.cfg.data['test'].get('dataset_info', None)
    if dataset_info is None:
        warnings.warn(
            'Please set `dataset_info` in the config.'
            'Check https://github.com/open-mmlab/mmpose/pull/663 for details.',
            DeprecationWarning)
    else:
        dataset_info = DatasetInfo(dataset_info)
    return det_model, pose_model, dataset, dataset_info


def det_pose_2d(args, image_files):
    # perform triangulation on onemouse1229 data
    # [[{'bbox': array[5,], 'keypoints': array[1, num_joint, 3]}], ...] length = num_cams
    # bbox in [x1, y1, x2, y2, score]
    # estimated keypoint coordinates in full image [num_obj, num_joint, 3], [x, y, score]

    results_path = f"D:/Pycharm Projects-win/mm_mouse/mmpose/work_dirs/" \
                   f"hrnet_w48_mouse_1229_256x256/results/3d"
    # image_files = load_images(img_ind)
    det_model, pose_model, dataset, dataset_info = init_models_datasets(args)

    keypoint_mview = []
    for i, img_file in enumerate(image_files):
        mmdet_results = inference_detector(det_model, img_file)
        # ic([mmdet_results[0][0]])
        # keep the person class bounding boxes.
        mouse_results = process_mmdet_results(mmdet_results, args.det_cat_id)
        mouse_results = [mouse_results[0]]
        pose_results, returned_outputs = inference_top_down_pose_model(
            pose_model,
            img_file,
            mouse_results,
            bbox_thr=args.bbox_thr,
            format='xyxy',
            dataset=dataset,
            dataset_info=dataset_info,
            return_heatmap=False,
            outputs=None)
        keypoint_mview.append(pose_results)

        # show the results
        vis_pose_result(
            pose_model,
            img_file,
            pose_results,
            dataset=dataset,
            dataset_info=dataset_info,
            kpt_score_thr=args.kpt_thr,
            radius=args.radius,
            thickness=args.thickness,
            show=args.show,
            out_file=f"{results_path}/{img_file[-11:-4]}-{i}.png")
    return keypoint_mview, dataset_info


def save_obj(obj, file):
    # name: "a.pkl"
    with open(file, "wb") as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(file):
    with open(file, 'rb') as f:
        return pickle.load(f)


def read_my_calibration_full(calibration_path):
    # BASE_DIR = "C:/Users/admin/PycharmProjects/pythonProject/image_examples"
    # calibration_file = os.path.join(BASE_DIR, "calibration_full.json")
    with open(calibration_path) as json_file:
        calibration_data = json.load(json_file)
        return calibration_data


def make_projection_matrix(calibration_data, cams=["cam0", "cam1", "cam2"]):
    """use the calibration data to compute the projection matrix for cameras
    Returns:
        projection_matrices: array [num_cam, 3, 4] projection matrix
    """
    projection_matrices = []
    for cam in cams:
        cam_matrix = calibration_data[cam]['K']
        cam_matrix = np.array(cam_matrix).reshape([3, 3])
        world_location = np.array(calibration_data[cam]['T']).reshape(3, 1)
        world_orientation = np.array(calibration_data[cam]['R']).reshape(3, 3)

        projection_matrix = np.matmul(cam_matrix, np.concatenate([world_orientation, world_location], axis=1))
        projection_matrices.append(projection_matrix)
    projection_matrices = np.array(projection_matrices)
    return projection_matrices


def triangulate(image_coordinates, projection_matrices):
    '''
    The base triangulation function for NCams. Takes image coordinates and projection matrices from
    2+ cameras and will produce a triangulated point with the desired approach.

    Arguments:
        image_coordinates {array or list of} -- the x,y coordinates of a given marker for multiple
            cameras. The points must be in the format (1,2) if in a list or (n,2) if an array.
        projection_matrices {list} -- the projection matrices for the cameras corresponding
        to each image points input.

    Keyword Arguments:
        mode {str} -- the triangulation method to use:
            full_rank - performs SVD to find the point with the least squares error between all
                projection lines. If a threshold is given along with confidence values then only
                points above the threshold will be used.
            best_n - uses the n number of cameras with the highest confidence values for the
                triangulation. If a threshold is given then only points above the threshold will
                be considered.
            cluster - [in development] performs all combinations of triangulations and checks for
                outlying points suggesting erroneous image coordinates from one or more cameras.
                After removing the camera(s) that produce out of cluser points it then performs the
                full_rank triangulation.
        confidence_values {list or array} -- the confidence values for the points given by the
            marking system (e.g. DeepLabCut)
        threshold {float} -- the minimum confidence value to accept for triangulation.

    Output:
        u_3d {(1,3) np.array} -- the triangulated point produced.

    '''
    u_3d = np.zeros((1, 3))
    u_3d.fill(np.nan)

    # Check if image coordinates are formatted properly
    if isinstance(image_coordinates, list):
        if len(image_coordinates) > 1:
            image_coordinates = np.vstack(image_coordinates)
        else:
            return u_3d

    if not np.shape(image_coordinates)[1] == 2:
        raise ValueError('ncams.reconstruction.triangulate only accepts numpy.ndarrays or lists of' +
                         'in the format (camera, [x,y])')

    num_cameras = np.shape(image_coordinates)[0]
    if num_cameras < 2:  # return NaNs if insufficient points to triangulate
        return u_3d

    if num_cameras != len(projection_matrices):
        raise ValueError('Different number of coordinate pairs and projection matrices given.')

    decomp_matrix = np.empty((num_cameras * 2, 4))
    for decomp_idx in range(num_cameras):
        point_mat = image_coordinates[decomp_idx]
        projection_mat = projection_matrices[decomp_idx]

        temp_decomp = np.vstack([
            [point_mat[0] * projection_mat[2, :] - projection_mat[0, :]],
            [point_mat[1] * projection_mat[2, :] - projection_mat[1, :]]])

        decomp_matrix[decomp_idx * 2:decomp_idx * 2 + 2, :] = temp_decomp

    Q = decomp_matrix.T.dot(decomp_matrix)
    u, _, _ = np.linalg.svd(Q)
    u = u[:, -1, np.newaxis]
    u_3d = np.transpose((u / u[-1, :])[0:-1, :])

    return u_3d


def triangulate_joints(keypoints_mview, projection_matrices, num_joint, kpt_thr):
    """
    perform triangulation on the multiview mmpose estimation results for a frame
    keypoints_mview: [num_cams, num_joints, 3], [x, y, score]
    projection_matrices: [num_cams, 3, 4]
    returns: keypoints_3d [num_joints, 3]
    """
    # num_obj = pose_mview[0][0]['keypoints'].shape[0]
    # num_joint = dataset_info.keypoint_num
    keypoints_3d = np.empty([num_joint, 3])
    keypoints_3d.fill(np.nan)
    # keypoints_mview = np.array([pose_mview[i][0]['keypoints'] for i in range(num_cams)])    #[num_cams, num_joints, 3], [x, y, score]
    for j in range(num_joint):
        cams_detected = keypoints_mview[:, j, 2] > kpt_thr
        cam_idx = np.where(cams_detected)[0]
        if np.sum(cams_detected) < 2:
            continue
        u_3d = triangulate(keypoints_mview[cam_idx, j, :2], projection_matrices[cam_idx])
        keypoints_3d[j, :] = u_3d
    return keypoints_3d


# functions for plot figures
def compute_axis_lim(triangulated_points):
    # triangulated_points in shape [num_frame, num_mouse=2, num_keypoint, 3 axis]
    xlim, ylim, zlim = None, None, None
    minmax = np.nanpercentile(triangulated_points, q=[0, 100], axis=(0, 1, 2)).T
    minmax *= 1.
    minmax_range = (minmax[:, 1] - minmax[:, 0]).max() / 2
    if xlim is None:
        mid_x = np.mean(minmax[0])
        xlim = mid_x - minmax_range, mid_x + minmax_range
    if ylim is None:
        mid_y = np.mean(minmax[1])
        ylim = mid_y - minmax_range, mid_y + minmax_range
    if zlim is None:
        mid_z = np.mean(minmax[2])
        zlim = mid_z - minmax_range, mid_z + minmax_range
    return xlim, ylim, zlim


def main(args):
    img_ind = 5
    image_files = load_images(img_ind)
    cams = ["cam0", "cam1", "cam2", "cam3", "cam4", "cam5"]
    results_path = f"D:/Pycharm Projects-win/mm_mouse/mmpose/work_dirs/" \
                   f"hrnet_w48_mouse_1229_256x256/results/3d"
    pose_results_file = f"{results_path}/{image_files[0][-11:-4]}_det_pose_results.pkl"

    """========detection box, estimate pose, and save the result======="""
    # pose_mview, dataset_info = det_pose_2d(args, image_files)
    # pose_results = {"pose_mview": pose_mview,
    #                 "dataset_info": dataset_info}
    # save_obj(pose_results, pose_results_file)

    """======= read the det_pose results========"""
    pose_results = load_obj(pose_results_file)
    pose_mview = pose_results['pose_mview']
    dataset_info = pose_results['dataset_info']
    # get an object keypoint in an image
    keypoints_mview = np.array([pose_mview[i][0]['keypoints'] for i in range(len(cams))])
    # ic(dir(dataset_info))
    num_joint = dataset_info.keypoint_num

    """===========Reading calibration data=========="""
    calibration_path = f"D:/Datasets/transfer_mouse/onemouse1229/calibration_full.json"
    calibration_data = read_my_calibration_full(calibration_path)

    """===========Computing projection matrix for each camera============="""
    projection_matrices = make_projection_matrix(calibration_data, cams)

    """===========triangulate all joint================"""
    keypoints_3d = triangulate_joints(keypoints_mview, projection_matrices, dataset_info, args)
    ic(keypoints_3d)
    keypoints_3d[:, 2] = 0.1 - keypoints_3d[:, 2]

    keypoints_3d = np.concatenate([keypoints_3d, np.ones([num_joint, 1])], axis=1)

    from mmpose.core import imshow_multiview_keypoints_3d

    # res = [{"keypoints_3d": keypoints_3d}]
    # img = imshow_keypoints_3d(res,
    #                           skeleton=dataset_info.skeleton,
    #                           pose_kpt_color=dataset_info.pose_kpt_color,
    #                           pose_link_color=dataset_info.pose_link_color,
    #                           vis_height=400,
    #                           axis_azimuth=70,
    #                           axis_limit=0.2,
    #                           axis_dist=10.0,
    #                           axis_elev=15.0,
    #                           )
    #
    # from PIL import Image
    # im = Image.fromarray(img)
    # im.save(f"{results_path}/your_file.jpeg")

    res = [keypoints_3d]
    img = imshow_multiview_keypoints_3d(
        res,
        skeleton=dataset_info.skeleton,
        pose_kpt_color=dataset_info.pose_kpt_color,
        pose_link_color=dataset_info.pose_link_color,
        space_size=[0.2, 0.3, 0.2],
        space_center=[0, 0, 0],
        kpt_score_thr=0.0,
    )
    from PIL import Image
    im = Image.fromarray(img)
    im.save(f"{results_path}/your_file.jpeg")

    # keypoints_3d = keypoints_3d.reshape([1, 1, -1, 3])
    # xlim, ylim, zlim = compute_axis_lim(keypoints_3d)
    # keypoints_3d = keypoints_3d.reshape([-1, 3])
    #
    # import matplotlib.pyplot as plt
    # from mpl_toolkits.mplot3d.art3d import Line3DCollection
    #
    # fig = plt.figure(figsize=[30, 30])
    # axes3 = fig.add_subplot(projection="3d")
    # view = (150, 120)
    # axes3.set_xlim3d(xlim)
    # axes3.set_ylim3d(ylim)
    # axes3.set_zlim3d(zlim)
    # axes3.set_box_aspect((1, 1, 1))
    # axes3.xaxis.grid(False)
    # axes3.view_init(view[0], view[1])
    # axes3.set_xlabel("X", fontsize=10)
    # axes3.set_ylabel("Y", fontsize=10)
    # axes3.set_zlabel("Z", fontsize=10)
    #
    # axes3.scatter(keypoints_3d[:, 0],
    #               keypoints_3d[:, 1],
    #               keypoints_3d[:, 2], s=5)
    # segs3d = keypoints_3d[tuple([skeleton])]
    # coll_3d = Line3DCollection(segs3d, linewidths=5)
    # axes3.add_collection(coll_3d)
    # plt.show()


if __name__ == '__main__':
    args = parse_args()
    main(args)
