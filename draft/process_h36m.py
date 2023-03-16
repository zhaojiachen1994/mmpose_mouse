import json
import os

import cv2
import numpy as np
from icecream import ic

from mmpose.core.bbox import bbox_xywh2xyxy
from mmpose.core.camera import MyCamera


def try_3dproject2d():
    data_root = "D:/Datasets/h36m_dataset/human3.6m_parse"
    camera_file = f"{data_root}/annotations/Human36M_subject1_camera.json"

    """parse the camera parameter, build a camera object"""
    cam_idx = 1
    with open(camera_file, 'r') as f:
        camera_params = json.load(f)
    cam_param = camera_params[str(cam_idx)]
    mycamera = MyCamera(cam_param)

    """read the annotation data"""
    box_file = f"{data_root}/annotations/Human36M_subject1_data.json"
    with open(box_file, 'r') as f:
        anno_data = json.load(f)
    ic(anno_data.keys())
    ic(len(anno_data['images']))

    id = 0
    image_file = anno_data['images'][id]['file_name']
    keypoint_vis = anno_data['annotations'][id]['keypoints_vis']
    bbox = anno_data['annotations'][id]['bbox']  # in xywh format
    bbox = np.array(bbox).reshape([-1, 4])
    bbox = bbox_xywh2xyxy(bbox)
    ic(anno_data['images'][id]['cam_idx'])
    image_file = f"{data_root}/images/{image_file}"
    ic(image_file)

    """visualize image and bbox"""
    img = cv2.imread(image_file)
    # cv2.imshow('image', img)
    # cv2.waitKey(0)   # waits until a key is pressed
    # cv2.destroyAllWindows()     # destroys the window showing image
    # img = imshow_bboxes(img, bbox, labels='human', )

    """read 3d annotations"""
    # joint_3d_file = f"{data_root}/annotations/Human36M_subject1_joint_3d.json"
    # with open(joint_3d_file, 'r') as f:
    #     joint_3d_data = json.load(f)
    # ic(joint_3d_data.keys())
    # ic(np.array(joint_3d_data['2']['1']['0']).shape)
    #
    # ic(anno_data['images'][id].keys())
    # action_idx = str(anno_data['images'][id]['action_idx'])
    # subaction_idx = str(anno_data['images'][id]['subaction_idx'])
    # frame_idx = str(anno_data['images'][id]['frame_idx'])
    # joints_3d = np.array(joint_3d_data[action_idx][subaction_idx][frame_idx])
    # ic(joints_3d.shape)
    # ic(joints_3d)
    # joint_2d = mycamera.world_to_pixel(joints_3d)
    # ic(joint_2d.shape)

    """visualize the projected points"""
    # pose_kpt_color = np.array(17*[109, 192, 91]).reshape(17, 3)
    # ic(pose_kpt_color.shape)
    # img = imshow_keypoints(img, [joint_2d], pose_kpt_color=pose_kpt_color)
    # cv2.imshow('image', img)
    # cv2.waitKey(0)   # waits until a key is pressed
    # cv2.destroyAllWindows()     # destroys the window showing image


def try_get_joint2d():
    data_root = "D:/Datasets/h36m_dataset/human3.6m_parse"
    subject = 11
    camera_file = f"{data_root}/annotations/Human36M_subject{subject}_camera.json"

    """parse the camera parameter, build a camera object"""
    with open(camera_file, 'r') as f:
        camera_params = json.load(f)
    # cam_param = camera_params[str(cam_idx)]
    # mycamera = MyCamera(cam_param)
    # cameras = [MyCamera(camera_params[str(cam_idx)]) for cam_idx in [1, 2, 3, 4]]
    cams = {}
    for cam_idx in [1, 2, 3, 4]:
        cams[cam_idx] = MyCamera(camera_params[str(cam_idx)])

    """read the annotation data"""
    anno_file = f"{data_root}/annotations/Human36M_subject{subject}_data.json"
    with open(anno_file, 'r') as f:
        anno_data = json.load(f)
    ic(anno_data.keys())
    ic(len(anno_data['images']))

    joint_3d_file = f"{data_root}/annotations/Human36M_subject{subject}_joint_3d.json"
    with open(joint_3d_file, 'r') as f:
        joint_3d_data = json.load(f)

    for i in range(len(anno_data['images'])):
        action_idx = str(anno_data['images'][i]['action_idx'])
        subaction_idx = str(anno_data['images'][i]['subaction_idx'])
        frame_idx = str(anno_data['images'][i]['frame_idx'])
        cam_idx = anno_data['images'][i]['cam_idx']
        joints_3d = np.array(joint_3d_data[action_idx][subaction_idx][frame_idx])
        joint_2d = cams[cam_idx].world_to_pixel(joints_3d)

        keypoints_vis_bool = anno_data['annotations'][i]['keypoints_vis']
        keypoints_vis_temp = np.zeros(17)
        keypoints_vis_temp[keypoints_vis_bool] = 2
        joint_2d[:, -1] = keypoints_vis_temp
        anno_data['annotations'][i]['keypoints'] = joint_2d.tolist()

    joint2d_file = f"{data_root}/annotations/Human36M_subject{subject}_joint_2d.json"
    with open(joint2d_file, "w") as f:
        json.dump(anno_data, f)


def reorder_annotations():
    # read subject1 anno file
    data_root = "D:/Datasets/h36m_dataset/human3.6m_parse"
    # subject = 1
    # anno_file = f"{data_root}/annotations/Human36M_subject{subject}_data.json"
    # with open(anno_file, 'r') as f:
    #     anno_data = json.load(f)
    # ic(anno_data.keys())
    # ic(len(anno_data['images']))
    # ic(anno_data['annotations'][0])

    actions_ids = [f"{i:02d}" for i in range(2, 17)]
    subact_ids = [f"{i:02d}" for i in range(1, 3)]
    cams = [f"{i:02d}" for i in range(1, 5)]

    id = 0

    for subject in [1, 5, 6, 7, 8, 9, 11]:
        # doesn't work for subject 11
        annotations = {}
        annotations['images'] = []
        annotations['annotations'] = []
        video_start = 0
        anno_file = f"{data_root}/annotations/Human36M_subject{subject}_joint_2d.json"
        with open(anno_file, 'r') as f:
            orginal_anno_data = json.load(f)
            global_id_start = anno_data['images'][0]['id']  # the start id of this subject in all dataset
        for action_id in actions_ids:
            for subact_id in subact_ids:
                image_folder = \
                    f"s_{subject:02d}_act_{action_id}_subact_{subact_id}_ca_04"

                img_video = os.listdir(f"{data_root}/images/{image_folder}")
                num_frames = len(img_video) - 1
                for i_frame in range(num_frames):
                    for i_cam, cam in enumerate(cams):
                        orginal_index = video_start + num_frames * i_cam + i_frame
                        img_one = orginal_anno_data['images'][orginal_index]
                        img_one['id'] = id
                        annotations['images'].append(img_one)
                        anno_one = orginal_anno_data['annotations'][orginal_index]
                        anno_one['id'] = id
                        anno_one['image_id'] = id
                        annotations['annotations'].append(anno_one)
                        id = id + 1
                video_start = video_start + num_frames * 4
        new_anno_file = f"{data_root}/annotations/Human36M_subject{subject}_data_reorder.json"
        with open(new_anno_file, "w") as f:
            json.dump(annotations, f)


def downsample_h36m():
    subject = 1


if __name__ == "__main__":
    reorder_annotations()
