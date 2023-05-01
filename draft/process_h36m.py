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

    actions_ids = [f"{i:02d}" for i in range(2, 16)]    # subject11 only works for (2, 16)
    subact_ids = [f"{i:02d}" for i in range(1, 3)]
    cams = [f"{i:02d}" for i in range(1, 5)]

    id = 0

    for subject in [9]:
        annotations = {}
        annotations['images'] = []
        annotations['annotations'] = []
        video_start = 0
        anno_file = f"{data_root}/annotations_old/Human36M_subject{subject}_joint_2d.json"
        with open(anno_file, 'r') as f:
            orginal_anno_data = json.load(f)
            ic(orginal_anno_data.keys())
            ic(len(orginal_anno_data['images']))
            ic(len(orginal_anno_data['annotations']))

            # global_id_start = orginal_anno_data['images'][0]['id']  # the start id of this subject in all dataset
        for action_id in actions_ids:
            for subact_id in subact_ids:
                image_folder = \
                    f"s_{subject:02d}_act_{action_id}_subact_{subact_id}_ca_04"

                img_video = os.listdir(f"{data_root}/images/{image_folder}")
                num_frames = len(img_video) - 1

                if subject == 11 and action_id == '02' and subact_id == '02':  # image is lost
                    video_start = video_start + num_frames * 3
                    continue

                # if subject == 9 and action_id == '05' and subact_id == '02':    # greating
                if subject == 9 and action_id == '05':  # greating
                    video_start = video_start + num_frames * 4
                    continue

                if subject == 9 and action_id == '10':  # sittingdown
                    video_start = video_start + num_frames * 4
                    continue

                # if subject == 9 and action_id == '10' and subact_id == '02':    # sittingdown
                if subject == 9 and action_id == '10':  # sittingdown
                    video_start = video_start + num_frames * 4
                    continue

                # if subject == 9 and action_id == '13' and subact_id == '01':  # waiting
                if subject == 9 and action_id == '13':  # waiting
                    video_start = video_start + num_frames * 4
                    continue

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
                # ic(action_id, subact_id, i_frame, i_cam)
        new_anno_file = f"{data_root}/annotations_old/Human36M_subject{subject}_data_reorder.json"
        with open(new_anno_file, "w") as f:
            json.dump(annotations, f)


def downsample_h36m():
    suffix = "_0002"
    if suffix == "_10hz":  #
        ss = 4 * 5
    if suffix == "":  #
        ss = 4 * 10
    elif suffix == "_0001":
        ss = 4 * 10000  # 0.001 to train
    elif suffix == "_0002":
        ss = 4 * 5000
    elif suffix == "_001":  # 0.01 to train
        ss = 4 * 1000
    elif suffix == "_002":  # 0.02 to train
        ss = 4 * 500
    elif suffix == "_005":  # 0.05 to train
        ss = 4 * 200
    elif suffix == "_01":  # 0.05 to train
        ss = 4 * 100
    elif suffix == "_02":  # 0.05 to train
        ss = 4 * 50

    # elif suffix == "_02":  # 0.05 to train
    # ss = 4 * 100

    elif suffix == "_02":  # 0.05 to train
        ss = 4 * 50

    # for person in [1,5,6,7,8,9]:
    for person in [1]:
        with open(
                f"D:/Datasets/h36m_dataset/human3.6m_parse/annotations_old/Human36M_subject{person}_data_reorder.json",
                'r',
                encoding='UTF-8') as f:
            load_dict = json.load(f)
            ic(load_dict.keys())
            new_dict = {}
            assert len(load_dict['images']) == len(load_dict['annotations'])
            new_dict['images'] = []
            new_dict['annotations'] = []
            print(len(load_dict['images']))
            for i in range(0, len(load_dict['images']) - 4, ss):
                # base down: 200 for 1/50, 1hz; 2000 for 1/500, 10% supervised, 1000 for
                # print(i, load_dict['images'][i]['cam_idx'])
                new_dict['images'].append(load_dict['images'][i])
                assert load_dict['images'][i]['cam_idx'] == 1
                new_dict['images'].append(load_dict['images'][i + 1])
                assert load_dict['images'][i + 1]['cam_idx'] == 2
                new_dict['images'].append(load_dict['images'][i + 2])
                assert load_dict['images'][i + 2]['cam_idx'] == 3
                new_dict['images'].append(load_dict['images'][i + 3])
                assert load_dict['images'][i + 3]['cam_idx'] == 4
                assert load_dict['images'][i]['frame_idx'] == load_dict['images'][i + 3]['frame_idx']
                new_dict['annotations'].append(load_dict['annotations'][i])
                new_dict['annotations'].append(load_dict['annotations'][i + 1])
                new_dict['annotations'].append(load_dict['annotations'][i + 2])
                new_dict['annotations'].append(load_dict['annotations'][i + 3])
            ic(new_dict.keys())
            ic(len(new_dict['annotations']))
            ic(len(new_dict['images']))
            f.close()

        for i in range(len(new_dict['images'])):
            new_dict['images'][i]['id'] = i
            new_dict['annotations'][i]['id'] = i
            new_dict['annotations'][i]['image_id'] = i

        with open(f"D:/Datasets/h36m_dataset/human3.6m_parse/annotations/Human36M_subject{person}_data{suffix}.json",
                  'w', encoding='UTF-8') as g:
            json.dump(new_dict, g)


def combine_multi_subjects():
    root = "D:/Datasets/h36m_dataset/human3.6m_parse/annotations"
    # suffix = "_00"
    suffix = ""
    total_anno = {'images': [], 'annotations': []}
    # subjects = "15678"
    # total_anno_file = f"{root}/Human36M_subjects_{subjects}{suffix}.json"
    subjects = "911"
    total_anno_file = f"{root}/Human36M_subjects_{subjects}{suffix}.json"
    for i in subjects:
        file = f"{root}/Human36M_subject{i}_data{suffix}.json"
        with open(file, 'r') as f:
            anno_i = json.load(f)
            total_anno['images'].extend(anno_i['images'])
            total_anno['annotations'].extend(anno_i['annotations'])
    ic(total_anno.keys())
    ic(len(total_anno['images']))
    for i in range(len(total_anno['images'])):
        total_anno['images'][i]['id'] = i
        total_anno['annotations'][i]['id'] = i
        total_anno['annotations'][i]['image_id'] = i
    total_anno = json.dumps(total_anno, indent=0)
    with open(total_anno_file, "w") as f:
        f.write(total_anno)


def process_p11():
    subject = 11
    data_root = "D:/Datasets/h36m_dataset/human3.6m_parse"
    anno_file = f"{data_root}/annotations_old/Human36M_subject11_data.json"
    with open(anno_file, 'r') as f:
        anno_data = json.load(f)
    ic(anno_data.keys())
    ic(len(anno_data['images']))
    ic(len(anno_data['annotations']))
    actions_ids = [f"{i:02d}" for i in range(2, 4)]  # subject11 only works for (2, 16)
    subact_ids = [f"{i:02d}" for i in range(1, 3)]
    cams = [f"{i:02d}" for i in range(1, 5)]
    video_start = 0
    for action_id in actions_ids:
        for subact_id in subact_ids:
            image_folder = \
                f"s_{subject:02d}_act_{action_id}_subact_{subact_id}_ca_04"
            img_video = os.listdir(f"{data_root}/images/{image_folder}")
            num_frames = len(img_video) - 1
            if action_id == '02' and subact_id == '02':
                video_start = video_start + num_frames * 3
                print(video_start)
                continue
            for i_frame in range(num_frames):
                for i_cam, cam in enumerate(cams):
                    orginal_index = video_start + num_frames * i_cam + i_frame
                    # ic(anno_data['images'][orginal_index]['cam_idx'], int(cam))
                    if anno_data['images'][orginal_index]['cam_idx'] != int(cam):
                        print(anno_data['images'][orginal_index]['id'])
                        print(orginal_index, cam, action_id, subact_id, i_frame,
                              anno_data['images'][orginal_index]['cam_idx'])

            # if action_id == 2 and subact_id == 2:
            #     video_start = video_start + num_frames * 3
            # else:
            video_start = video_start + num_frames * 4


if __name__ == "__main__":
    # reorder_annotations()
    downsample_h36m()
    # combine_multi_subjects()
    # process_p11()
