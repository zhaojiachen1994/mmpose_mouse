import json
import os
import shutil

import numpy as np
import pandas as pd
from icecream import ic


def get_links(joint, skeleton):
    """
    Args:
        joint: list of string, name of joint
        skeleton: list of pairs of string, skeleton name
    Returns:
    """
    # joint = [
    #     'left_ear_tip', 'right_ear_tip', 'nose', 'neck', 'body_middle',
    #     'tail_root', 'tail_middle', 'tail_end',
    #     'left_paw', 'left_shoulder',
    #     'right_paw', 'right_shoulder',
    #     'left_foot', 'left_hip',
    #     'right_foot', 'right_hip']
    # skeleton = [
    #     ['left_ear_tip', 'nose'],
    #     ['right_ear_tip', 'nose'],
    #     ['nose', 'neck'],
    #     ['neck', 'body_middle'],
    #     ['body_middle', 'tail_root'],
    #     ['tail_root', 'tail_middle'],
    #     ['tail_middle', 'tail_end'],
    #     ['neck', 'left_shoulder'],
    #     ['left_shoulder', 'left_paw'],
    #     ['neck', 'right_shoulder'],
    #     ['right_shoulder', 'right_paw'],
    #     ['body_middle', 'left_hip'],
    #     ['left_hip', 'left_foot'],
    #     ['body_middle', 'right_hip'],
    #     ['right_hip', 'right_foot']
    # ]
    num_joint = len(joint)
    name2index = {}
    index2name = {}
    for i, j in enumerate(joint):
        name2index[j] = i
        index2name[i] = j
    links = [[name2index[i], name2index[j]] for [i, j] in skeleton]
    return links


def _keypoints_to_coco(keypoints):  # 根据数据返回一个列表，可视化个数与以关键点为边界的box坐标信息
    key_list = []
    vis_num = 0
    N = keypoints.shape[0] // 3
    invalid = np.isnan(keypoints)  # 判断是否为NaN
    min_x = 10000
    min_y = 10000
    max_x = 0
    max_y = 0
    for curr_id in range(N):  # 如果是NaN，将nan改成0
        num_id = 3 * curr_id
        if invalid[num_id]:
            key_list.append(0)
            key_list.append(0)
            key_list.append(0)
        else:  # 如果不是NaN，更新max,min，列表末尾加x,y,2
            if keypoints[num_id] > max_x:
                max_x = keypoints[num_id]
            if keypoints[num_id + 1] > max_y:
                max_y = keypoints[num_id + 1]
            if keypoints[num_id] < min_x:
                min_x = keypoints[num_id]
            if keypoints[num_id + 1] < min_y:
                min_y = keypoints[num_id + 1]
            key_list.append(float(keypoints[num_id]))
            key_list.append(float(keypoints[num_id + 1]))
            key_list.append(2)
            vis_num += 1
    box = [min_x, min_y, max_x - min_x, max_y - min_y]
    return key_list, vis_num, box


def _expand_box(box, ratio=0.2, W=1600, H=1600):  # 将以关键点为边界的box进行扩展
    x, y, w, h = box
    x1 = x - w * ratio / 2
    y1 = y - h * ratio / 2
    x2 = x + w + w * ratio / 2
    y2 = y + h + h * ratio / 2
    if x1 < 0:
        x1 = 0
    if y1 < 0:
        y1 = 0
    if x2 >= W:
        x2 = W - 1
    if y2 >= H:
        y2 = H - 1
    w_new = x2 - x1
    h_new = y2 - y1
    return [x1, y1, w_new, h_new]


def dlc2coco(folder, cam, joint_name, skeleton, image_start=0):  # 将dlc转换为coco格式
    # file = "C:/Users/wolf/Desktop/mouse/change/transfermouse/datasets/mouse1229/20221229-1-cam0/"
    h5file = f"{folder}/CollectedData_cyd.h5"
    df = pd.read_hdf(h5file)
    values = df.values
    num_images = len(values)

    csvfile = f"{folder}/CollectedData_cyd.csv"
    csv = pd.read_csv(csvfile, sep=',', header='infer', usecols=[2])
    # get the image name
    csv_li = csv.values.tolist()
    name = []
    for s_li in csv_li:
        name.append(s_li[0])

    # add the visible value
    keypoints = np.array(values).reshape([-1, 2])
    vs = np.array([0 if np.isnan(keypoints[i, 0]) else 2 for i in range(len(keypoints))]).reshape([-1, 1])
    keypoints = np.concatenate([keypoints, vs], axis=1)
    keypoints = keypoints.reshape([num_images, -1])
    ic(keypoints.shape)
    np.nan_to_num(keypoints, nan=0.0)

    total_anno = {"annotations": [],
                  "images": [],
                  "categories": [{"id": 1, "keypoints": joint_name, "name": "mouse", "skeleton": skeleton}]}

    for i in range(num_images):
        key_list, vis_num, box = _keypoints_to_coco(keypoints[i])
        bbox = _expand_box(box)
        total_anno['annotations'].append({})
        total_anno['annotations'][i]["image_id"] = i+image_start
        total_anno['annotations'][i]["id"] = i+image_start
        total_anno['annotations'][i]["category_id"] = 1
        total_anno['annotations'][i]["bbox"] = bbox
        total_anno['annotations'][i]["area"] = bbox[2] * bbox[3]
        total_anno['annotations'][i]["keypoints"] = key_list
        total_anno['annotations'][i]["num_keypoints"] = 16
        total_anno['annotations'][i]["segmentation"] = []
        total_anno['annotations'][i]["iscrowd"] = 0
        total_anno['images'].append({})
        total_anno['images'][i]["file_name"] = f"20221229-1-cam{cam}/{name[i + 2]}"
        total_anno['images'][i]["id"] = i+image_start
        total_anno['images'][i]["height"] = 1600
        total_anno['images'][i]["width"] = 1600

    return total_anno


def rename_images(path):
    for i in range(6):
        folder = f"{path}/20221229-1-cam{i}"
        file_list = os.listdir(folder)
        for item in file_list:
            if item.endswith(".png" or ".jpg"):
                src = f"{folder}/{item}"
                tar = f"{path}/images/20221229_1_{item[:-4]}_cam{i}{item[-4:]}"
                shutil.copy(src, tar)


def change_imagename_json(path):
    for i in range(6):
        file = f"{path}/anno_20221229-1-cam{i}.json"
        with open(file, 'r') as f:
            anno = json.load(f)
            image_info_list = anno['images']
            for ii in image_info_list:
                ii['file_name'] = f"20221229_1_{ii['file_name'][:-4]}_cam{i}{ii['file_name'][-4:]}"
            anno = json.dumps(anno, indent=1)
        with open(file, "w", newline='\n') as f:
            f.write(anno)


def combine_json_multi_cams(path, cams, num_image=199):
    anno_dict = {}
    for i in cams:
        file = f"{path}/anno_20221229-1-cam{i}.json"
        with open(file, 'r') as f:
            anno = json.load(f)
            anno_dict[i] = anno
    ic(anno_dict.keys())
    total_anno = {"annotations": [],
                  "images": [],
                  "categories": anno_dict[0]['categories']
                  }
    for i in range(num_image):
        for cam in cams:
            annotations = anno_dict[cam]["annotations"]
            annotations[i]['image_id'] = annotations[i]['image_id'] + cam + i * (len(cams) - 1)
            annotations[i]['id'] = annotations[i]['id'] + cam + i * (len(cams) - 1)
            total_anno['annotations'].append(annotations[i])

            images_part = anno_dict[cam]["images"]
            images_part[i]['id'] = images_part[i]['id'] + cam + i * (len(cams) - 1)
            total_anno['images'].append(images_part[i])
    return total_anno


def scale_cam_T():
    # the cam position unit is m, want mm, so cam'T * 1000
    calibration_file = "D:/Datasets/transfer_mouse/onemouse1229/calibration_adjusted_old.json"
    with open(calibration_file, 'r') as f:
        data = json.load(f)

    ic(data)
    for key in data.keys():
        data[key]['T'] = (np.array(data[key]['T']) * 1000).tolist()

    new_calibration_file = "D:/Datasets/transfer_mouse/onemouse1229/calibration_adjusted.json"
    data = json.dumps(data, indent=1)
    with open(new_calibration_file, "w", newline='\n') as f:
        f.write(data)


def scale_joint_3d():
    file = "D:/Datasets/transfer_mouse/onemouse1229/joints_3d.json"
    with open(file, 'r') as f:
        data = json.load(f)
    # ic(len(data['joint_3d']))
    data['joint_3d'] = (np.array(data['joint_3d']) * 1000).tolist()
    file = "D:/Datasets/transfer_mouse/onemouse1229/anno_20221229_joints_3d.json"
    data = json.dumps(data, indent=1)
    with open(file, "w", newline='\n') as f:
        f.write(data)


if __name__ == "__main__":
    # scale_cam_T()
    # scale_joint_3d()

    joint_name = [
        'left_ear_tip', 'right_ear_tip', 'nose', 'neck', 'body_middle',
        'tail_root', 'tail_middle', 'tail_end',
        'left_paw', 'left_shoulder',
        'right_paw', 'right_shoulder',
        'left_foot', 'left_hip',
        'right_foot', 'right_hip']
    skeleton_name = [
        ['left_ear_tip', 'nose'],
        ['right_ear_tip', 'nose'],
        ['nose', 'neck'],
        ['neck', 'body_middle'],
        ['body_middle', 'tail_root'],
        ['tail_root', 'tail_middle'],
        ['tail_middle', 'tail_end'],
        ['neck', 'left_shoulder'],
        ['left_shoulder', 'left_paw'],
        ['neck', 'right_shoulder'],
        ['right_shoulder', 'right_paw'],
        ['body_middle', 'left_hip'],
        ['left_hip', 'left_foot'],
        ['body_middle', 'right_hip'],
        ['right_hip', 'right_foot']
    ]
    skeleton = get_links(joint_name, skeleton_name)
    path = "D:/Datasets/transfer_mouse/onemouse1229"



    # create json files
    # for i in [0, 1, 2, 3, 4, 5]:
    #     total_anno = dlc2coco(f"{path}/20221229-1-cam{i}", i, joint_name, skeleton)
    #     ic(total_anno.keys())
    #     total_anno = json.dumps(total_anno, indent=1)
    #     anno_file = f"{path}/anno_20221229-1-cam{i}.json"
    #     with open(anno_file, "w", newline='\n') as f:
    #         f.write(total_anno)

    # read the dlc label part2
    # for i in [0, 1, 2, 3, 4, 5]:
    #     total_anno = dlc2coco(f"{path}/labeled-data_part2/20221229-1-cam{i}", i, joint_name, skeleton, image_start=99)
    #     ic(total_anno.keys())
    #     anno_file = f"{path}/labeled-data_part2/anno_20221229-1-cam{i}.json"
    #     total_anno = json.dumps(total_anno, indent=1)
    #     with open(anno_file, "w", newline='\n') as f:
    #         f.write(total_anno)

    """combine two parts"""
    # for i in [0, 1, 2, 3, 4, 5]:
    #     file_part1 = f"{path}/labeled-data_part1/anno_20221229-1-cam{i}.json"
    #     with open(file_part1, 'r') as f:
    #         anno_part1 = json.load(f)
    #     ic(anno_part1.keys())
    #     ic(len(anno_part1['annotations']))
    #     file_part2 = f"{path}/labeled-data_part2/anno_20221229-1-cam{i}.json"
    #     with open(file_part2, 'r') as f:
    #         anno_part2 = json.load(f)
    #     ic(len(anno_part2['annotations']))
    #     total_anno = {}
    #     total_anno['annotations'] = anno_part1['annotations']+anno_part2['annotations']
    #     total_anno['images'] = anno_part1['images']+anno_part2['images']
    #     total_anno['categories'] = anno_part1['categories']
    #     # ic(len(total_anno['annotations']))
    #     anno_file = f"{path}/anno_20221229-1-cam{i}.json"
    #     total_anno = json.dumps(total_anno, indent=1)
    #     with open(anno_file, "w", newline='\n') as f:
    #         f.write(total_anno)


    # combine all json files
    total_anno_file = f"{path}/anno_20221229-1-012345.json"
    total_anno = combine_json_multi_cams(path, cams=[0, 1, 2, 3, 4, 5])
    total_anno = json.dumps(total_anno, indent=1)
    with open(total_anno_file, "w", newline='\n') as f:
        f.write(total_anno)

    # split train and evaluation files
    total_anno_file = f"{path}/anno_20221229-1-012345.json"
    with open(total_anno_file, 'r') as f:
        total_anno = json.load(f)
    ic(total_anno.keys())
    ic(len(total_anno['annotations']))
    #
    train_anno = {
        'annotations': total_anno['annotations'][:960],
        'images': total_anno['images'][:960],
        'categories': total_anno['categories']
    }
    train_anno_file = f"{path}/anno_20221229-1-012345_train.json"
    with open(train_anno_file, "w") as f:
        json.dump(train_anno, f)

    eval_anno = {
        'annotations': total_anno['annotations'][960:],
        'images': total_anno['images'][960:],
        'categories': total_anno['categories']}
    eval_anno_file = f"{path}/anno_20221229-1-012345_test.json"
    with open(eval_anno_file, "w") as f:
        json.dump(eval_anno, f)
