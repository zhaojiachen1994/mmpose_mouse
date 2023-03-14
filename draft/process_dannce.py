import json

import numpy as np

if __name__ == "__main__":
    """20230202 convert nan value in json to 0"""
    # file = "D:/Datasets/transfer_mouse/dannce_20230130/annotations_visible.json"
    # with open(file, 'r') as f:
    #     anno = json.load(f)
    # ic(anno.keys)
    # ic(len(anno['annotations']))
    # for i in range(len(anno['annotations'])):
    #     keypoints = np.array(anno['annotations'][i]['keypoints'])
    #     keypoints = np.nan_to_num(keypoints)
    #     anno['annotations'][i]['keypoints'] = keypoints.tolist()
    # new_file = "D:/Datasets/transfer_mouse/dannce_20230130/annotations_visible_new.json"
    # with open(new_file, "w") as outfile:
    #     json.dump(anno, outfile)

    """split the train and eval set"""
    # total_anno_file = "D:/Datasets/transfer_mouse/dannce_20230130/annotations_visible_new2.json"
    # with open(total_anno_file, 'r') as f:
    #     total_anno = json.load(f)
    #
    # train_anno = {
    #     'annotations': total_anno['annotations'][:930],
    #     'images': total_anno['images'],
    #     'categories': total_anno['categories']
    # }
    # train_anno_file = "D:/Datasets/transfer_mouse/dannce_20230130/annotations_visible_train930_new2.json"
    # with open(train_anno_file, "w") as f:
    #     json.dump(train_anno, f)
    #
    # eval_anno = {
    #     'annotations': total_anno['annotations'][930:],
    #     'images': total_anno['images'],
    #     'categories': total_anno['categories']}
    # eval_anno_file = "D:/Datasets/transfer_mouse/dannce_20230130/annotations_visible_eval930_new2.json"
    # with open(eval_anno_file, "w") as f:
    #     json.dump(eval_anno, f)

    """add the scene id and cam id for the ann_file"""
    # total_anno_file = "D:/Datasets/transfer_mouse/dannce_20230130/annotations_visible_new.json"
    # with open(total_anno_file, 'r') as f:
    #     total_anno = json.load(f)
    # ic(total_anno.keys())
    # num_images = len(total_anno['annotations'])
    # num_scenes = int(num_images/6)
    # for i in range(num_scenes):
    #     for c in range(6):
    #         total_anno['annotations'][6*i+c]['scene_id'] = i
    #         total_anno['annotations'][6*i+c]['cam_id'] = c
    # file2 = "D:/Datasets/transfer_mouse/dannce_20230130/annotations_visible_new2.json"
    # with open(file2, "w") as f:
    #     json.dump(total_anno, f)

    """add img_id for joint3d annotations and save in dict format"""
    # ann_3d_file = "D:/Datasets/transfer_mouse/dannce_20230130/joints_3d_old.json"
    # with open(ann_3d_file, 'rb') as f:
    #     data = json.load(f)
    # data = np.array(data['joint_3d'])
    # [num_sample, num_joints, _] = data.shape
    # ic(num_sample)
    #
    # ann_3d_dict = {}
    # for i in range(num_sample):
    #     ann_3d_dict[i] = data[i].tolist()
    # ic(ann_3d_dict)
    #
    # new_ann_3d_file = "D:/Datasets/transfer_mouse/dannce_20230130/joints_3d.json"
    # with open(new_ann_3d_file, 'w') as f:
    #     json.dump(ann_3d_dict, f)

    """split the joint3d file to train and evaluate set"""
    total_anno_file = "D:/Datasets/transfer_mouse/dannce_20230130/joints_3d_list.json"
    with open(total_anno_file, 'r') as f:
        data = json.load(f)
    data = np.array(data['joint_3d'])
    [num_sample, num_joints, _] = data.shape
    joints_3d_visible = np.ones_like(data, dtype=np.float32) * 2
    joints_3d_visible[np.isnan(data)] = 0.0
    joints_3d_visible = joints_3d_visible[:, :, 0]

    # data_train = data[:155]
    # data_test = data[155:]
    ann_3d_dict = {}
    ann_train_3d_dict = {}
    ann_test_3d_dict = {}
    for i in range(num_sample):
        ann_3d_dict[i] = {}
        ann_3d_dict[i]['joints_3d'] = data[i].tolist()
        ann_3d_dict[i]['joints_3d_visible'] = joints_3d_visible[i].tolist()
        if i < 155:
            ann_train_3d_dict[i] = {}
            ann_train_3d_dict[i]['joints_3d'] = data[i].tolist()
            ann_train_3d_dict[i]['joints_3d_visible'] = joints_3d_visible[i].tolist()

        else:
            # ann_test_3d_dict[i] = data[i].tolist()
            ann_test_3d_dict[i] = {}
            ann_test_3d_dict[i]['joints_3d'] = data[i].tolist()
            ann_test_3d_dict[i]['joints_3d_visible'] = joints_3d_visible[i].tolist()

    new_ann_3d_file = "D:/Datasets/transfer_mouse/dannce_20230130/joints_3d.json"
    with open(new_ann_3d_file, 'w') as f:
        json.dump(ann_3d_dict, f)

    new_ann_train_3d_file = "D:/Datasets/transfer_mouse/dannce_20230130/joints_3d_train.json"
    with open(new_ann_train_3d_file, 'w') as f:
        json.dump(ann_train_3d_dict, f)

    new_ann_test_3d_file = "D:/Datasets/transfer_mouse/dannce_20230130/joints_3d_test.json"
    with open(new_ann_test_3d_file, 'w') as f:
        json.dump(ann_test_3d_dict, f)
