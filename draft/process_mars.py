import json

import numpy as np
from icecream import ic


def add_area():
    root = "D:/Datasets/MARS-PoseAnnotationData"
    ann_file = f"{root}/MARS_keypoints_front_black.json"
    with open(ann_file, 'r') as file:
        str = file.read()
        data = json.loads(str)
    ic(data.keys())
    ic(data['annotations'][0])
    for i in range(len(data['annotations'])):
        w = data['annotations'][i]['bbox'][2]
        h = data['annotations'][i]['bbox'][3]
        area = round(w * h, 2)
        data['annotations'][i]['area'] = area
        ic(area)
    anno_file2 = f"{root}/MARS_keypoints_front_black2.json"
    data = json.dumps(data, indent=1)
    with open(anno_file2, "w", newline='\n') as f:
        f.write(data)


def select_sample():
    """choose the high-quality samples for mars dataset"""
    # train a model on total dataset, then predict total dataset,
    # choose the sample with score>0.8 for top view and score>0.55 for front view

    # for top view
    result_file = "../work_dirs/dataset_mars_p5/result_keypoints.json"
    view = "top"
    score_thr = 0.9

    # for front view
    # result_file = "../work_dirs/dataset_mars_p5_front/result_keypoints.json"
    # view = "front"
    # score_thr = 0.69

    with open(result_file, 'r') as file:
        str = file.read()
        data = json.loads(str)
    ic(len(data))

    ids = []  # the high-quality image id
    for i in range(len(data)):
        score = np.mean(np.array(data[i]["keypoints"])[[2, 5, 8, 11, 14]])
        if score > score_thr:
            # if data[i]['score'] > score_thr:
            #     ic(i, np.mean(np.array(data[i]["keypoints"])[[2, 5, 8, 11, 14]]))
            ids.append(data[i]['image_id'])
    ic(len(ids))

    # build new annotation file with only high-quality samples

    total_anno_file = f"D:/Datasets/MARS-PoseAnnotationData/MARS_keypoints_{view}_black2.json"
    with open(total_anno_file, 'r') as file:
        str = file.read()
        total_anno = json.loads(str)

    hq_anno = {}
    hq_anno['annotations'] = []
    hq_anno['images'] = []
    hq_anno['categories'] = total_anno['categories']
    for i in range(len(ids)):
        # for i in range(10):
        hq_id = ids[i]
        anno_temp = total_anno['annotations'][hq_id]
        anno_temp['image_id'] = i
        anno_temp['id'] = i
        hq_anno['annotations'].append(anno_temp)

        image_temp = total_anno['images'][hq_id]
        image_temp['id'] = i
        hq_anno['images'].append(image_temp)

    root = "D:/Datasets/MARS-PoseAnnotationData"
    anno_file_hq = f"{root}/MARS_keypoints_{view}_black_hq.json"
    hq_anno = json.dumps(hq_anno, indent=1)
    with open(anno_file_hq, "w", newline='\n') as f:
        f.write(hq_anno)


if __name__ == "__main__":
    ic("===")
    select_sample()
