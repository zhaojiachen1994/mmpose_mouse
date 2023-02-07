import json

import numpy as np
from icecream import ic

if __name__ == "__main__":
    # 20230202 convert nan value in json to 0
    file = "D:/Datasets/transfer_mouse/dannce_20230130/annotations_visible.json"
    with open(file, 'r') as f:
        anno = json.load(f)
    ic(anno.keys)
    ic(len(anno['annotations']))
    for i in range(len(anno['annotations'])):
        keypoints = np.array(anno['annotations'][i]['keypoints'])
        keypoints = np.nan_to_num(keypoints)
        anno['annotations'][i]['keypoints'] = keypoints.tolist()
    new_file = "D:/Datasets/transfer_mouse/dannce_20230130/annotations_visible_new.json"
    with open(new_file, "w") as outfile:
        json.dump(anno, outfile)

    # split the train and eval set
    total_anno_file = "D:/Datasets/transfer_mouse/dannce_20230130/annotations_visible_new.json"
    with open(total_anno_file, 'r') as f:
        total_anno = json.load(f)

    train_anno = {
        'annotations': total_anno['annotations'][:930],
        'images': total_anno['images'],
        'categories': total_anno['categories']
    }
    train_anno_file = "D:/Datasets/transfer_mouse/dannce_20230130/annotations_visible_train930_new.json"
    with open(train_anno_file, "w") as f:
        json.dump(train_anno, f)

    eval_anno = {
        'annotations': total_anno['annotations'][930:],
        'images': total_anno['images'],
        'categories': total_anno['categories']}
    eval_anno_file = "D:/Datasets/transfer_mouse/dannce_20230130/annotations_visible_eval930_new.json"
    with open(eval_anno_file, "w") as f:
        json.dump(eval_anno, f)
