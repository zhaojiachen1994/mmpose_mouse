import json
from icecream import ic
import json

from icecream import ic

if __name__ == "__main__":
    root = "D:/Datasets/MARS-PoseAnnotationData"
    raw_ann_file = f"{root}/MARS_keypoints_top_black.json"
    with open(raw_ann_file, 'r') as file:
        str = file.read()
        data = json.loads(str)
    ic(data.keys())
    ic(data['img_metas'])
