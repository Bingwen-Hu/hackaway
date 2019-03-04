import os
from os.path import basename
import json
import pandas as pd
from collections import Counter

import cv2

TRAIN_DIR = "data/round1_train/restricted/"


def read_json(path: str):
    with open(path, 'r') as f:
        jsonfile = json.load(f)
    return jsonfile

def count(jsonfile: dict):
    return len(jsonfile)


def summary_image(filenames, filesizes, fileids):
    pass


def crop_sample(annotations, images):
    filename = images.file_name[4]
    path = os.path.join(TRAIN_DIR, filename)
    bbox = annotations[annotations.image_id==4]
    bbox = bbox.bbox
    img = cv2.imread(path)
    for bb in bbox:
        cv2.rectangle(img, (int(bb[0]), int(bb[1])), (int(bb[0]+bb[2]), int(bb[1]+bb[3])), 2)
    cv2.imshow("img", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    path = 'data/round1_train/train_no_poly.json'
    jsonfile = read_json(path)
    
    print(f"----- Summary {basename(path)}-----")
    for key in ['categories', 'images', 'annotations']:
        print(f"{key:15s} {count(jsonfile[key])}")

    images = jsonfile['images']
    categories = jsonfile['categories']
    annotations = jsonfile['annotations']


    # convert to pandas DataFrame
    categories = pd.DataFrame(data=categories)
    annotations = pd.DataFrame(data=annotations)
    images = pd.DataFrame(data=images)
    
    # number of kinds
    category_ids = annotations.category_id.tolist()
    print("category ids", Counter(category_ids))

    # size statical
    images['area'] = images.apply(lambda x: x.width * x.height, axis=1)
    print(images.describe())

    # show sample
    crop_sample(annotations, images)