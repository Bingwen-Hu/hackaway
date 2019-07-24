import sys
import os
import os.path as osp
# enable import `deepfree` 
sys.path.insert(0, osp.abspath('..'))

import cv2
from deepfree.datasets.utils import visual
from pycocotools.coco import COCO


# dataset path
person_json = "/media/data/urun_tandong_video/data/COCO/annotations/person_keypoints_train2014.json"
imdir = "/media/data/urun_tandong_video/data/COCO/images/train2014/"

coco = COCO(person_json)
keys = list(coco.imgs.keys())

# search for a good key with 17 keypoints
find = False
for key in keys:
    ann_ids = coco.getAnnIds(key)
    ann_metas = coco.loadAnns(ann_ids)
    if len(ann_metas) > 0:
        for ann_meta in ann_metas:
            if ann_meta['num_keypoints'] > 16:
                keypoints = ann_meta['keypoints']
                find = True
                break
    if find:
        break

im_meta = coco.loadImgs(key)[0] # list
im_path = f"{imdir}/{im_meta['file_name']}"
im = cv2.imread(im_path)
ngroup = 17 # 17 pairs of x,y,v
visual.plot_keypoints(im, keypoints, ngroup, f"{im_meta['file_name']}")
