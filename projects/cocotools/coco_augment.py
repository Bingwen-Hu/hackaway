# To augment the images, a new image set and a new annotation file should be generated
# Before this step, crop image should be generated. See coco_crop.py

import os
import cv2
import json
import random
import math
from uuid import uuid1
from imutils import paths


# paths settings
root = '/media/data/urun_tandong_video/data/signal'
cropdir = os.path.join(root, 'crops')
images_path = os.path.join(root, "train")
json_path = os.path.join(root,'signal.json')
aug_images_path = os.path.join(root, "aug_train")
aug_json_path = os.path.join(root, "aug_signal.json")

# variables prepare
os.makedirs(aug_images_path, exist_ok=True)
crops = list(paths.list_images(cropdir))
with open(json_path) as f:
    js = json.load(f)
anns = js['annotations']

threshold = 200
im_x = 10
im_y = 10
im_h = 1800 - threshold
im_w = 3200 - threshold

def distance(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

def generate_xy(xs=im_x, xe=im_w, ys=im_y, ye=im_h, exist=[]):
    # generate 7 point
    num = 7 - len(exist)
    for _ in range(num):
        p_new = random.randint(xs, xe), random.randint(ys, ye)
        overlap = False
        for p in exist:
            if distance(p, p_new) < threshold:
                overlap = True
                break
        if not overlap:
            exist.append(p_new)
        overlap = False
    return exist

def generate_crops(num):
    paths = random.choices(crops, k=num)
    labels = [int(os.path.basename(p)[:3]) for p in paths]
    imgs = [cv2.imread(p) for p in paths]
    return labels, imgs

# deep copy
from copy import deepcopy
new_anns = deepcopy(anns)


# NOTE: only for image that only has one target!!
# if the image has more than one annotation, it will be repeatly augmented!
total = len(anns)
global_ann_id = total
for i, ann in enumerate(anns):
    filename = ann['image_name']
    image_path = os.path.join(images_path, filename)
    img = cv2.imread(image_path)
    org_x, org_y, org_w, org_h = ann['bbox']
    # category_id = ann['category_id']
    # generate other 5 points
    xy_list = generate_xy(exist=[(org_x,org_y)])
    xy_list.remove((org_x,org_y))
    # generate crop to paste
    labels, imgs = generate_crops(num=len(xy_list))
    for (x, y), label, crop in zip(xy_list, labels, imgs):
        try:
            h, w = crop.shape[:2]
        except:
            continue
        else:
            new_ann = {
                    'area': w*h,
                    'bbox': [x,y,w,h],
                    'category_id': label,
                    'id': global_ann_id,
                    'image_id': ann['image_id'],
                    'image_name': filename,
                    'iscrowd': 0,
                    'segmentation': [],
            }
            # update new anns
            new_anns.append(new_ann)
            global_ann_id += 1
            # paste crop into image
            img[y:y+h, x:x+w, :] = crop
    # finally, output the augmented image
    print(f"Augment image:({i}/{total})")
    cv2.imwrite(os.path.join(aug_images_path, filename), img)

# update json file
js['annotations'] = new_anns

with open(aug_json_path, 'w') as f:
    json.dump(js, f)

