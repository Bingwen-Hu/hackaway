"""
To augment the images, a new image set and a new annotation file should be generated
Before this step, crop image should be generated. See coco_crop.py

NOTE: only for image that only has one target!!
if the image has more than one annotation, it will be repeatly augmented!
"""

import os
import cv2
import json
import random
import math
from copy import deepcopy
from uuid import uuid1
from imutils import paths


# paths settings
root = '/media/data/urun_tandong_video/data/signal'
cropdir = os.path.join(root, 'crops')
images_path = os.path.join(root, "train")
annotation_path = os.path.join(root,'signal.json')
aug_images_path = os.path.join(root, "aug_train")
aug_annotation_path = os.path.join(root, "aug_signal.json")

# variables prepare
os.makedirs(aug_images_path, exist_ok=True)
crops = list(paths.list_images(cropdir))
with open(annotation_path) as f:
    js = json.load(f)
anns = js['annotations']

# threshold is used to avoid overlap cross different objects
# the input image is 3200x1800
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
        # check whether the new generated point overlaps with existing one
        # if yes, set `overlap = True`
        for p in exist:
            if distance(p, p_new) < threshold:
                overlap = True
                break
        # if overlaped, simplely skip this point
        if not overlap:
            exist.append(p_new)
        overlap = False
    return exist

def generate_crops(num):
    """Randomly select certain number pieces of crops"""
    paths = random.choices(crops, k=num)
    labels = [int(os.path.basename(p)[:3]) for p in paths]
    imgs = [cv2.imread(p) for p in paths]
    return labels, imgs


# deep copy, so modify the new one will never affect the old one
new_anns = deepcopy(anns)
total = len(anns)
# we generate new annotations on the original annotations list, so we should check the 
# number of annotations and use a global variable to assign the annotation id
global_ann_id = total
for i, ann in enumerate(anns):
    filename = ann['image_name']
    image_path = os.path.join(images_path, filename)
    img = cv2.imread(image_path)
    org_x, org_y, org_w, org_h = ann['bbox']
    # generate new points
    xy_list = generate_xy(exist=[(org_x,org_y)])
    # we don't want to overwrite the original annotation, so remove the orginal one
    xy_list.remove((org_x,org_y))
    # generate crop to paste
    labels, imgs = generate_crops(num=len(xy_list))
    for (x, y), label, crop in zip(xy_list, labels, imgs):
        try:
            h, w = crop.shape[:2]
        except:
            print("Some crop seems broken, skip it")
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

with open(aug_annotation_path, 'w') as f:
    json.dump(js, f)