"""
this script crops down the target from image and use its class label as its name
for example:
name = "001_fdfdscdcb.jpg"
so `int(001)` will return the class label


To augment the images, a new image set and a new annotation file should be generated
see coco_augment.py
"""

import os
import cv2
import json
from uuid import uuid1

# paths settings
root = '/media/data/urun_tandong_video/data/signal'
cropdir = os.path.join(root, 'crops')
images_path = os.path.join(root, "train")
annotation_path = "/media/data/urun_tandong_video/data/signal/signal.json"
# create a directory to save the cropped pieces
os.makedirs(cropdir, exist_ok=True)


with open(annotation_path) as f:
    js = json.load(f)

anns = js['annotations']
for ann in anns:
    x, y, w, h = ann['bbox']
    class_id = ann['category_id']
    image_name = ann['image_name']
    img = cv2.imread(os.path.join(images_path, image_name))
    crop = img[y:y+h, x:x+w,:]
    prefix = f"{class_id:03}"
    cv2.imwrite(f'{cropdir}/{prefix}_{uuid1()}.jpg', crop)
    print(f"crop {image_name}")