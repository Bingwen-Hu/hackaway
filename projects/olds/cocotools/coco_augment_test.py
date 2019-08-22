"""
Test the effection of augmenetation

"""

import os
import json
import cv2
import random


images_path = "/media/data/urun_tandong_video/data/signal/aug_images"
annotation_path = "/home/mory/projects/aug_signal.json"
savedir = "/home/mory/projects/test"


with open(annotation_path) as f:
    js = json.load(f)

anns = js['annotations']
images = js['images']
random.shuffle(images)

# just test 5 image
count = 0
for image in images:
    image = image['file_name']
    path = os.path.join(images_path, image)
    img = cv2.imread(path)
    for ann in anns:
        filename = ann['image_name']
        if image == filename:
            x, y, w, h = ann['bbox']
            class_id = ann['category_id']
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(img, f"{class_id}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
    cv2.imwrite(os.path.join(savedir, image), img)
    count += 1
    if count > 5:
        break
