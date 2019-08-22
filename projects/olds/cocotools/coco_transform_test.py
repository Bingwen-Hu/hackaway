"""
This python script is used to create an annotation comparable to MS coco format for test images

"""

import json
import os
import cv2

root = "/media/data/urun_tandong_video/data/signal/"
classes_path = 'classes.txt'
images_dir = 'test'
save_json_filename = "signal_test.json"

# final save format
dataset = {}
dataset['categories'] = []
dataset['annotations'] = []
dataset['images'] = []
dataset['licences'] = []
dataset['info'] = {}



with open(os.path.join(root, classes_path)) as f:
    classes = f.read().strip().split('\n')
print("number of classes is: ", len(classes))

# build mapping between number and class
for i, cls in enumerate(classes):
    dataset['categories'].append({'id': i, 'name': cls, 'supercategory': 'mark'})

image_names = [f for f in os.listdir(os.path.join(root, images_dir))]
print("number of images is: ", len(image_names))


for image_i, image_name in enumerate(image_names):
    print(f"add {image_name}")
    img = cv2.imread(f"{root}/{images_dir}/{image_name}")
    height, width, _ = img.shape
    # add image information
    dataset['images'].append({
        'file_name': image_name,
        'id': image_i,
        'width': width,
        'height': height
    })


with open(save_json_filename, 'w') as f:
    json.dump(dataset, f)
