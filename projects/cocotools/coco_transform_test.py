import json
import os
import cv2

root = "/root/mmdetection/data/signal/"
root = "/media/data/urun_tandong_video/data/signal/"

# final save format
dataset = {}
dataset['categories'] = []
dataset['annotations'] = []
dataset['images'] = []
dataset['licences'] = []
dataset['info'] = {}



with open(os.path.join(root, 'classes.txt')) as f:
    classes = f.read().strip().split('\n')
print("number of classes is: ", len(classes))

# build mapping between number and class
for i, cls in enumerate(classes):
    dataset['categories'].append({'id': i, 'name': cls, 'supercategory': 'mark'})

image_names = [f for f in os.listdir(os.path.join(root, '0'))]
print("number of images is: ", len(image_names))


# with open(os.path.join(root, 'train_label_fix.csv')) as f:
#    annos = f.read().strip().split('\n')
#    annos = annos[1:]

# print("number of annotations is: ", len(annos))


for image_i, image_name in enumerate(image_names):
    print(f"add {image_name}")
    img = cv2.imread(f"{root}/test/{image_name}")
    height, width, _ = img.shape
    # add image information
    dataset['images'].append({
        'file_name': image_name,
        'id': image_i,
        'width': width,
        'height': height
    })

json_name = "test_all.json"
with open(json_name, 'w') as f:
    json.dump(dataset, f)
