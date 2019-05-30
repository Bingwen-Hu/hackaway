import json
import os
import cv2

root = "/media/data/urun_tandong_video/data/signal"
classes_path = 'classes.txt'
images_dir = 'train'
raw_annotations = 'train_label_fix.csv'
save_json_filename = "signal.json"

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
    dataset['categories'].append({'id': i, 'name': cls, 'supercategory': 'mark'}) # value of supercategory varies to your problem

image_names = [f for f in os.listdir(os.path.join(root, images_dir))]
print("number of images is: ", len(image_names))


with open(os.path.join(root, raw_annotations)) as f:
    annos = f.read().strip().split('\n')
    annos = annos[1:] # skip the first line, which contains name of fields
print("number of annotations is: ", len(annos))


for image_i, image_name in enumerate(image_names):
    img = cv2.imread(f"{root}/{images_dir}/{image_name}")
    height, width = img.shape[:2]
    # add image information
    dataset['images'].append({
        'file_name': image_name,
        'id': image_i,
        'width': width,
        'height': height
    })

    for anno_i, anno in enumerate(annos):
        img_name,x1,y1,x2,y2,x3,y3,x4,y4,cls_ = anno.strip().split(',')
        # match
        if image_name == img_name:
            print("Processing image {}, annotations {}".format(image_i, anno_i))
            width, height = int(x3)-int(x1), int(y3)-int(y1)
            dataset['annotations'].append({
                'area': width * height,
                'bbox': [int(x1), int(y1), width, height],
                'category_id': int(cls_),
                'id': anno_i,
                'image_id': image_i,
                'image_name': image_name,
                'iscrowd': 0,
                'segmentation': list(map(int, [x1,y1,x2,y2,x3,y3,x4,y4])),
            })

with open(save_json_filename, 'w') as f:
    json.dump(dataset, f)
