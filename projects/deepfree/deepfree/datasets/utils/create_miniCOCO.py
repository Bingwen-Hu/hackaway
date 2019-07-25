# script to create mini COCO dataset for convenient Test
import os
import os.path as osp
import json
import random

from pycocotools import coco


def create_miniCOCO_keypoints(
        annotations_file, 
        images_directory, 
        number=1000,
        output_directory="mini"):
    """Create a mini version of MS COCO dateset for pose estimation

    Args:
        annotations_file: path to annotation file, like `person_xxx.json`
        images_directory: path, usually as COCO/images or COCO/images/val2014
        number: integer, number of images in new datasets
        output_directory: where the new datasets place
    """
    dataset = coco.COCO(annotations_file)
    image_ids = list(dataset.imgs.keys())
    keys = random.choices(image_ids, k=number)

    print('Create miniCOCO annotations file...') 
    miniCOCO = {}
    # coco.COCO.dataset is the original json file
    miniCOCO['licenses'] = dataset.dataset['licenses']
    miniCOCO['info'] = dataset.dataset['info']
    miniCOCO['categories'] = dataset.dataset['categories']
    images = dataset.loadImgs(keys)
    annotation_ids = dataset.getAnnIds(keys)
    annotations = dataset.loadAnns(annotation_ids)
    miniCOCO['annotations'] = annotations
    miniCOCO['images'] = images
    # create target directory
    output_annotations = osp.join(output_directory, 'annotations')
    output_images = osp.join(output_directory, 'images')
    os.makedirs(output_annotations, exist_ok=False)
    os.makedirs(output_images, exist_ok=False)
    basename = f"mini_{osp.basename(annotations_file)}"
    miniCOCO_filename = osp.join(output_directory, basename)
    with open(miniCOCO_filename, 'w') as f:
        json.dump(miniCOCO, f)
    print("Done!")

    print("Copy images into new directory...")
    image_files = [image['file_name'] for image in images]
    for image_file in image_files:
        image_path = osp.join(images_directory, image_file)
        os.system(f"cp {image_path} {output_images}/")
    print("Done!")
        