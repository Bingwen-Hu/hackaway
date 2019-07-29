# example for usage of KeyPoint
# 本脚本假定ignore_mask已经生成了

import sys
sys.path.append('..')

from fireu.data import coco


if __name__ == '__main__':
    images_directory = '/data/minicoco/images'
    annotation_file = "/data/minicoco/annotations/mini_person_keypoints_val2014.json"
    dataset = coco.KeyPoint(images_directory, annotation_file, coco.KeyPointParams)

    for i in range(len(dataset)):
        im_id, ann_metas, im = dataset.get_im_with_anns(index=i)
        mask = dataset.get_ignore_mask(im_id)
        poses = dataset.convert_joint_order(ann_metas)
        im, mask, heatmaps, pafs = dataset.generate_labels(im, mask, poses)
        