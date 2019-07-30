# example for usage of KeyPoint
# 本脚本假定ignore_mask已经生成了

import sys
sys.path.append('..')

import cv2
import numpy as np

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
        

        # resize to view
        shape = (dataset.params.insize, dataset.params.insize)
        pafs = cv2.resize(pafs.transpose(1, 2, 0), shape).transpose(2, 0, 1)
        heatmaps = cv2.resize(heatmaps.transpose(1, 2, 0), shape).transpose(2, 0, 1)
        mask = cv2.resize(mask.astype(np.uint8) * 255, shape) == 255

        # overlay labels
        img_to_show = im.copy()
        img_to_show = dataset.overlay_PAFs(img_to_show, pafs)
        img_to_show = dataset.overlay_heatmap(img_to_show, heatmaps[:-1].max(axis=0))
        img_to_show = dataset.overlay_ignore_mask(img_to_show, mask)
        # cv2.imwrite("resized_img.png", resized_img)
        # cv2.imwrite('img_to_show.png', img_to_show)
        cv2.imshow('w', np.hstack((im, img_to_show)))
        k = cv2.waitKey(0)
        if k == ord('q'):
            sys.exit()
