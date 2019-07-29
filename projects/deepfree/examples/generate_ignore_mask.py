# 在姿态估计中，如何使用本库来生成mask
import sys
sys.path.append('..')

import cv2
import numpy as np


from fireu.data import coco



if __name__ == "__main__":
    images_directory = '/data/minicoco/images'
    annotation_file = "/data/minicoco/annotations/mini_person_keypoints_val2014.json"
    dataset = coco.KeyPoint(images_directory, annotation_file, coco.KeyPointParams)

    for i in range(len(dataset)):
        im_id, ann_metas, im = dataset.get_im_with_anns(index=i)
        _, mask = dataset.make_ignore_mask(im.shape[:2], ann_metas)

        # visual
        im_origin = dataset.plot_masks_and_keypoints(im, ann_metas)
        im_train = dataset.plot_ignore_mask(im, mask)
        cv2.imshow("image", np.hstack([im_origin, im_train]))
        k = cv2.waitKey()
        if k == ord('q'):
            break

        dataset.save_ignore_mask(im_id, mask)