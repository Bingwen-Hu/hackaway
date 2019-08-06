# 在姿态估计中，如何使用本库来生成mask
import cv2
import argparse
import numpy as np


from fireu.data import coco



def get_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--vis', action='store_true', help="whether to visual or not")
    args = argparser.parse_args()
    return args

if __name__ == "__main__":
    # 命令行参数
    args = get_args()
    # 数据集路径，根据实际的路径进行设置
    images_directory = '/data/minicoco/images'
    annotation_file = "/data/minicoco/annotations/mini_person_keypoints_val2014.json"
    # 初始化数据集
    dataset = coco.KeyPoint(images_directory, annotation_file, coco.KeyPointParams)

    for i in range(len(dataset)):
        im_id, ann_metas, im = dataset.get_im_with_anns(index=i)
        _, mask = dataset.make_ignore_mask(im.shape[:2], ann_metas)

        # visual
        if args.vis:
            im_origin = dataset.plot_masks_and_keypoints(im, ann_metas)
            im_train = dataset.plot_ignore_mask(im, mask)
            cv2.imshow("image", np.hstack([im_origin, im_train]))
            k = cv2.waitKey()
            if k == ord('q'):
                break
        else:
            dataset.save_ignore_mask(im_id, mask)