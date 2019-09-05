# example for usage of KeyPoint
# 本脚本假定ignore_mask已经生成了
import cv2
import numpy as np

from epc.keypoint import KeyPointTrain, KeyPointParams


if __name__ == '__main__':
    images_directory = '/data/minicoco/images'
    annotation_file = "/data/minicoco/annotations/mini_person_keypoints_val2014.json"
    dataset = KeyPointTrain(images_directory, annotation_file, KeyPointParams)
    dataset.plot_labels()