import cv2
import numpy as np


def plot_keypoints(im, keypoints, ngroup):
    parts = np.split(np.array(keypoints), ngroup)
    for part in parts:
        x, y = part[:2]
        cv2.circle(im, (x, y), radius=2, color=(0, 0, 255))
    cv2.imwrite("visual.png", im)