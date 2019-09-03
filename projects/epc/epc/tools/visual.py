import cv2
import numpy as np

from pycocotools import coco


def plot_keypoints(im, keypoints, ngroup, output):
    """General function to plot keypoint of COCO images

    Args:
        im: image return by cv2.imread
        keypoints: comes from COCO annotation's `keypoint` field    
        ngroup: number of point group in keypoints. In COCO is 17
        output: filename to write out.
    """
    parts = np.split(np.array(keypoints), ngroup)
    for (i, part) in enumerate(parts):
        x, y = part[:2]
        cv2.circle(im, (x, y), radius=2, color=(0, 0, 255))
        cv2.putText(im, str(i), (x-1,y-1), fontFace=cv2.FONT_HERSHEY_PLAIN, 
            fontScale=0.5, color=(0, 255, 0), thickness=1)
    cv2.imwrite(f"{output}.png", im)
