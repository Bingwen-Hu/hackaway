"convert RGB image to RGBA"

import cv2
import numpy as np


def BGR2BGRA(img, alpha=0.5):
    b, g, r = cv2.split(img)

    alpha_channel = np.ones(b.shape, dtype=b.dtype) * 128

    img_BGRA = cv2.merge((b, g, r, alpha_channel))
    return img_BGRA


if __name__ == "__main__":
    img = cv2.imread('/home/mory/buddha/cloud.jpeg')
    img_ = BGR2BGRA(img)
    cv2.imwrite('/home/mory/buddha/cloud_a.jpeg', img_)
