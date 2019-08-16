import os
import cv2
from uuid import uuid1
from imutils import paths


def gen_xy(imgpath):
    width = cv2.imread(imgpath).shape[1]
    # first set, into four
    size = width // 4
    xy_1 = [i*size for i in range(4)]
    xy_1.append(width)
    xy_1 = [(s, e) for (s, e) in zip(xy_1, xy_1[1:])]
    # second set, into two part by middle
    size = width // 2
    xy_2 = [i*size for i in range(2)]
    xy_2.append(width)
    xy_2 = [(s, e) for (s, e) in zip(xy_2, xy_2[1:])]
    # third set, into two part by first part and reset
    size = width // 4
    xy_3 = [(0, size), (size, width - size),
            (0, width - size), (width - size, width)]
    crops_xy = []
    crops_xy.extend(xy_1)
    crops_xy.extend(xy_2)
    crops_xy.extend(xy_3)
    return crops_xy


def crop(imgpath, crops_xy):
    """separate a image (150x60) into 
    four pieces"""
    img = cv2.imread(imgpath)
    for (s, e) in crops_xy:
        crop = img[:, s:e]
        cv2.imwrite("/home/jenny/datasets/img/sub/{}.png".format(uuid1()), crop)



if __name__ == "__main__":
    path = '/home/jenny/datasets/img/'
    images = list(paths.list_images(path))
    crops_xy = gen_xy(images[0])
    for i, f in enumerate(images):
        crop(f, crops_xy)
        
