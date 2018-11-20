import cv2
import random
from imutils import paths
from os.path import basename
import numpy as np
from uuid import uuid1


width1 = '/home/jenny/Downloads/hkcaptcha/sub/width1/'
width2 = '/home/jenny/Downloads/hkcaptcha/sub/width2/'
width3 = '/home/jenny/Downloads/hkcaptcha/sub/width3/'
image1 = list(paths.list_images(width1))
image2 = list(paths.list_images(width2))
image3 = list(paths.list_images(width3))


def gen(kind, savedir):
    # NOTE: captcha len is 4
    if kind == 1:
        pieces = [random.choice(image1) for _ in range(4)]
        codes = ''.join([basename(p)[:1] for p in pieces])
            
    elif kind == 2:
        pieces = [random.choice(image2) for _ in range(2)]
        codes = ''.join([basename(p)[:2] for p in pieces])

    elif kind == 3:
        p1, p3 = random.choice(image1), random.choice(image3)
        if random.randint(0, 1) == 0:
            codes = basename(p1)[0] + basename(p3)[:3]
            pieces = [p1, p3]
        else:
            codes = basename(p3)[:3] + basename(p1)[0]
            pieces = [p3, p1]

    elif kind == 4:
        pieces = [random.choice(image1) for _ in range(2)]
        p2 = random.choice(image2)
        codes = ''.join([basename(p)[0] for p in pieces]) + basename(p2)[:2]
        pieces.append(p2)
        
    merge(pieces, codes, savedir=savedir)
    

def merge(imagepaths, name, savedir=None):
    pieces = [cv2.imread(p) for p in imagepaths]
    merged = np.concatenate(pieces, axis=1)
    if savedir is not None:
        cv2.imwrite(f"{savedir}/{name}{uuid1()}.png", merged)
    return merged


if __name__ == '__main__':
    images_numbers = 10
    for i in range(images_numbers):
        kind = i % 4 + 1
        gen(kind, '/home/jenny/datasets/')