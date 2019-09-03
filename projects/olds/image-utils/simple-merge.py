import os.path as osp
import random
from uuid import uuid1


import cv2
import numpy as np
from imutils import paths



def select_pieces(pieces, num):
    croppieces = random.choices(pieces, k=num)
    names = [osp.basename(piece) for piece in croppieces]
    chars = [name[0] for name in names]
    code = ''.join(chars)
    return croppieces, code
    

def merge(croppieces, name, savedir=None):
    pieces = [cv2.imread(p) for p in croppieces]
    merged = np.concatenate(pieces, axis=1)
    if savedir is not None:
        path = osp.join(savedir, f"{name}-{uuid1()}.png")
        cv2.imwrite(path, merged)
    return merged


if __name__ == '__main__':
    directory = 'pieces'
    pieces = list(paths.list_images(directory))
    images_numbers = 10
    for i in range(images_numbers):
        croppieces, name = select_pieces(pieces, num=5)
        merge(croppieces, name, 'gen')
