# get original picture from several captcha
import os
import numpy as np
from PIL import Image
import glob
from collections import Counter
from imutils import paths


def collect_images_numpy(imagepaths):
    for impath in imagepaths:
        img = Image.open(impath)
        data = np.array(img)
        yield data


def extract_images(datalist):
    d = datalist[0]
    try:
        row, col, channel = d.shape
        new_image = np.zeros((row, col, channel), dtype=np.uint8)
    except:
        row, col = d.shape
        new_image = np.zeros((row, col), dtype=np.uint8)
    
    for i in range(row):
        for j in range(col):
            dlist = [tuple(data[i][j]) for data in datalist]
            s, _ = Counter(dlist).most_common(1)[0]
            new_image[i][j] = s
    return new_image
    



if __name__ == '__main__':
    
    imagedir = '/home/jenny/Downloads/hk500_tag/lightg'
    imagepaths = paths.list_images(imagedir)

    datalist = list(collect_images_numpy(imagepaths))
    r = extract_images(datalist)
    new = Image.fromarray(r)
    new.save(f'{imagedir}/new.jpg')