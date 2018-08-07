# get original picture from several captcha
import os
import numpy as np
from PIL import Image
import glob
from collections import Counter

imagedir = 'E:/captcha-data/img500/temp'
imagepaths = glob.glob(f'{imagedir}/*') 

def collect_images_numpy(imagepaths):
    for impath in imagepaths:
        img = Image.open(impath)
        data = np.array(img)
        yield data


def extract_images(datalist):
    d = datalist[0]
    try:
        row, col, channel = d.shape
        new_image = np.ndarray((row, col, channel), dtype=np.uint8)
    except:
        row, col = d.shape
        new_image = np.ndarray((row, col), dtype=np.uint8)
    
    for i in range(row):
        for j in range(col):
            dlist = [data[i][j] for data in datalist]
            slist = [sum(d) for d in dlist]
            s, _ = Counter(slist).most_common(1)[0]
            index = slist.index(s)
            new_image[i][j] = dlist[index]
    return new_image
    



if __name__ == '__main__':
    datalist = list(collect_images_numpy(imagepaths))
    r = extract_images(datalist)
    new = Image.fromarray(r)
    new.save(f'{imagedir}/new.jpg')