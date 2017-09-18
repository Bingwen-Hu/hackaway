# -*- coding: utf-8 -*-
from PIL import Image, ImageDraw
from uuid import uuid1
import numpy as np
import os
import glob

from config import WIDTH, HEIGHT, DIR_IMAGE, DIR_SAVE



def crop_and_save(path, parts, savedir):
    """
    path: 图片路径
    parts:表示把图片切成几个部分，一般是验证码的个数
    """
    codes = os.path.basename(path)[:parts]
    img = Image.open(path)

    vcrop_images = multi_vertical_crop(img)

    if vcrop_images is None:
        return None
    hcrop_images = [horizon_crop(im) for im in vcrop_images]

    # save it
    for c, im in zip(codes, hcrop_images):
        im.save(os.path.join(savedir, f"{c}{uuid1()}.jpg"))


def multi_vertical_crop(img, pieces=3):
    width, height = img.size
    data = np.array(img)

    # cup by columns
    points = [i for i in range(width) if np.sum(data[:, i]) > 0]
    cutpoints = [i for i in range(len(points)-1) if points[i]+1 != points[i+1]]
    if len(cutpoints) != pieces:
        print("image has something unfit")
        return None
    i, j, k = cutpoints
    cutpoints = ((points[0], points[i]), (points[i+1], points[j]),
                 (points[j+1], points[k]), (points[k+1], points[-1]))

    imagelist = []
    for start, end in cutpoints:
        imagelist.append(img.crop((start, 0, end, height)))
    return imagelist

def all_crop(img):
    img_ = horizon_crop(img)
    img_ = vertical_crop(img_)
    return img_

def horizon_crop(img):
    width, height = img.size
    data = np.array(img)

    #cup by rows
    points = [i for i in range(height) if np.sum(data[i, :]) < 255 * width]
    start, end = points[0], points[-1]
    img_ = img.crop((0, start, width, end+1))
    return img_

def vertical_crop(img):
    width, height = img.size
    data = np.array(img)

    #cut by columns
    points = [i for i in range(width) if np.sum(data[:, i]) < 255 * height]
    start, end = points[0], points[-1]
    img_ = img.crop((start, 0, end+1, height))
    return img_


def random_select(dirpath, num):
    paths = glob.glob(dirpath+"*")
    np.random.shuffle(paths)
    return paths[:num]

### dwnews ###
def image_paste_dwnews(dirpath, mode, num, savepath=None):
    image = Image.new(mode, (WIDTH, HEIGHT))
    paths = random_select(dirpath, num)
    codes = [os.path.basename(p)[0] for p in paths]
    images = [Image.open(p) for p in paths]

    randint = np.random.randint(10)
    h_offset = 8
    for c, p in zip(codes, images):
        v_offset = 25 if c in "bdfhijklt" else 35
        v_offset -= randint
        w, h = p.size
        image.paste(p, [h_offset, v_offset, w+h_offset, h+v_offset])
        h_offset += w + 4

    if savepath is not None:
        code = ''.join(codes)
        savepath = os.path.join(savepath, f"{code}{uuid1()}.jpg")
        image.save(savepath)
    else:
        return image

def image_paste_sina(dirpath, mode, num, savepath=None):
    image = Image.new(mode, (WIDTH, HEIGHT), color=255)
    paths = random_select(dirpath, num)
    codes = [os.path.basename(p)[0] for p in paths]
    images = [Image.open(p) for p in paths]

    h_offset = np.random.randint(5, 15)
    v_offset = np.random.randint(5, 15)

    for c, p in zip(codes, images):
        w, h = p.size
        image.paste(p, [h_offset, v_offset, w+h_offset, h+v_offset])
        h_offset += w + 1

    # rotate
    angle = np.random.randint(-10, 10)
    image = image.rotate(angle, Image.BICUBIC)
    image = remove_dark(image)

    # add_acr
    image = add_arc(image)

    if savepath is not None:
        code = ''.join(codes)
        savepath = os.path.join(savepath, f"{code}{uuid1()}.png")
        image.save(savepath)
    else:
        return image

def remove_dark(img):
    data = np.array(img)
    data_ = np.where(data==0, 255, data)
    img_ = Image.fromarray(data_)
    return img_

def add_arc(img):
    dr = ImageDraw.Draw(img)
    randint = np.random.randint(0, 4)
    if randint == 0:
        dr.arc(((3,3), (90, 35)), 50, -200, fill=118)
    elif randint == 1:
        dr.arc(((3,3), (90, 35)), -120, -50, fill=118)
    elif randint == 2:
        dr.line(((0, 20), (100, 20)), width=2, fill=118)
    return img