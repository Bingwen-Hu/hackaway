import os
import glob
from uuid import uuid1

import numpy as np
from PIL import Image
from skimage import morphology

import process
import crop


def rbg2gray():
    filepaths = glob.glob("E:/captcha-data/sina2/wb/*")
    for p in filepaths:
        img = Image.open(p).convert("L")
        img = process.pixel_replace(img, 140, 255, True)
        name = os.path.basename(p)
        img.save(f"temp/{name}")


def crop_and_save(path, pieces, savedir):
    """
    Args:
        path: 图片路径
        pieces:表示把图片切成几个部分，一般是验证码的个数
    """
    # 假定验证码已经标注好，且所有验证码的文本数都一样
    codes = os.path.basename(path)[:pieces]
    img = Image.open(path)
    vcrop_images = crop.multi_vertical_crop(img)

    if vcrop_images is None:
        return None
    # 纵切成功，开始横切
    hcrop_images = [crop.horizon_crop(im) for im in vcrop_images]

    # 按以文本作为开头命名
    for c, im in zip(codes, hcrop_images):
        im.save(os.path.join(savedir, f"{c}{uuid1()}.jpg"))
#
filepaths = glob.glob("E:/captcha-data/sina2/wb/*")

imglist = [Image.open(p).convert("L") for p in filepaths]
imglist = [process.pixel_replace(img, 140, 255, True) for img in imglist]
imglist = [process.pixel_replace(img, 1, 255, False) for img in imglist]
imgdict = {os.path.basename(p):img for (p, img) in zip(filepaths, imglist)}

def erosion_and_dilation(image):
    data = np.array(image)
    data = morphology.erosion(data)
    data = morphology.dilation(data)
    image = Image.fromarray(data)
    return image

def save(code, imagelist):
    for c, img in zip(code, imagelist):
        img = crop.horizon_crop(img)
        img.save(f"crops/{c}{uuid1()}.jpg")


def combine(imgdict):
    failed = {}
    for name, img in imgdict.items():
        code = name[:4]
        imagelist = crop.multi_vertical_crop(img)
        if imagelist is None:
            failed.update({name:img})
        else:
            save(code, imagelist)
    return failed

