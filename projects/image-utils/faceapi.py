# auto label image using face++ API
import requests
import glob
import os
from uuid import uuid1

from PIL import Image
from utils import padding


api_key = '7zuC2S1YbMWr4Tcs7l1Igk6QB41mpQwj'
api_secret = 'TOP4TM1X-ITyjXTQBbzGHB0ByG_StWBc'

def char_detect(api_key, api_secret, image):
    url = f'https://api-cn.faceplusplus.com/imagepp/v1/recognizetext?api_key={api_key}&api_secret={api_secret}'
    files = {"image_file": ('dontcare', open(image, 'rb'), 'image/jpg')}
    r = requests.post(url, files=files)
    return r.json()


def pastex9(img):
    width, height = img.size
    imgx9 = padding(img, (height*3, width*3), True)
    angles = [-30, -60, -90, 90, 30, 60, 45, -45]
    xys = [
        [0, 0], [width, 0], [width*2, 0],
        [0, height], [width*2, height],
        [0, height*2], [width, height*2], [width*2, height*2],
    ]
    for angle, (x, y) in zip(angles, xys):
        aimg = img.rotate(angle, resample=Image.BICUBIC)
        imgx9.paste(aimg, (x, y, x+width, y+height))
    return imgx9


def pastex25(img):
    width, height = img.size
    imgx9 = padding(img, (height*5, width*5), True)
    angles = [
        -15, -30, -45, -60, -75, -90, -105, -120, -135, -150, -165, -180,
        15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180,
    ]
    xys = [
        [0, 0], [width, 0], [width*2, 0], [width*3, 0], [width*4, 0],
        [0, height], [width, height], [width*2, height], [width*3, height], [width*4, height],
        [0, height*2], [width, height*2], [width*3, height*2], [width*4, height*2],
        [0, height*3], [width, height*3], [width*2, height*3], [width*3, height*3], [width*4, height*3],
        [0, height*4], [width, height*4], [width*2, height*4], [width*3, height*4], [width*4, height*4],
    ]
    for angle, (x, y) in zip(angles, xys):
        aimg = img.rotate(angle, resample=Image.BICUBIC)
        imgx9.paste(aimg, (x, y, x+width, y+height))
    return imgx9



from collections import Counter
import string

def get_result(res):
    result = res['result']
    textlines = [r['child-objects'] for r in result if r['type'] == 'textline']
    charlist = [d for lst in textlines for d in lst]
    values = [char['value'] for char in charlist if char['value'] not in string.printable+'一']
    counter = Counter(values)
    clist = counter.most_common()
    return clist

def judge(clist, imgpath):
    answer = os.path.basename(imgpath)[:3]
    first_char = None
    first_count = None
    for (char, count) in clist:
        if char in answer:
            return char, 0
        if first_char is None:
            first_char = char
            first_count = count
    if first_count and first_count > 1:
        return first_char, 1
    return None, 2

if __name__ == '__main__':
    img = 'E:/captcha-data/images/20/1M73AE.png'
    r = char_detect(api_key, api_secret, img)