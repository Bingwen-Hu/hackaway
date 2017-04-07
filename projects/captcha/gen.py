# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 10:33:42 2017

@author: Administrator
"""

from captcha.image import ImageCaptcha  # pip install captcha  
import numpy as np
from PIL import Image  
import random

number = ['0','1','2','3','4','5','6','7','8','9']
ALPHABET = ['A','B','C','D','E','F','G','H','I','J','K','L','M',
            'N','O','P','Q','R','S','T','U','V','W','X','Y','Z']

def random_captcha_text(char_set=number+ALPHABET, captcha_size=5):  
    captcha_text = []  
    for i in range(captcha_size):  
        c = random.choice(char_set)  
        captcha_text.append(c)  
    return captcha_text

def gen_captcha_text_and_image():  
    image = ImageCaptcha()  
   
    captcha_text = random_captcha_text()  
    captcha_text = ''.join(captcha_text)  
   
    captcha = image.generate(captcha_text)  
    #image.write(captcha_text, captcha_text + '.jpg')  # 写到文件  
   
    captcha_image = Image.open(captcha)  
    captcha_image = np.array(captcha_image)  
    return captcha_text, captcha_image

def filter_dataset(data):
    newdata = [d for d in data if d[1].shape==(60, 160, 3)]
    return newdata


def gen_dataset(nums=10000):
    data = [gen_captcha_text_and_image() for i in range(nums)]
    data = filter_dataset(data)
    y, X = zip(*data)
    return X, y