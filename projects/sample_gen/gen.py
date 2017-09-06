# -*- coding: utf-8 -*-
"""
识别字体比例: 93x160
"""

import os
from PIL import Image, ImageFont, ImageDraw
import uuid
import numpy as np
import matplotlib.pyplot as plt

from config import *





def random_text(charset=CHARSET, num=CHAR_NUM):
    index = [np.random.randint(len(charset)) for i in range(num)]
    text = [charset[i] for i in index]
    return ''.join(text)

def show_image(text, font):
    image = Image.new("RGB", (WIDTH, HEIGHT), (0, 0, 0))
    dr = ImageDraw.Draw(image)

    h_offset = 8
    v_offset = 10
    delta = 0
    for t in text:
        dr.text((h_offset+delta, v_offset), t, font=font)
        delta += int(CHAR_SIZE/1.7)
        if t in 'wm': delta += 17
        if t in 'filr': delta -= 12
    angle = np.random.randint(-5, 5)

    image = image.rotate(angle)

    image.save(f"./data/{text}{uuid.uuid1()}.jpg")

if __name__ == '__main__':
    # config
    font = ImageFont.truetype(FONT, CHAR_SIZE)

    for i in range(100000):
        text = random_text(CHARSET, CHAR_NUM)
        show_image(text, font)
#    word1 = codecs.open("wordset/word1.txt", 'r', 'utf-8').read()
#    punct = codecs.open("wordset/punctuation.txt").read()
#    chars = punct

#    text = '警'
#    size = [20, 24, 28, 32, 36, 40, 50, 56, 62, 68, 74, 80, 86]
#    fonts = os.listdir("./fonts")
#    size_ = size[-1]
#    font = fonts[1]
#    gen_image(text, size_, font)

#    font = "E:/Mory/gogs/tensorflow-ocr/fonts/bold/msyhbd.ttf"

#    dirname = os.path.basename(font)[:-4]
#    size = 20
#    dir = f"data/{dirname}/punct/{size}/"
#
#    for w in chars:
#        test_gen_image(w, size, font, dir)
#        break
#    gen_image(chars, size, font, dir)
#==============================================================================
# special
#==============================================================================
#    font = "E:/Mory/gogs/tensorflow-ocr/fonts/sim/FanZhenCuShongJianTi.ttf"
#    size = 26
#    dir = f"data/special/{size}/"
#    downs = "﹄﹂"
#
#    gen_image(downs, size, font, dir, 0, 5)
#
#
#    ups = "﹃﹁"
#    gen_image(ups, size, font, dir, 0, -10)
