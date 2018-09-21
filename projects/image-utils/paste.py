import numpy as np
import os
from PIL import Image
from uuid import uuid1


def random_select_pieces(piecepaths, num):
    np.random.shuffle(piecepaths)
    return piecepaths[:num]


def paste(mode, bgcolor, piecepaths, size, savepath=None):
    """将几个片段拼成一个文件
    Args:
        mode: 图片的模式, RGB, L等
        piecepaths: 片段的路径
        size: 生成图片的大小
        savepath: 保存图片的路径，当提供时，保存图片

    Returns:
        新的PIL.Image对象
    """
    image = Image.new(mode, size, color=bgcolor)
    codes = [os.path.basename(path)[0] for path in piecepaths]
    pieces = [Image.open(path) for path in piecepaths]


    h_offset = np.random.randint(0, 10)
    v_offset = np.random.randint(1, 10)

    for c, p in zip(codes, pieces):            
        h_ = np.random.randint(0, 7)
        v_ = np.random.randint(-1, 3)
        w, h = p.size
        image.paste(p, [h_offset, v_offset, w+h_offset, h+v_offset])
        h_offset = h_offset + w + h_
        v_offset = v_offset + v_

    if savepath is not None:
        code = ''.join(codes)
        savepath = os.path.join(savepath, f"{code}{uuid1()}.jpg")
        image.save(savepath)

    return image


def sogou_paste(piecepaths, savepath=None):
    image = Image.new('RGB', (140, 44), color=(170, 170, 170))
    codes = [os.path.basename(path)[0] for path in piecepaths]
    pieces = [Image.open(path) for path in piecepaths]

    x = 0
    for p in pieces:
        w, _ = p.size
        image.paste(p, [x, 0, x+w, 44])
        x = x + w

    if savepath is not None:
        code = ''.join(codes)
        savepath = os.path.join(savepath, f"{code}{uuid1()}.jpg")
        image.save(savepath)

    return image

def sogou_paste2(piecepaths, savepath=None):
    image = Image.new('RGB', (140, 44), color=(170, 170, 170))
    codes = [os.path.basename(path)[:clen] for (clen, path) in piecepaths]
    pieces = [Image.open(path) for (clen, path) in piecepaths]

    x = 0
    for p in pieces:
        w, _ = p.size
        image.paste(p, [x, 0, x+w, 44])
        x = x + w

    if savepath is not None:
        code = ''.join(codes)
        savepath = os.path.join(savepath, f"{code}{uuid1()}.jpg")
        image.save(savepath)

    return image


if __name__ == '__main__':
    import glob 
    import random
    
    w1_c1 = glob.glob('E:/captcha-data/images/new81/sogou200/1/1/*.png')
    
    w2_c1 = glob.glob('E:/captcha-data/images/new81/sogou200/2/1/*.png')
    w3_c4 = glob.glob('E:/captcha-data/images/new81/sogou200/3/4/*.png')
    
    w2_c3 = glob.glob('E:/captcha-data/images/new81/sogou200/2/3/*.png')
    w3_c2 = glob.glob('E:/captcha-data/images/new81/sogou200/3/2/*.png')

    w2_c2 = glob.glob('E:/captcha-data/images/new81/sogou200/2/2/*.png')
    w3_c3 = glob.glob('E:/captcha-data/images/new81/sogou200/3/3/*.png')
    
    combines = [
        [w1_c1, w2_c1, w3_c4],
        [w1_c1, w2_c2, w3_c3],
        [w1_c1, w2_c3, w3_c2],
    ]
    modes = [
        [1, 1, 4],
        [1, 2, 3], 
        [1, 3, 2],
    ]

    for i in range(100000):
        r = np.random.randint(0, 10)
        if r <= 1:
            r = 0
        elif 1 < r <= 8:
            r = 1
        elif 8 < r <= 10:
            r = 2
        combine = combines[r]
        mode = modes[r]
        dictpaths = [(m, random.choice(lst)) for (m, lst) in zip(mode, combine)]
        random.shuffle(dictpaths)
        image = sogou_paste2(dictpaths, "E:/captcha-data/sogou/rgen6/")