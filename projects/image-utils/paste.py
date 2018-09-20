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


if __name__ == '__main__':
    import glob 
    import random
    
    heads = glob.glob('E:/captcha-data/images/new500/crops/heads/*.png')
    tails = glob.glob('E:/captcha-data/images/new500/crops/tails/*.png')
    
    for i in range(100000):
        pieces = random_select_pieces(heads, 5)
        tail = random.choice(tails)
        pieces.append(tail)
        random.shuffle(pieces)
        image = sogou_paste(pieces, "E:/captcha-data/sogou/rgen4/")