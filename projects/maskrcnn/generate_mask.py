import os
import glob
import random

import numpy as np
from PIL import Image


def random_select_pieces(piecepaths, num):
    random.shuffle(piecepaths)
    return piecepaths[:num]

def paste(mode, piecepaths, size, bgcolor=255, savepath=None):
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

    h_offset = random.randint(15, 20)
    v_offset = random.randint(5, 10)
    masks = []
    for c, p in zip(codes, pieces):
        w, h = p.size
        bbox = [h_offset, v_offset, w+h_offset, h+v_offset]
        image.paste(p, bbox)
        mask = generate(image, p, bbox)
        masks.append(mask)
        h_offset = h_offset + w + 8


    return image.convert("RGB"), masks, ''.join(codes)

def generate(image, piece, bbox):
    width, height = image.size
    mask = np.zeros((height, width))
    data = np.array(piece)
    minimask = np.where(data<=140, 1, 0)
    w_start, h_start, w_end, h_end = bbox
    mask[h_start:h_end, w_start:w_end] = minimask
    return mask

def align_mask(masks):
    pass



if __name__ == '__main__':
    all_pieces = glob.glob("data/*")
    three_pieces = random_select_pieces(all_pieces, 3)
    img, masks, codes = paste("L", three_pieces, (100, 100))
