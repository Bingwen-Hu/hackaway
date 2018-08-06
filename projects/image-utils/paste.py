import numpy as np
import os
from PIL import Image



def random_select_pieces(piecepaths, num):
    np.random.shuffle(piecepaths)
    return piecepaths[:num]

def paste(mode, piecepaths, size, savepath=None):
    """将几个片段拼成一个文件
    Args:
        mode: 图片的模式, RGB, L等
        piecepaths: 片段的路径
        size: 生成图片的大小
        savepath: 保存图片的路径，当提供时，保存图片

    Returns:
        新的PIL.Image对象
    """
    image = Image.new(mode, size, color=255)
    codes = [os.path.basename(path) for path in piecepaths]
    pieces = [Image.open(path) for path in piecepaths]

    h_offset = np.random.randint(15, 20)
    v_offset = np.random.randint(5, 10)

    for c, p in zip(codes, pieces):
        w, h = p.size
        image.paste(p, [h_offset, v_offset, w+h_offset, h+v_offset])
        h_offset = h_offset + w + 8

    if savepath is not None:
        code = ''.join(codes)
        savepath = os.path.join(savepath, f"{code}{uuid1()}.jpg")
        image.save(savepath)

    return image