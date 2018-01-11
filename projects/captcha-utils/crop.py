# -*- coding: utf-8 -*-
import numpy as np




def multi_vertical_crop(img, pieces=3):
    """按列对图片进行切割。首先找出目标片段所在的列，这个需要按需调整筛选条件。
    然后，筛选出边界点，即cutpoints。因为边界点的下一点即是新的片段的起始点，
    所以有了pieces个边界点后，就可以进行切割了。

    Args:
        img: PIL.Image object
        pieces: number of pieces

    Returns:
        list of PIL.Image object
    """
    width, height = img.size
    data = np.array(img)

    # 以黑点(<140)的数目大于2作为分界条件
    points = [i for i in range(width) if np.sum(data[:, i] < 140) > 0]
    # 找出边界点
    cutpoints = [i for i in range(len(points)-1) if points[i]+1 != points[i+1]]
    if len(cutpoints) != pieces:
        print("image has something unfit")
        return None
    i, j, k = cutpoints
    # 边界点的下一点即是新片段的起始点
    cutpoints = ((points[0], points[i]), (points[i+1], points[j]),
                 (points[j+1], points[k]), (points[k+1], points[-1]))

    imagelist = []
    for start, end in cutpoints:
        # end+1是因为crop不包含边界，需要+1包含
        imagelist.append(img.crop((start, 0, end+1, height)))
    return imagelist


def horizon_crop(img):
    """按行切割，规则按需设置"""
    width, height = img.size
    data = np.array(img)

    # 这里设置的规则是一行中黑点(<140)不小于2个
    points = [i for i in range(height) if np.sum(data[i, :] < 140) >= 1]
    start, end = points[0], points[-1]
    # +1 保留最后一行
    img_ = img.crop((0, start, width, end+1))
    return img_


def vertical_crop(img):
    """按列剪切，简版，详细文档见multi_vertical_crop"""
    width, height = img.size
    data = np.array(img)
    points = [i for i in range(width) if np.sum(data[:, i]) < 255 * height]
    start, end = points[0], points[-1]
    img_ = img.crop((start, 0, end+1, height))
    return img_

