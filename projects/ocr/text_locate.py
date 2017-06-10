# -*- coding: utf-8 -*-
"""
Created on Mon May 15 09:35:32 2017

@author: Administrator

Locate the text in an image!
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
from skimage import feature, measure
from skimage.morphology import dilation
from skimage.color import rgb2gray
# 以“进”为标杆
Character_Len = 36
Character_Wid = 27

Charactor_ad_Len = 24
Charactor_ad_Wid = 24

def logic2num(img):
    """
    将阈值逻辑图像转换为0，1图像
    """
    img2 = np.where(img==True, 1, 0)
    return img2

def lineFilter(img, length=Character_Len):
    """
    逐行遍历，遇到0时，检查pos是否有值，检查cnt是否大于指定长度，如是，
    清空[pos, pos+cnt]之间的连线，重置pos和cnt，否则只重置pos和cnt
    遇到1时，第一次的位置由pos记录，cnt每次增1
    """
    for i, row in enumerate(img):
        cnt = 0
        pos = None
        for index, j in enumerate(row):
            if j == 0: # 一开始或者已经有积累
                if (cnt > Character_Len) and (pos is not None):
                    img[i, pos:pos+cnt] = 0
                pos = None
                cnt = 0
            else:
                if cnt == 0:
                    pos = index # 记下非0位置
                cnt += 1
    return img
    

def ratioFilter(img, ratio=0.5, width=12, factor=1, iters=1):
    """
    利用字幕连续的性质，将有大量空白的地方过滤掉。对于比较平滑的图片效果较好
    """
    def loop(img, factor):        
        _, length = img.shape
        sum_ = np.sum(img, axis=1)
        mean_ = np.mean(img) * length * factor
        for i, s in enumerate(sum_):
            if s < mean_:
                img[i, :] = 0
        return img
    
    for i in range(iters):
        img_r = loop(img, factor)
        img = img_r
    return img



def pixelFileter(img, threshold=(200,200,200)):
    """
    使用文本像素点为白色的先验知识
    """
    img_ = np.where(img<threshold, (0, 0, 0), img)
    img_ = np.array(img_, dtype='uint8', copy=False)
    return img_

def locateText(img, minwidth=10, start_row=0):
    """
    非通用算法，只是针对性的字幕获取方法
    """
    #img = np.pad(img, ((0, 1), (0, 1)), 'minimum')
    num_row, num_col = img.shape
    index_row = start_row                               # 行索引
    index_col = 0                                       # 列索引
    
    top = None
    bottom = None
    left = None
    right = None
    
    
    for index_row in range(start_row, num_row):
        row = img[index_row, :]
        if not np.any(row) or index_row == num_row-1:  # 为空行或已触底
            if top is None:                         # 目前为止没有搜索到点
                continue                                # 换行
            elif bottom is None: 
                bottom = index_row                  # 将有行的索引作为底线
                if bottom - top < minwidth:     # 不满足要求
                    top = None                      # 重新初始化
                    bottom = None
                    continue                            # 换列
                # 满足要求，开始对top和bottom之间的区域进行列遍历
                for index_col in range(num_col):
                    column = img[top:bottom, index_col]
                    if not np.any(column) or index_col == num_col-1: # 为空列或已触底
                        if left is None:            # 目前为止没有搜索到点
                            continue                    # 换列
                        else:                           # pos_left有值
                            rect = img[top:bottom, index_col:]
                            # 后面有数据，且还没有触底
                            if np.any(rect) and index_col != num_col-1:
                                continue                # 换列
                            else:                       # 已经没数据了
                                right = index_col   # 将有列的索引作为右线
                                break                   # 搜索完成
                    else:
                        if left is None:            # 如果第一个非空列
                            left = index_col        # 记下列的位置
            else:
                break                                   # 搜索完成
        else:                                           # 不是空行
            if top is None:                         # 如果第一个非空行
                top = index_row                     # 记下行的位置
            # 否则什么也不做，继续行遍历
    return [top, bottom, 
            left, right] # 触边

def multi_locate(img, times=1, start_row=0):
    results = []
    for i in range(times):
        rect = locateText(img, start_row=start_row)
        start_row = rect[1]
        results.append(rect)
    return results


def crop(img, times=1):
    img = np.array(img)
    img_ = img
    
    img = pixelFileter(img)
    img = rgb2gray(img)
    img = feature.canny(img, sigma=2.8)
    img = logic2num(img)
    img = dilation(img)
    img = lineFilter(img)
    img = ratioFilter(img, factor=3)

    results = multi_locate(img, times=times)
    for (top, bottom, left, right) in results:
        plt.imshow(img_[top:bottom, left:right])
    

if __name__ == '__main__':
    img = Image.open("resources/pic/gzsc.mp4_32.jpg")
    plt.imshow(img)
    
    img = np.array(img)
    
    img = pixelFileter(img)
    plt.imshow(img)
    
    img = rgb2gray(img)
    plt.imshow(img)
    
    img = feature.canny(img, sigma=2.8)
    img = logic2num(img)
    plt.imshow(img)

    
    img = dilation(img)
    plt.imshow(img)

    img = lineFilter(img)
    plt.imshow(img)

    img = ratioFilter(img, factor=3)
    plt.imshow(img)
    