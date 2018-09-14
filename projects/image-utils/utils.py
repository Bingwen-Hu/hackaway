""" utility for image padding to a certain size, rotate or flip """
from PIL import Image, ImageDraw, ImageFilter
import numpy as np



def resize(img, size):
    img = img.resize(size, Image.BICUBIC)
    return img

def padding(img, size, RGB):
    """
    Args:
        img: PIL.Image object
        size: target  (height, width)
        RGB: boolean, whether in RGB format or not
    
    Returns:
        PIL.Image object with margin padding
    """
    data = np.array(img)

    if RGB:
        rows, cols, _ = data.shape
        const = (0, 0, 0)
    else:
        rows, cols = data.shape
        const = 0
    
    height, width = size
    assert height >= rows, "padding size must larger than original size"

    top = (height - rows) / 2
    left = (width - cols) / 2
    right = int(left)
    bottom = int(top)
    if type(left) != type('int'):
        left = right + 1
    if type(top) != type('int'):
        top = bottom + 1
    if RGB:
        padding_tuple = ((top, bottom), (left, right), (0, 0))
    else:
        padding_tuple = ((top, bottom), (left, right))
    newdata = np.pad(data, padding_tuple, mode='constant')
    return Image.fromarray(newdata)


def rotate(img, angle):
    img = img.rotate(angle, resample=Image.BICUBIC)
    return img

def blur(img):
    return img.filter(ImageFilter.SMOOTH)

def pixel_replace(image, threshold, replace, larger=False):
    """replace certain pixel
    Args:
        image: PIL.Image object
        threshold: threshold value
        replace: replace value
        larger: choose whether > or <

    Returns:
        A new PIL.Image object
    """
    image = np.array(image)
    if larger:
        image = np.where(image > threshold, replace, image)
    else:
        image = np.where(image < threshold, replace, image)
    return Image.fromarray(image)


def add_spot(image, num, color):
    """
    Args:
         image: PIL.Image object
         num: number of spot
         color: color of spot

    Returns:
        spotted_image PIL.Image object
    """
    data = np.array(image)
    height, width = data.shape
    all_pos = [[i, j] for i in range(height) for j in range(width)]
    np.random.shuffle(all_pos)
    use_pos = all_pos[:num]
    for (x, y) in use_pos:
        data[x][y] = color
    img = Image.fromarray(data)
    return img


def rotate(image, resample=None, angle=None):
    if angle is None:
        angle = np.random.randint(-15, 15)
    if resample is None:
        resample = Image.BICUBIC
    image = image.rotate(angle, resample)
    return image


def remove_dark(img):
    data = np.array(img)
    data_ = np.where(data==0, 255, data)
    img_ = Image.fromarray(data_)
    return img_

def add_arc(img):
    dr = ImageDraw.Draw(img)
    randint = np.random.randint(0, 4)
    if randint == 0:
        dr.arc(((3,3), (90, 35)), 50, -200, fill=118)
    elif randint == 1:
        dr.arc(((3,3), (90, 35)), -120, -50, fill=118)
    elif randint == 2:
        dr.line(((0, 20), (100, 20)), width=2, fill=118)
    return img


