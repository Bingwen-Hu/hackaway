""" utility for image padding to a certain size, rotate or flip """
from PIL import Image
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


from PIL import ImageFilter
def blur(img):
    return img.filter(ImageFilter.SMOOTH)
    