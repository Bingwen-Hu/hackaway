import numpy as np
from PIL import Image, ImageDraw


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


