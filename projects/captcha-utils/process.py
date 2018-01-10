import numpy as np
from PIL import Image


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
