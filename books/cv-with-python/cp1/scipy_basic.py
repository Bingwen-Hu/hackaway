from PIL import Image
import numpy as np
from scipy.ndimage import filters, measurements, morphology


def ganssian_filter():
    # gaussian filter
    img = np.array(Image.open("E:/Mory/rose.jpg").convert("L"))
    img2 = filters.gaussian_filter(img, 5)

    # for RGB img
    img_RGB = np.array(Image.open("E:/Mory/rose.jpg"))
    img2_RGB = np.zeros(img_RGB.shape)
    for i in range(3):
        img2_RGB[:, :, i] = filters.gaussian_filter(img_RGB[:, :, i], 5)
    img2_RGB = np.uint8(img2_RGB)

    return img2, img2_RGB


def sobel_derivative():
    img = np.array(Image.open("E:/Mory/rose.jpg").convert('L'))

    # sobel derivative filters
    img_x = np.zeros(img.shape)
    filters.sobel(img, 1, img_x)

    img_y = np.zeros(img.shape)
    filters.sobel(img, 0, img_y)

    magnitude = np.sqrt(img_x ** 2 + img_y ** 2)
    return magnitude


def gaussian_derivative():
    img = np.array(Image.open("E:/Mory/rose.jpg").convert('L'))

    sigma = 5  # standard deviation
    img_x = np.zeros(img.shape)
    filters.gaussian_filter(img, (sigma, sigma), 1, output=img_x)

    img_y = np.zeros(img.shape)
    filters.gaussian_filter(img, (sigma, sigma), 0, output=img_y)

    return img_x, img_y


def binary_image():
    img = np.array(Image.open("E:/Mory/object.png").convert('L'))
    img = 1 * (img < 128)

    labels, nb_objects = measurements.label(img)
    print("Number of objects:", nb_objects)
