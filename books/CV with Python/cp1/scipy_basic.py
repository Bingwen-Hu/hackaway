from PIL import Image
import numpy as np
from scipy.ndimage import filters

# gaussian filter
img = np.array(Image.open("E:/Mory/rose.jpg").convert("L"))
img2 = filters.gaussian_filter(img, 5)

# for RGB img
img_RGB = np.array(Image.open("E:/Mory/rose.jpg"))
img2_RGB = np.zeros(img_RGB.shape)
for i in range(3):
    img2_RGB[:, :, i] = filters.gaussian_filter(img_RGB[:, :, i], 5)
img2_RGB = np.uint8(img2_RGB)
