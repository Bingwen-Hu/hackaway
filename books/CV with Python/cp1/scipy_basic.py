from PIL import Image
import numpy as np
from scipy.ndimage import filters

# gaussian filter
img = np.array(Image.open("E:/Mory/rose.jpg").convert("L"))
img2 = filters.gaussian_filter(img, 5)
