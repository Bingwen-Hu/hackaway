# matplotlib prefer BGR while Matplotlib like RGB

import cv2
import matplotlib.pyplot as plt
import numpy as np


img = cv2.imread('E:/Mory/rose.jpg')
b, g, r = cv2.split(img)
img2 = cv2.merge([r, g, b])
plt.subplot(121)
plt.imshow(img)
plt.subplot(122)
plt.imshow(img2)
plt.show()


cv2.imshow("bgr image", img)
cv2.imshow("rgb image", img2)
cv2.waitKey(0)
cv2.destroyAllWindows()


