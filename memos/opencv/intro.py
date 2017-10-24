# -*- coding: utf-8 -*-
"""
opencv memos
"""
import matplotlib.pyplot as plt
import numpy as p
import cv2

#==============================================================================
# section1, read and display
#==============================================================================
# 0 is gray, 1 is color -1 is unchange
img = cv2.imread("E:/Mory/rose.jpg", 0)

cv2.imshow("image", img)

# this two line is needed
cv2.waitKey(0)
cv2.destroyAllWindows(cv2.WINDOW_AUTOSIZE)


#==============================================================================
# section2, resize the window
#==============================================================================
cv2.namedWindow("image", cv2.WINDOW_NORMAL)
cv2.imshow("image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()


#==============================================================================
# section3, write an image
#==============================================================================
cv2.imwrite("/tmp/soitis.jpg", img)


#==============================================================================
# section4, using matplotlib
#==============================================================================
plt.imshow(img, cmap='gray', interpolation='bicubic')
plt.xticks([]), plt.yticks([])
plt.show()


