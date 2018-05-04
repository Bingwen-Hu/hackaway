import cv2

img = cv2.imread('E:/Mory/rose.jpg')

cv2.rectangle(img, (50, 50), (100, 100), (0, 255, 0))
cv2.imshow("rose", img)
cv2.waitKey(0)