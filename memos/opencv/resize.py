# compare cv2 resize and skimage resize
import cv2
import skimage
import matplotlib.pyplot as plt


img = cv2.imread('/home/mory/hackaway/projects/pcn/1.jpg')
# notice the position of height and width
img_cv2 = cv2.resize(img.copy(), (100, 200))
img_ski = skimage.transform.resize(img.copy(), (200, 100))


cv2.imshow("cv2", img_cv2)
cv2.imshow("ski", img_ski)
cv2.waitKey(0)
cv2.destroyAllWindows()
