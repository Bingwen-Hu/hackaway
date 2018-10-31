import cv2

img = cv2.imread('E:/Mory/github/hackaway/projects/pytorch-yolov3-tiny/dog-cycle-car.png')

# draw line
right_bottom = img.shape[1], img.shape[0]
cv2.line(img, (0, 0), right_bottom, [0, 255, 0])

center = tuple(x // 2 for x in right_bottom)
cv2.circle(img, center, radius=min(right_bottom)//2, color=[0, 0, 255])

cv2.rectangle(img, (0, 0), right_bottom, color=[255, 128, 128], thickness=4, lineType=2)

cv2.imshow('demo', img)
cv2.waitKey(0)
