import cv2
import numpy as np

img = cv2.imread('E:/Mory/github/hackaway/projects/pytorch-yolov3-tiny/dog-cycle-car.png')

def basic(img):
    # draw line
    right_bottom = img.shape[1], img.shape[0]
    cv2.line(img, (0, 0), right_bottom, [0, 255, 0])

    center = tuple(x // 2 for x in right_bottom)
    cv2.circle(img, center, radius=min(right_bottom)//2, color=[0, 0, 255])

    cv2.rectangle(img, (0, 0), right_bottom, color=[255, 128, 128], thickness=4, lineType=2)

    cv2.imshow('demo', img)
    cv2.waitKey(0)

def argu(img):
    flipped = cv2.flip(img, -1) # -1, 0, 1
    cv2.imshow('argu', flipped)
    cv2.waitKey(0)

def mask(img):
    shape = img.shape[:2]
    center = tuple(x // 2 for x in shape)
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.circle(mask, center, min(center), color=255, thickness=-1)
    masked = cv2.bitwise_and(img, img, mask=mask)
    cv2.imshow("mask", mask)
    cv2.imshow('mask applied to image', masked)
    cv2.waitKey(0)


mask(img)