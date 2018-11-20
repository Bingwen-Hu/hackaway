import cv2
import numpy as np


def findContour(imagepath):
    img = cv2.imread(imagepath)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow('image', img)

    blurred = cv2.GaussianBlur(gray, (11, 11), 0)
    edged = cv2.Canny(blurred, 30, 150)
    cv2.imshow('Edges', edged)

    (_, cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    alphebat = img.copy()
    cv2.drawContours(alphebat, cnts, -1, (0, 255, 0), 2)
    cv2.imshow('Coins', alphebat)
    cv2.waitKey(0)

