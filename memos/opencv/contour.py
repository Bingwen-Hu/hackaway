import cv2
import numpy as np
from uuid import uuid1
from imutils import paths


def findContour(imagepath):
    img = cv2.imread(imagepath)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


    blurred = cv2.GaussianBlur(gray, (11, 11), 0)
    edged = cv2.Canny(blurred, 30, 150)


    (_, cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    alphebat = img.copy()
    cv2.drawContours(alphebat, cnts, -1, (0, 255, 0), 2)

    for i, c in enumerate(cnts):
        (x, y, w, h)  = cv2.boundingRect(c)
        crop = img[y:y+h, x:x+w]
        cv2.imwrite(f"{uuid1()}.png", crop)


if __name__ == '__main__':
    directory = "/home/jenny/Downloads/hkcaptcha/unuse/crop/"
    imgpaths = paths.list_images(directory)
    
    for imgpath in imgpaths:
        findContour(imgpath)
        