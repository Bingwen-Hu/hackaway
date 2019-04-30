import cv2
from landmark import detect


if __name__ == '__main__':
    imgpath = 'images/timg.jpeg'
    img = detect(imgpath)
    cv2.imshow("image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()