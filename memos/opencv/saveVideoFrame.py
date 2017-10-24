import cv2
import os


if not os.path.exists("/tmp/cutVideo/"):
    os.makedirs("/tmp/cutVideo/")


video = cv2.VideoCapture("E:/Mory/0.mp4")

while(video.isOpened()):
    ret, frame = video.read()
    if ret == True:
        cv2.imwrite("frame.png", frame)
        break

video.release()
