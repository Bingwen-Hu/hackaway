import cv2
import time
import numpy as np


def single_show():
    cap = cv2.VideoCapture("/home/mory/Downloads/org.avi")

    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            cv2.imshow("frame", frame)
        else:
            break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        time.sleep(0.040)

    cap.release()
    cv2.destroyAllWindows()

import time
def multi_show(writepath=None):
    cap1 = cv2.VideoCapture('/home/mory/Downloads/dingcut.mp4')
    cap2 = cv2.VideoCapture('/home/mory/Downloads/dingcut.avi')
    fps = cap1.get(cv2.CAP_PROP_FPS)

    imgsize = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    videowriter = cv2.VideoWriter(writepath, fourcc, fps, imgsize)

    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        if ret1 == ret2 == True:
            # cut out
            frame1 = frame1[:, 300:-300, :]
            frame2 = frame2[:, 300:-300, :]
            concat = np.hstack([frame2, frame1, frame2])
            cv2.imshow('frame', concat)
            videowriter.write(concat)
        else:
            break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        time.sleep(0.5/fps)
    videowriter.release()
    cap1.release()
    cap2.release()
    cv2.destroyAllWindows()

multi_show('/home/mory/Downloads/concat.avi')


def nice_show(path):
    pass
