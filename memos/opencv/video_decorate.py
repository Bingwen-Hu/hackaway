import cv2
import time
import numpy as np
import os

def alignment(video, imgdir, skip=None):
    cap = cv2.VideoCapture(video)
    images = sorted(os.listdir(imgdir), key=lambda x: x[:4])
    images = [os.path.join(imgdir, f) for f in images]
    index = 0
    # skip some frame
    if skip is not None:
        for i in range(skip):
            ret, frame = cap.read()
        images = images[skip:]
    # yield 
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        img = cv2.imread(images[index])
        index += 1
        yield frame, img



def reside(video, imgdir):
    for frame, img in alignment(video, imgdir):
        h, w = img.shape[:2]
        img = cv2.resize(img, (2*w, 2*h))
        h, w = img.shape[:2]
        w_ = w//3
        h_ = h//3
        frame = cv2.resize(frame, (w_, h_))
        img[h-h_:, 0:w_, :] = frame
        cv2.imshow('test', img)
        cv2.waitKey(1)
    cv2.destroyAllWindows()



def nice(video, imgdir):
    for frame, img in alignment(video, imgdir):
        skel = img[:, 182:182+142, :]
        skel_layer = np.hstack([skel, skel, skel])
        h, w = skel_layer.shape[:2]
        frame = cv2.resize(frame, (w, h))
        seven = np.vstack([skel_layer, frame, skel_layer])
        cv2.imshow('test', seven)
        cv2.waitKey(1)
    cv2.destroyAllWindows()
        

def six(video, imgdir):
    offset = 80
    cutoff = 90
    c = 0
    for frame, img in alignment(video, imgdir):
        img = cv2.resize(img, (2*img.shape[1], 2*img.shape[0]))
        start_c = img.shape[1] // 3 + offset
        skel = img[cutoff:, start_c:start_c+img.shape[1]//3, :]
        skel_layer = np.hstack([skel, skel, skel])

        # original
        start_c = frame.shape[1] // 3 + offset * 5
        frame = frame[cutoff*6:, start_c:start_c+frame.shape[1]//3, :]
        frame = cv2.resize(frame, (skel.shape[1], skel.shape[0]))
        frame_layer = np.hstack([skel, frame, skel])
        six_layer = np.vstack([skel_layer, frame_layer])
        cv2.imwrite(f"six/{c}.png", six_layer)
        c += 1
        # cv2.imshow('test', six_layer)
        # cv2.waitKey(1)
    # cv2.destroyAllWindows()


def six2(video, imgdir):
    for frame, img in alignment(video, imgdir):
        frame = cv2.resize(frame, (img.shape[1], img.shape[0]))
        skel_layer = np.hstack([img, img])
        frame_layer = np.hstack([frame, img])
        six2_layer = np.vstack([skel_layer, frame_layer])
        cv2.imshow('test', six2_layer)
        cv2.waitKey(1)
    cv2.destroyAllWindows()
        

def nice_demo():
    video = '/home/mory/data/final/haicao/haicao2.mp4'
    imgdir = '/home/mory/Downloads/haicao2'
    nice(video, imgdir)

def six_demo():
    video = '/home/mory/data/final/haicao/haicao2.mp4'
    imgdir = '/home/mory/Downloads/haicao2'
    six(video, imgdir)


def six2_demo():
    video = '/home/mory/data/final/haohi/haohi2.mp4'
    imgdir = '/home/mory/Downloads/haohi2'
    six2(video, imgdir)


def reside_demo():
    video = '/home/mory/data/final/haicao/haicao1.mp4'
    imgdir = '/home/mory/Downloads/haicao1'
    reside(video, imgdir)


def tangle():
    video1 = '/home/mory/data/final/haohi/haohi2.mp4'
    imgdir1 = '/home/mory/Downloads/haohi2'
    video2 = '/home/mory/data/final/haohi/haohi1.mp4'
    imgdir2 = '/home/mory/Downloads/haohi1'
    c = 0
    for (frame1, img1), (frame2, img2) in zip(alignment(video1, imgdir1), alignment(video2, imgdir2, skip=30)):
        # resize
        frame1 = cv2.resize(frame1, (img1.shape[1], img1.shape[0]))
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
        frame2 = cv2.resize(frame2, (img1.shape[1], img1.shape[0]))

        
        skel_layer = np.hstack([img1, img2])
        frame_layer = np.hstack([frame1, frame2])
        tangle_layer = np.vstack([skel_layer, frame_layer])
        cv2.imwrite(f'tangle/{c}.png', tangle_layer) 
        c += 1
        # cv2.imshow('test', tangle_layer)
        # cv2.waitKey(1)
    # cv2.destroyAllWindows()