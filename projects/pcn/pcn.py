import numpy as np
import cv2


class Window:
    def __init__(self, x, y, width, angle, score):
        self.x = x
        self.y = y
        self.width = width
        self.angle = angle
        self.score = score

class Window2:
    def __init__(self, x, y, w, h, angle, scale, conf):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.angle = angle
        self.scale = scale
        self.conf = conf

def rotate_point(x, y, centerX, centerY, angle):
    x -= centerX
    y -= centerY
    theta = -angle * np.pi / 180
    rx = int(centerX + x * np.cos(theta) - y * np.sin(theta))
    ry = int(centerY + x * np.sin(theta) + y * np.cos(theta))
    return rx, ry


def draw_line(img, pointlist):
    thick = 2
    cyan = (0, 255, 255)
    blue = (0, 0, 255)
    cv2.line(img, pointlist[0], pointlist[1], cyan, thick)
    cv2.line(img, pointlist[1], pointlist[2], cyan, thick)
    cv2.line(img, pointlist[2], pointlist[3], cyan, thick)
    cv2.line(img, pointlist[3], pointlist[0], blue, thick)

def draw_face(img, face:Window):
    x1 = face.x
    y1 = face.y
    x2 = face.width + face.x -1
    y2 = face.width + face.y -1
    centerX = (x1 + x2) // 2
    centerY = (y1 + y2) // 2
    lst = (x1, y1), (x1, y2), (x2, y2), (x2, y1)
    pointlist = []
    for x, y in lst:
        rx, ry = rotate_point(x, y, centerX, centerY, face.angle)
        pointlist.append((rx, ry))
    draw_line(img, pointlist)

def crop_face(img, face:Window, cropsize):
    pass

## that all above PCN.h
## following is PCN.cpp
class PCN:
    @classmethod
    def loadModel(modelDetect:str, net1:str, net2:str, net3:str):
        pass

    @classmethod
    def resizeImg(img, scale:float):
        pass

    @classmethod
    def compareWin(w1:Window2, w2:Window2):
        pass

    @classmethod
    def legal(x, y, img):
        pass

    @classmethod
    def inside(x, y, rect:Window2):
        pass

    @classmethod
    def smooth_angle(a, b):
        pass

    @classmethod
    def smooth_window(winlist: list<Window2>):
        pass

    @classmethod
    def IoU(w1:Window2, w2:Window2) -> float:
        pass

    @classmethod
    def MNS(winlist:list<Window2>, local:bool, threshold:float) -> list<Window2>:
        pass

    @classmethod
    def deleteFP(winlist:list<Window2>):
        pass

    @classmethod
    def preprocess_img(img, dim=None):
        pass

    # method overload allow input as vector
    @classmethod
    def set_input(img, net):
        pass

    @classmethod
    def pad_img(img):
        pass

    @classmethod
    def trans_window(img, img_pad, winlist:list<Window2>):
        pass

    @classmethod
    def stage1(img, img_pad, net, thres):
        pass

    @classmethod
    def stage2(img, img180, net, thres, dim, winlist):
        pass

    @classmethod
    def stage3(img, img180, img90, imgNeg90, net, thres, dim, winlist):
        pass

    @classmethod
    def detect(img, img_pad):
        pass

    @classmethod
    def track(img, net, thres, dim, winlist):
        pass

    # class level variable
    net_ = [None, None, None]
    minFace_ = 0
    scale_ = 0
    stride_ = 0
    classThreshold_ = [0, 0, 0]
    nmsThreshold_ = [0, 0, 0]
    angleRange_ = 0
    stable_ = 0
    period_ = 0
    trackThreshold_ = 0
    augScale_ = 0


## here comes to PCN section

def init(net:PCN, modelDetect, net1, net2, net3):
    net.loadModel()




