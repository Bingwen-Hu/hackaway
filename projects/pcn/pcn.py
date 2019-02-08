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
    def loadModel():
        from models import PCN1, PCN2, PCN3
        pcn1 = torch.load('pth/pcn1.pth')
        pcn2 = torch.load('pth/pcn2.pth')
        pcn3 = torch.load('pth/pcn3.pth')
        return pcn1, pcn2, pcn3


    @classmethod
    def resizeImg(img, scale:float):
        h, w = img.shape
        h_, w_ = int(h_ / scale), int(w_ / scale)
        return cv2.resize(img, (w_, h_))        

    @classmethod
    def compareWin(w1:Window2, w2:Window2):
        return w1.conf > w2.conf

    @classmethod
    def legal(x, y, img):
        if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
            return True
        else:
            return False

    @classmethod
    def inside(x, y, rect:Window2):
        if rect.x <= x < rect.x + rect.w and rect.y <= y < rect.y + rect.h:
            return True
        else:
            return False

    @classmethod
    def smooth_angle(a, b):
        if a > b:
            a, b = b, a
        # a <= b
        diff = (b - a) % 360
        if diff < 180:
            return a + diff // 2
        else:
            return b + (360 - diff) // 2

    @classmethod
    def smooth_window(winlist: list<Window2>):
        pass

    @classmethod
    def IoU(w1:Window2, w2:Window2) -> float:
        xOverlap = max(0, min(w1.x + w1.w - 1, w2.x + w2.w - 1) - max(w1.x, w2.x) + 1)
        yOverlap = max(0, min(w1.y + w1.h - 1, w2.y + w2.h - 1) - max(w1.y, w2.y) + 1)
        intersection = xOverlap * yOverlap
        unio = w1.w * w1.h + w2.w * w2.h - intersection
        return intersection / unio

    @classmethod
    def MNS(winlist:list<Window2>, local:bool, threshold:float) -> list<Window2>:
        length = len(winlist)
        if length == 0:
            return winlist
        winlist.sort(key=lambda x: x.conf, reverse=True)        
        flag = [0] * length
        for i in range(length):
            if flag[i]:
                continue
            for j in range(i+1, length):
                if local and abs(winlist[i].scale - winlist[j].scale) > EPS:
                    continue
                if IoU(winlist[i], winlist[j]) > threshold:
                    flag[j] = 1
        ret = [winlist[i] for i in range(length) if flag[i])
        return ret

    @classmethod
    def deleteFP(winlist:list<Window2>):
        pass

    @classmethod
    def preprocess_img(img, dim=None):


    # method overload allow input as vector
    @classmethod
    def set_input(img, net):
        pass

    @classmethod
    def pad_img(img):
        row = min(int(img.shape[0] * 0.2), 100)
        col = min(int(img.shape[1] * 0.2), 100)
        ret = cv2.copyMakeBorder(img, row, row, col, col, cv2::BORDER_CONSTANT)
        return ret

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




