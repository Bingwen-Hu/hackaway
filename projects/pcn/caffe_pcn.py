import numpy as np
import caffe
import cv2
import pcn


resizeImg = pcn.resizeImg
preprocess_img = pcn.preprocess_img
legal = pcn.legal
Window2 = pcn.Window2

EPS = pcn.EPS
scale_ = pcn.scale_
stride_ = pcn.stride_
thres = pcn.classThreshold_[0]
minFace_ = pcn.minFace_


def forward_caffe(img, net):
    if type(img) == list:
        img = np.stack(img, axis=0)
        rows = img[0].shape[0]
        cols = img[0].shape[1]
        net_input = np.transpose(img, (0, 3, 1, 2))
        num = img.shape[0]
    else:
        rows = img.shape[0]
        cols = img.shape[1]
        net_input = np.transpose(img, (2, 0, 1))
        net_input = net_input[np.newaxis, :, :, :]
        num = 1
    net.blobs['data'].reshape(num, 3, rows, cols)
    net.reshape()
    net.blobs['data'].data[...] = net_input
    out = net.forward()
    return out

def stage1_caffe(img, img_pad, net, thres):
    row = (img_pad.shape[0] - img.shape[0]) // 2
    col = (img_pad.shape[1] - img.shape[1]) // 2
    winlist = []
    netSize = 24
    curScale = round(minFace_ / netSize, 3)
    img_resized = resizeImg(img, curScale)
    layerdetect = 0
    while min(img_resized.shape[:2]) >= netSize:
        img_resized = preprocess_img(img_resized)

        out = forward_caffe(img_resized, net)
        cls_prob = out['cls_prob'].data
        bbox = out['bbox_reg_1'].data
        rotate = out['rotate_cls_prob'].data

        w = netSize * curScale
        for i in range(cls_prob.shape[2]): # cls_prob[2]->height
            for j in range(cls_prob.shape[3]): # cls_prob[3]->width
                if cls_prob[0, 1, i, j] > thres:
                    print('cls_prob[0, 1, {}, {}] = {}'.format(i, j, cls_prob[0, 1, i, j]))
                    layerdetect += 1
                    sn = bbox[0, 0, i, j]
                    xn = bbox[0, 1, i, j]
                    yn = bbox[0, 2, i, j]
                    rx = int(j * curScale * stride_ - 0.5 * sn * w + sn * xn * w + 0.5 * w) + col
                    ry = int(i * curScale * stride_ - 0.5 * sn * w + sn * yn * w + 0.5 * w) + row
                    rw = int(w * sn)
                    if legal(rx, ry, img_pad) and legal(rx + rw - 1, ry + rw -1, img_pad):
                        if rotate[0, 1, i, j] > 0.5:
                            winlist.append(Window2(rx, ry, rw, rw, 0, curScale, cls_prob[0, 1, i, j]))
                        else:
                            winlist.append(Window2(rx, ry, rw, rw, 180, curScale, cls_prob[0, 1, i, j]))
        print("layer detect", layerdetect)
        img_resized = resizeImg(img_resized, scale_)

        curScale = round(img.shape[0] / img_resized.shape[0],3)
    return winlist



def stage2_caffe(img, img180, net, thres, dim, winlist):
    length = len(winlist)
    if length == 0:
        return winlist
    datalist = []
    height = img.shape[0]
    for win in winlist:
        if abs(win.angle) < EPS:
            datalist.append(preprocess_img(img[win.y:win.y+win.h, win.x:win.x+win.w,:], dim))
        else:
            y2 = win.y + win.h -1
            y = height - 1 - y2
            datalist.append(preprocess_img(img[y:y+win.h, win.x:win.x+win.w, :], dim))
        
    out = forward_caffe(datalist, net)
    cls_prob = out['cls_prob'].data
    bbox = out['bbox_reg_1'].data
    rotate = out['rotate_cls_prob'].data

    ret = []
    for i in range(length):
        if cls_prob[i, 1, 0, 0] > thres:
            sn = bbox[i, 0, 0, 0]
            xn = bbox[i, 1, 0, 0]
            yn = bbox[i, 2, 0, 0]
            cropX = winlist[i].x
            cropY = winlist[i].y 
            cropW = winlist[i].w
            if abs(winlist[i].angle) > EPS:
                cropY = height - 1 - (cropY + cropW - 1)
            w = int(sn * cropW)
            x = int(cropX - 0.5 * sn * cropW + cropW * sn * xn + 0.5 * cropW)
            y = int(cropY - 0.5 * sn * cropW + cropW * sn * yn + 0.5 * cropW)
            maxRotateScore = 0
            maxRotateIndex = 0
            for j in range(3):
                if rotate[i, j, 0, 0] > maxRotateScore:
                    maxRotateScore = rotate[i, j].item()
                    maxRotateIndex = j
            if legal(x, y, img) and legal(x+w-1, y+w-1, img):
                angle = 0
                if abs(winlist[i].angle) < EPS:
                    if maxRotateIndex == 0:
                        angle = 90
                    elif maxRotateIndex == 1:
                        angle = 0
                    else:
                        angle = -90
                    ret.append(Window2(x, y, w, w, angle, winlist[i].scale, cls_prob[i, 1, 0, 0]))
                else:
                    if maxRotateIndex == 0:
                        angle = 90
                    elif maxRotateIndex == 1:
                        angle = 180
                    else:
                        angle = -90
                    ret.append(Window2(x, height-1-(y+w-1), w, w, angle, winlist[i].scale, cls_prob[i, 1, 0, 0]))
    return ret


def stage3_caffe():
    pass

def detect_caffe():
    pass

if __name__ == "__main__":
    imgpath = 'imgs/5.jpg'
    img = cv2.imread(imgpath)
    imgPad = pcn.pad_img(img)
    img180 = cv2.flip(imgPad, 0)

    pcn1 = caffe.Net('model/PCN-1.prototxt', 'model/PCN.caffemodel', caffe.TEST)
    pcn2 = caffe.Net('model/PCN-2.prototxt', 'model/PCN.caffemodel', caffe.TEST)
    winlist = stage1_caffe(img, imgPad, pcn1, thres)
    winlist = pcn.NMS(winlist)
    winlist = stage2_caffe(img, img180, pcn2, pcn.classThreshold_[1], 24, winlist)