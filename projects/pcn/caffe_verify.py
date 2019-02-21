import caffe
import cv2
import pcn

resizeImg = pcn.resizeImg
preprocess_img = pcn.preprocess_img
legal = pcn.legal
Window2 = pcn.Window2

scale_ = pcn.scale_
stride_ = pcn.stride_
thres = pcn.classThreshold_[0]
minFace_ = pcn.minFace_

def forward_caffe(img, net):
    net.blobs['data'].data[...] = img
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
                    print(f'cls_prob[0, 1, {i}, {j}] = {cls_prob[0, 1, i, j]}')
                    layerdetect += 1
                    sn = bbox[0, 0, i, j]
                    xn = bbox[0, 1, i, j]
                    yn = bbox[0, 2, i, j]
                    rx = int(j * curScale * stride_ - 0.5 * sn * w + sn * xn * w + 0.5 * w) + col
                    ry = int(i * curScale * stride_ - 0.5 * sn * w + sn * yn * w + 0.5 * w) + row
                    rw = int(w * sn)
                    if legal(rx, ry, img_pad) and legal(rx + rw - 1, ry + rw -1, img_pad):
                        if rotate[0, 1, i, j] > 0.5:
                            winlist.append(Window2(rx, ry, rw, rw, 0, curScale, cls_prob[0, 1, i, j])
                        else:
                            winlist.append(Window2(rx, ry, rw, rw, 180, curScale, cls_prob[0, 1, i, j])
        print("layer detect", layerdetect)
        img_resized = resizeImg(img_resized, scale_)                    
        curScale = round(img.shape[0] / img_resized.shape[0],3)
    return winlist                
    
if __name__ == "__main__":
    imgpath = 'imgs/5.jpg'
    img = cv2.imread(imgpath)
    imgPad = pcn.pad_img(img)
    thres = 0.37
    
    pcn1 = caffe.Net('model/PCN-1.prototxt', 'model/PCN.caffemodel', caffe.TEST)
    stage1_caffe(img, imgPad, pcn1, thres)