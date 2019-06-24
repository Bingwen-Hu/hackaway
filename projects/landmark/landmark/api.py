import os
import cv2

import torch
import numpy as np

import facessh

from .models import Landmark



def load_model(): 
    net = Landmark()
    cwd = os.path.dirname(__file__) 
    net.load_state_dict(torch.load(os.path.join(cwd, 'sd_landmark.pth')))
    net.eval()
    return net

net = load_model()

def detect(image:str, mode='fast'):
    global net
    img = cv2.imread(image)
    dets = facessh.detect(img, scale_mode=mode)
    results = []
 
    for index, det in enumerate(dets):
        x1 = max(int(det[0]), 0)
        y1 = max(int(det[1]), 0)
        x2 = min(int(det[2]), img.shape[1])
        y2 = min(int(det[3]), img.shape[0])
        roi = img[y1:y2 + 1, x1:x2 + 1, ]
        gray_img = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
        w = 60
        h = 60

        res = cv2.resize(gray_img, (w, h), interpolation=cv2.INTER_CUBIC)
        resize_mat = np.float32(res)        
        mean, std_dev = cv2.meanStdDev(resize_mat)
        new_m = mean[0][0]
        new_sd = std_dev[0][0]
        new_img = (resize_mat - new_m) / (0.000001 + new_sd)

        with torch.no_grad():
            new_img = torch.FloatTensor(new_img[None, None, ...])
            torch_out = net(new_img)
        points = torch_out.numpy().squeeze()
        points[0::2] = points[0::2] * (x2-x1) + x1
        points[1::2] = points[1::2] * (y2-y1) + y1
        coutour = points[0:17*2].tolist()
        left_eyebrow = points[17*2:22*2].tolist()
        right_eyebrow = points[22*2:27*2].tolist()
        nose = points[27*2:36*2].tolist()
        left_eye = points[36*2:42*2].tolist()
        right_eye = points[42*2:48*2].tolist()
        mouse = points[49*2:].tolist()
        landmark = {
            'coutour': coutour, 'left_eyebrow': left_eyebrow, 'right_eyebrow': right_eyebrow,
            'left_eye': left_eye, 'right_eye': right_eye, 'mouth': mouse, 'nose': nose,
        }
        results.append({'bbox': [x1, y1, x2, y2], 'landmark': landmark})
    return results

def show(image:str):
    img = cv2.imread(image)
    results = detect(image)
    for result in results:
        landmark_ = result['landmark']
        x1, y1, x2, y2 = result['bbox']
        for k in landmark_:
            points = landmark_[k]
            for i in range(0, len(points), 2):
                cv2.circle(img, (int(points[i]), int(points[i+1])), 1, (128, 255, 255), 2)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.imshow("image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return img
