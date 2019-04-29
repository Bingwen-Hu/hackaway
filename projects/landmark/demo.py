import sys
import os

import time
import pprint

import dlib
import cv2
import numpy as np
import torch

from imutils import paths
from models import Landmark


images_dir = './images/'
result_dir = './results/'

image_list = paths.list_images(images_dir)

torch_net = Landmark()
torch_net.load_state_dict(torch.load('sd_landmark.pth'))
torch_net.eval()

detector = dlib.get_frontal_face_detector()

for image in image_list:
    img = cv2.imread(image)
    # The 1 in the second argument indicates that we should upsample the image
    # 1 time.  This will make everything bigger and allow us to detect more
    # faces.
    dets = detector(img, 1)
 
    for index, det in enumerate(dets):
        x1 = det.left()
        y1 = det.top()
        x2 = det.right()
        y2 = det.bottom()
        if x1 < 0: x1 = 0
        if y1 < 0: y1 = 0
        if x2 > img.shape[1]: x2 = img.shape[1]
        if y2 > img.shape[0]: y2 = img.shape[0]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        roi = img[y1:y2 + 1, x1:x2 + 1, ]
        gray_img = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
        w = 60
        h = 60

        res = cv2.resize(gray_img, (w, h), 0.0, 0.0, interpolation=cv2.INTER_CUBIC)
        resize_mat = np.float32(res)        
        mean, std_dev = cv2.meanStdDev(resize_mat)
        new_m = mean[0][0]
        new_sd = std_dev[0][0]
        new_img = (resize_mat - new_m) / (0.000001 + new_sd)

        with torch.no_grad():
            new_img = torch.FloatTensor(new_img[None, None, ...])
            torch_out = torch_net(new_img)
        points = torch_out.numpy().squeeze()
        point_pair_l = len(points)
        for i in range(point_pair_l // 2):
            x = points[2*i] * (x2 - x1) + x1
            y = points[2*i+1] * (y2 - y1) + y1
            cv2.circle(img, (int(x), int(y)), 1, (128, 255, 255), 2)
    cv2.imwrite(os.path.join(result_dir, os.path.basename(image)), img)