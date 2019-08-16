from __future__ import division

from .models import *
from .utils.utils import *
from .utils.datasets import *

import os
import sys
import time
import datetime
import argparse

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

from skimage.transform import resize

class Options:
    config_path = './yolo/config/yolov3.cfg'
    weights_path ='./yolo/weights/yolov3.weights'
    class_path = './yolo/data/coco.names'
    conf_thres = 0.8
    nms_thres = 0.4
    img_size = 416
    use_cuda = True
opt = Options()

cuda = torch.cuda.is_available() and opt.use_cuda

os.makedirs('output', exist_ok=True)

# Set up model
model = Darknet(opt.config_path, img_size=opt.img_size)
model.load_weights(opt.weights_path)

if cuda:
    model.cuda()

model.eval() # Set in evaluation mode
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

classes = load_classes(opt.class_path) # Extracts class labels from file


def preprocess(img, img_size):
    h, w, _ = img.shape
    dim_diff = np.abs(h - w)
    # Upper (left) and lower (right) padding
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    # Determine padding
    pad = ((pad1, pad2), (0, 0), (0, 0)) if h <= w else ((0, 0), (pad1, pad2), (0, 0))
    # Add padding
    input_img = np.pad(img, pad, 'constant', constant_values=127.5) / 255.
    # Resize and normalize
    input_img = resize(input_img, (*img_size, 3), mode='reflect')
    # Channels-first
    input_img = np.transpose(input_img, (2, 0, 1))
    # As pytorch tensor
    input_img = torch.from_numpy(np.array([input_img])).float()
    return input_img


def detect(img):
    img = cv2.imread(img) if type(img) == str else img
    input_img = preprocess(img, (opt.img_size, opt.img_size))
    if cuda:
        input_img = input_img.cuda()

    # Get detections
    with torch.no_grad():
        detections = model(input_img)
        detections = non_max_suppression(detections, 80, opt.conf_thres, opt.nms_thres)
    # single image
    detections = detections[0]
    detections = detections.cpu().numpy()
    # The amount of padding that was added
    pad_x = max(img.shape[0] - img.shape[1], 0) * (opt.img_size / max(img.shape))
    pad_y = max(img.shape[1] - img.shape[0], 0) * (opt.img_size / max(img.shape))
    # Image height and width after padding is removed
    unpad_h = opt.img_size - pad_y
    unpad_w = opt.img_size - pad_x

    # return bbox and label
    def fetch_bbox_label(detection):
        x1, y1, x2, y2, conf, cls_conf, cls_pred = detection
        label = classes[int(cls_pred)]
        box_h = ((y2 - y1) / unpad_h) * img.shape[0]
        box_w = ((x2 - x1) / unpad_w) * img.shape[1]
        y1 = ((y1 - pad_y // 2) / unpad_h) * img.shape[0]
        x1 = ((x1 - pad_x // 2) / unpad_w) * img.shape[1]
        y2 = y1 + box_h
        x2 = x1 + box_w
        return {'bbox': list(map(int, (x1, y1, x2, y2))), 'label': label} 
    return list(map(fetch_bbox_label, detections)) 