import os

import dlib
import cv2

import numpy as np

from .models import mini_XCEPTION, LABELS
from .utils import load_model, preprocess_input


detector = dlib.get_frontal_face_detector()
net = load_model()

def detect(img):
    global net
    if type(img) == str:
        img = cv2.imread(img)
    dets = detector(img, 1)
    results = []
 
    for index, det in enumerate(dets):
        x1 = max(det.left(), 0)
        y1 = max(det.top(), 0)
        x2 = min(det.right(), img.shape[1])
        y2 = min(det.bottom(), img.shape[0])
        roi = img[y1:y2, x1:x2, :]
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        roi = cv2.resize(roi, (64, 64))
        roi = preprocess_input(roi)

        res = net.predict(roi[None, :, :, None])
        class_id = np.argmax(res)
        label = LABELS[class_id]
        results.append({'bbox': [x1, y1, x2, y2], 'emotion': label})

    return results

        
