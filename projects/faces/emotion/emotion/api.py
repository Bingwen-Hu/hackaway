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
        probability = res[0, class_id]
        label = LABELS[class_id]
        results.append({'bbox': [x1, y1, x2, y2], 'emotion': label, 'probability': probability})

    return results


def detect_cropped(img):
    """input a face and detect its emotion"""
    if type(img) == str:
        img = cv2.imread(img)
    roi = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    roi = cv2.resize(roi, (64, 64))
    roi = preprocess_input(roi)
    res = net.predict(roi[None, :, :, None])
    class_id = np.argmax(res)
    probability = res[0, class_id]
    label = LABELS[class_id]
    return label, probability

 
def show(img):
    if type(img) == str:
        img = cv2.imread(img)
    results = detect(img)
    for result in results:
        x1, y1, x2, y2 = result['bbox']
        emotion = "{}: {:.3f}".format(result['emotion'], result['probability'])
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, emotion, (x1, y1), cv2.FONT_HERSHEY_COMPLEX, fontScale=1, color=(0, 255, 0), thickness=2)

    cv2.imshow("Emotion", img)
    cv2.imwrite('test.jpg', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()