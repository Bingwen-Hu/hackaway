import os

import cv2
import torch
import numpy as np

from .models import resnet_face18

threshold = 0.25

def load_model():
    cwd = os.path.dirname(__file__)
    net = resnet_face18(use_se=False)
    net = torch.nn.DataParallel(net)
    net.load_state_dict(torch.load(os.path.join(cwd, 'resnet18_110.pth'), map_location='cpu'))
    net.eval()
    return net

net = load_model()

def load_image(image):
    if type(image) == str:
        image = cv2.imread(image, 0)
    image = cv2.resize(image, (128, 128))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = np.dstack((image, np.fliplr(image)))
    image = image.transpose((2, 0, 1))
    image = image[:, np.newaxis, :, :]
    image = image.astype(np.float32, copy=False)
    image -= 127.5
    image /= 127.5
    return image


def cosin_metric(x1, x2):
    return np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))


def featurize(img_path):
    image = load_image(img_path)
    image = torch.from_numpy(image)
    with torch.no_grad():
        feature = net(image)
    feature = feature.numpy()
    fe1 = feature[::2]
    fe2 = feature[1::2]
    feature = np.hstack([fe1, fe2]).squeeze()
    return feature


def compare(f1, f2):
    """compare two faces or two features
    Args:
        f1: face feature vector 1
        f2: face feature vector 2
    Returns:
        face distance
    """
    return cosin_metric(f1, f2)