import os

import cv2
import torch
import numpy as np

from .models import resnet_face18


def load_model():
    net = resnet_face18(use_se=False)
    cwd = os.path.dirname(__file__)
    net.load_state_dict(torch.load(os.path.join(cwd, 'resnet18_110.pth'), map_location='cpu'))
    net.eval()
    return net

net = load_model()

def load_image(img_path):
    image = cv2.imread(img_path, 0)
    image = np.dstack((image, np.fliplr(image)))
    image = image.transpose((2, 0, 1))
    image = image[:, np.newaxis, :, :]
    image = image.astype(np.float32, copy=False)
    image -= 127.5
    image /= 127.5
    return image



def featurize(img_path):
    image = load_image(img_path)
    feature = net(image)
    return feature