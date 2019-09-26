import os
import os.path as osp
import cv2
import numpy as np

import torch
from torchvision import transforms
from PIL import Image
from .net.pspnet import PSPNet



class Parameter:
    snapshot = "weights/PSPNet_last"

class HumanParseTest:
    
    def __init__(self, params: Parameter):
        self.params = params
        self.snapshot = params.snapshot

    def preprocess(self, im):
        fns = [
            transforms.Resize((256, 256), 3),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
        for fn in fns:
            im = fn(im) 
        im = im[None, ...]
        return im

    def inference(self, net, im):
        w, h = im.size
        input = self.preprocess(im)
        with torch.no_grad():
            pred_seg, _ = net(input)
        pred_seg = pred_seg[0]
        pred = pred_seg.numpy().transpose(1, 2, 0)
        pred = cv2.resize(pred, (w, h), interpolation=cv2.INTER_CUBIC)
        pred = np.asarray(np.argmax(pred, axis=2), dtype=np.uint8).reshape([h, w])
        return pred

    def save(self, prediction, filename):
        im = Image.fromarray(prediction, mode='P')
        colors = [c * 12 for c in range(20)]
        im.putpalette(colors)
        im.save(filename)


HPT = HumanParseTest(Parameter)

net = PSPNet(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='squeezenet')
net = torch.nn.DataParallel(net)
weights = osp.join(osp.dirname(__file__), HPT.snapshot)
net.load_state_dict(torch.load(weights, map_location='cpu'))
net.eval()


def parse(path_or_iofile):
    im = Image.open(path_or_iofile)
    return HPT.inference(net, im)

def save(prediction, filename):
    HPT.save(prediction, filename)

