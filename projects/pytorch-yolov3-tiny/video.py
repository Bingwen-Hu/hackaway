# detect special kind of class and return their position and crop image.
# input is a video or a single image

from __future__ import division
import time
import torch 
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2
from util import *
import argparse
import os 
import os.path as osp
from darknet import Darknet

import random 
import pickle as pkl

def arg_parse():
    parser=argparse.ArgumentParser(description='YOLO v3 Detection Module')
    parser.add_argument("--video", dest='video', help="Image / Directory containing images to perform detection upon", required=True, type=str)
    parser.add_argument("--det", dest='det', help="Image / Directory to store detections to", default="det", type=str)
    parser.add_argument("--bs", dest="bs", help="Batch size", default=1)
    parser.add_argument("--confidence", dest="confidence", help="Object Confidence to filter predictions", default=0.5)
    parser.add_argument("--nms_thresh", dest="nms_thresh", help="NMS Threshhold", default=0.4)
    parser.add_argument("--cfg", dest='cfgfile', help="Config file", default="cfg/yolov3.cfg", type=str)
    parser.add_argument("--weights", dest='weightsfile', help="weightsfile", default="yolov3.weights", type=str)
    parser.add_argument("--reso", dest='reso', help="Input resolution of the network. Increase to increase accuracy. Decrease to increase speed", default="416", type=str)
    parser.add_argument("--scales", dest="scales", help="Scales to use for detection", default="1,2,3", type=str)
    parser.add_argument("--names", dest='names', help='names file', default='data/coco.names', type=str)
    return parser.parse_args()


args = arg_parse()
scales = args.scales

batch_size = int(args.bs)
confidence = float(args.confidence)
nms_thesh = float(args.nms_thresh)
classes = load_classes(args.names) 
num_classes = len(classes)

# Set up the neural network
print("Loading network.....")
model = Darknet(args.cfgfile)
model.load_weights(args.weightsfile)
print("Network successfully loaded")

model.net_info["height"] = args.reso
inp_dim = int(model.net_info["height"])
assert inp_dim % 32 == 0 
assert inp_dim > 32

# Set the model in evaluation mode
model.eval()

# colors used by draw
colors = pkl.load(open("pallete", "rb"))


        
def cprep_image(orig_im: np.ndarray, inp_dim):
    """ Prepare image for inputting to the neural network. 
    Returns a Variable  """
    dim = orig_im.shape[1], orig_im.shape[0]
    img = (letterbox_image(orig_im, (inp_dim, inp_dim)))
    img_ = img[:,:,::-1].transpose((2,0,1)).copy()
    img_ = torch.from_numpy(img_).float().div(255.0).unsqueeze(0)
    return img_, orig_im, dim



def predict(image, testing=False):
    imlist = [image]
    batches = list(map(cprep_image, imlist, [inp_dim for x in range(len(imlist))]))
    im_batches = [x[0] for x in batches]
    orig_ims = [x[1] for x in batches]
    im_dim_list = [x[2] for x in batches]
    im_dim_list = torch.FloatTensor(im_dim_list).repeat(1, 2)
    with torch.no_grad():
        prediction = model(Variable(im_batches[0]), False)
    prediction = write_results(prediction, confidence, num_classes, nms=True, nms_conf=nms_thesh)
    output = prediction

    im_dim_list = torch.index_select(im_dim_list, 0, output[:,0].long())
    scaling_factor = torch.min(inp_dim/im_dim_list,1)[0].view(-1,1)
    
    output[:,[1,3]] -= (inp_dim - scaling_factor*im_dim_list[:,0].view(-1,1))/2
    output[:,[2,4]] -= (inp_dim - scaling_factor*im_dim_list[:,1].view(-1,1))/2
    
    output[:,1:5] /= scaling_factor
    
    for i in range(output.shape[0]):
        output[i, [1,3]] = torch.clamp(output[i, [1,3]], 0.0, im_dim_list[i,0])
        output[i, [2,4]] = torch.clamp(output[i, [2,4]], 0.0, im_dim_list[i,1])
    
    coord = [output_[1:5].tolist() for output_ in output]

    # classes
    labels = [int(output_[-1]) for output_ in output]
    labels = [classes[label] for label in labels]

    return coord, labels

def draw(img, coords, labels):
    def helper(img, coord, label):
        coord = list(map(int, coord))
        c1 = coord[0:2]
        c2 = coord[2:4] 
        img = cv2.rectangle(img, c1, c2, random.choice(colors))
        img = cv2.putText(img, label, (c1[0], c1[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225,255,255], 1)
    for coord, label in zip(coords, labels):
        helper(img, coord, label) 
    return img

if __name__ ==  '__main__':
    video = cv2.VideoCapture('/home/mory/Downloads/test.mp4')
    while True:
        ret, frame = video.read()
        if not ret:
            print('done!')
            break
        coords, labels = predict(frame)
        img = draw(frame, coords, labels)
        cv2.imshow('Test', img)
    cv2.destroyAllWindows()