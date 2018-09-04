from __future__ import division
import torch 
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from util import *
import argparse
from darknet import Darknet


def arg_parse():
    parser=argparse.ArgumentParser(description='YOLO v3 Detection Module')
    parser.add_argument("--image", dest='image', help="Image / Directory containing images to perform detection upon", default='test.jpg', type=str)
    parser.add_argument("--bs", dest="bs", help="Batch size", default=1)
    parser.add_argument("--confidence", dest="confidence", help="Object Confidence to filter predictions", default=0.5)
    parser.add_argument("--nms_thresh", dest="nms_thresh", help="NMS Threshhold", default=0.4)
    parser.add_argument("--cfg", dest='cfgfile', help="Config file", default="cfg/yolov3-tiny.cfg", type=str)
    parser.add_argument("--weights", dest='weightsfile', help="weightsfile", default="yolov3-tiny.weights", type=str)
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

def predict(image, testing=False):
    imlist = [image]
    batches = list(map(prep_image, imlist, [inp_dim for x in range(len(imlist))]))
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

    if testing is True:
        from PIL import Image
        img = Image.open(image)
        for i, xy in enumerate(coord):
            crop = img.crop(xy)
            crop.save(f'{i}.jpg')
            
    return coord

if __name__ ==  '__main__':
    path = 'test.jpg'
    coord = predict(path, testing=True)
    print(coord)