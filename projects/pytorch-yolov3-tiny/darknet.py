from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from PIL import Image
import numpy as np

from util import predict_transform


def get_test_input():
    img = Image.open('dog-cycle-car.png')
    img = img.resize((416, 416), Image.BICUBIC)
    img_ = np.array(img)
    img_ = img_.transpose((2, 0, 1))
    img_ = img_[np.newaxis, :, :, :] / 255.0
    img_ = torch.from_numpy(img_).float()
    img_ = Variable(img_)
    return img_


def parse_cfg(cfgfile):
    """Takes a configuration file and return a list with certain parameter
    in a form of list of dicts """
    file = open(cfgfile, 'r')
    lines = file.read().split('\n')
    file.close()
    lines = [x for x in lines if len(x) > 0]
    lines = [x for x in lines if x[0] != "#"]
    lines = [x.strip() for x in lines]

    blocks = []
    block = {}
    # author original
    for line in lines:
        if line[0] == '[':
            if len(block) != 0:
                blocks.append(block)
                block = {}
            block['type'] = line[1:-1].strip()
        else:
            key, value = line.split('=')
            block[key.strip()] = value.strip()
    blocks.append(block)
    
    return blocks


def create_modules(blocks):
    net_info = blocks[0]
    module_list = nn.ModuleList()
    prev_filters = 3
    output_filters = []

    for index, x in enumerate(blocks[1:]):
        module = nn.Sequential()
        if x['type'] == 'convolutional':
            activation = x['activation']
            try:
                batch_normalize = int(x['batch_normalize'])
                bias = False
            except:
                batch_normalize = 0
                bias = True

            filters = int(x['filters'])
            padding = int(x['pad'])
            kernel_size = int(x['size'])
            stride = int(x['stride'])

            if padding:
                pad = (kernel_size - 1) // 2
            else:
                pad = 0
            
            conv = nn.Conv2d(prev_filters, filters, kernel_size, stride, pad, bias=bias)
            module.add_module('conv_{}'.format(index), conv)

            if batch_normalize:
                bn = nn.BatchNorm2d(filters)
                module.add_module('batch_norm_{}'.format(index), bn)
            
            if activation == 'leaky':
                activn = nn.LeakyReLU(0.1, inplace=True)
                module.add_module('leaky_{}'.format(index), activn)

        elif x['type'] == 'maxpool':
            kernel_size = int(x['size'])
            stride = int(x['stride'])
            if stride == 1:
                maxpool = MaxPoolStride1(kernel_size)
            else:
                maxpool = nn.MaxPool2d(kernel_size, stride)
            module.add_module('maxpool_{}'.format(index), maxpool)

        elif x['type'] == 'upsample':
            stride = int(x['stride'])
            upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            module.add_module('upsample_{}'.format(index), upsample)

        elif x['type'] == 'route':
            x['layers'] = x['layers'].split(',')
            try:
                start, end = [int(ind) for ind in x['layers']]
            except:
                start, end = int(x['layers'][0]), 0
            
            if start > 0:
                start = start - index
            if end > 0:
                end = end - index
            
            route = EmptyLayer()
            module.add_module('route_{}'.format(index), route)
            if end < 0:
                filters = output_filters[index + start] + output_filters[index + end]
            else: # not end
                filters = output_filters[index + start]
        
        elif x['type'] == 'shortcut':
            shortcut = EmptyLayer()
            module.add_module('shortcut_{}'.format(index), shortcut)
        
        elif x['type'] == 'yolo':
            mask = x['mask'].split(',')
            mask = [int(x) for x in mask]
            anchors = x['anchors'].split(',')
            anchors = [int(a) for a in anchors]
            anchors = [(anchors[i], anchors[i+1]) for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in mask]

            detection = DetectionLayer(anchors)
            module.add_module('Detection_{}'.format(index), detection)
        
        module_list.append(module)
        prev_filters = filters
        output_filters.append(filters)
        
    return net_info, module_list


class DetectionLayer(nn.Module):

    def __init__(self, anchors):
        super(DetectionLayer, self).__init__()
        self.anchors = anchors
    

class EmptyLayer(nn.Module):

    def __init__(self):
        super(EmptyLayer, self).__init__()


class MaxPoolStride1(nn.Module):
    def __init__(self, kernel_size):
        super(MaxPoolStride1, self).__init__()
        self.kernel_size = kernel_size
        self.pad = kernel_size - 1
    
    def forward(self, x):
        padded_x = F.pad(x, (0, self.pad, 0, self.pad), mode="replicate")
        pooled_x = nn.MaxPool2d(self.kernel_size, self.pad)(padded_x)
        return pooled_x

class Darknet(nn.Module):
    
    def __init__(self, cfgfile):
        super(Darknet, self).__init__()
        self.blocks = parse_cfg(cfgfile)
        self.net_info, self.module_list = create_modules(self.blocks)

    def forward(self, x, CUDA):
        modules = self.blocks[1:]
        outputs = {}
        write = False
        for i, module in enumerate(modules):
            module_type = module['type']
            if module_type in ['convolutional', 'upsample', 'maxpool']:
                x = self.module_list[i](x)

            elif module_type == 'route':
                layers = module['layers']
                layers = [int(a) for a in layers]            
                
                try:
                    start, end = layers
                    if start > 0:
                        start = start - i
                    if end > 0:
                        end = end - i
                    
                    map1 = outputs[i + start]
                    map2 = outputs[i + end]
                    x = torch.cat((map1, map2), 1)

                except:
                    start = layers[0]
                    if start > 0:
                        start = start - i
                    x = outputs[i + start]
                print(i, module_type, x.size())
            elif module_type == 'shortcut':
                from_ = int(module['from'])
                x = outputs[i-1] + outputs[i+from_]

            elif module_type == 'yolo':
                anchors = self.module_list[i][0].anchors
                input_dim = int(self.net_info['height'])
                num_classes = int(module['classes'])

                x = x.data
                x = predict_transform(x, input_dim, anchors, num_classes, CUDA)
                if not write:
                    detections = x
                    write = True
                else:
                    detections = torch.cat((detections, x), 1)

            outputs[i] = x
        
        return detections



if __name__ == '__main__':
    model = Darknet('./cfg/yolov3-tiny.cfg')
    inputs = get_test_input()
    pred = model(inputs, torch.cuda.is_available())
    print(pred.size())
    print(pred)