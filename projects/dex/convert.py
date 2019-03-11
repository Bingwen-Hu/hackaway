import caffe
import torch
import torch.nn as nn
import torch.nn.functional as F

from models import Age, Gender



def convert_Age():
    torch_net = Age()

    caffe_net = caffe.Net('age.prototxt', "dex_chalearn_iccv2015.caffemodel", caffe.TEST)
    caffe_params = caffe_net.params

    mappings = {
        'conv1_1': torch_net.conv1,
        'conv2_1': torch_net.conv2,
        'conv3_1': torch_net.conv3,
        'fc4_1': torch_net.conv4,
        'fc5_1': torch_net.cls_prob,
        'fc6_1': torch_net.rotate,
        'bbox_reg_1': torch_net.bbox,
    }

    for k, layer in mappings.items():
        layer.weight.data.copy_(torch.from_numpy(caffe_params[k][0].data))
        layer.bias.data.copy_(torch.from_numpy(caffe_params[k][1].data))
    torch.save(torch_net, 'pth/pcn1.pth')

