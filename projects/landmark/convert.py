import caffe
import torch
import torch.nn as nn
import torch.nn.functional as F

from models import LandMark



def convert_PCN1():
    torch_net = PCN1()

    caffe_net = caffe.Net('model/PCN-1.prototxt', "model/PCN.caffemodel", caffe.TEST)
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


def convert_PCN2():
    torch_net = PCN2()

    caffe_net = caffe.Net('model/PCN-2.prototxt', "model/PCN.caffemodel", caffe.TEST)
    caffe_params = caffe_net.params

    mappings = {
        'conv1_2': torch_net.conv1,
        'conv2_2': torch_net.conv2,
        'conv3_2': torch_net.conv3,
        'fc4_2': torch_net.fc,
        'fc5_2': torch_net.cls_prob,
        'fc6_2': torch_net.rotate,
        'bbox_reg_2': torch_net.bbox,
    }

    for k, layer in mappings.items():
        layer.weight.data.copy_(torch.from_numpy(caffe_params[k][0].data))
        layer.bias.data.copy_(torch.from_numpy(caffe_params[k][1].data))
    torch.save(torch_net, 'pth/pcn2.pth')

def convert_PCN3():
    torch_net = PCN3()

    caffe_net = caffe.Net('model/PCN-3.prototxt', "model/PCN.caffemodel", caffe.TEST)
    caffe_params = caffe_net.params

    mappings = {
        'conv1_3': torch_net.conv1,
        'conv2_3': torch_net.conv2,
        'conv3_3': torch_net.conv3,
        'conv4_3': torch_net.conv4,
        'fc5_3': torch_net.fc,
        'fc6_3': torch_net.cls_prob,
        'bbox_reg_3': torch_net.bbox,
        'rotate_reg_3': torch_net.rotate,
    }

    for k, layer in mappings.items():
        layer.weight.data.copy_(torch.from_numpy(caffe_params[k][0].data))
        layer.bias.data.copy_(torch.from_numpy(caffe_params[k][1].data))
    torch.save(torch_net, 'pth/pcn3.pth')

if __name__ == "__main__":
    convert_PCN1()
    convert_PCN2()
    convert_PCN3()
