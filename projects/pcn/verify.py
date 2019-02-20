import torch
import caffe
import numpy as np
from models import PCN1, PCN2, PCN3


def load_caffe():
    pcn1 = caffe.Net('model/PCN-1.prototxt', 'model/PCN.caffemodel', caffe.TEST)
    pcn2 = caffe.Net('model/PCN-2.prototxt', 'model/PCN.caffemodel', caffe.TEST)
    pcn3 = caffe.Net('model/PCN-3.prototxt', 'model/PCN.caffemodel', caffe.TEST)
    return pcn1, pcn2, pcn3


def forward_caffe(img, net):
    net.blobs['data'].data[...] = img
    out = net.forward()
    return out


def setInput(imgpath:str, net):
    image = caffe.io.load_image(imgpath)
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2, 0, 1))
    transformer.set_raw_scale('data', 255)
    transformer.set_channel_swap('data', (2, 1, 0))
    image = transformer.preprocess('data', image)
    return image

def load_torch():
    pcn1 = torch.load('pth/pcn1.pth')
    pcn2 = torch.load('pth/pcn2.pth')
    pcn3 = torch.load('pth/pcn3.pth')
    return pcn1, pcn2, pcn3

def forward_torch(img, net):
    img = img[np.newaxis, :, :, :]
    img = torch.Tensor(img)
    net.eval()
    with torch.no_grad():
        output = net(img)
    return {'cls_prob': output[0], "bbox_reg": output[2], "rotate_prob": output[1]}


def diff(lst_caffe, lst_torch):
    def helper(caffekey, torchkey):
        layer_caffe = lst_caffe[caffekey].data
        layer_torch = lst_torch[torchkey].numpy()
        diff = layer_torch - layer_caffe
        diffsum = np.sum(diff)
        diffmean = np.mean(diff)
        print("\n{}: diffsum={:.4f}, diffmean={:.4f}".format(torchkey, diffsum, diffmean))
    caffekeys = ['conv1_3', 'pool1_3', 'conv2_3', 'pool2_3', 'conv3_3', 'pool3_3']
    caffekeys.extend(['conv4_3', 'fc5_3', 'cls_prob', 'rotate_reg_3', 'bbox_reg_3'])
    torchkeys = ['conv1', 'maxpool1', 'conv2', 'maxpool2', 'conv3', 'maxpool3']
    torchkeys.extend(['conv4', 'fc', 'cls_prob', 'rotate', 'bbox'])
    for ckey, tkey in zip(caffekeys, torchkeys):
        helper(ckey, tkey)




if __name__ == '__main__':
    imgpath = 'imgs/1.jpg'
    nets_caffe = load_caffe()
    nets_torch = load_torch()
    # test net1
    img = setInput(imgpath, nets_caffe[0])
    out_caffe1 = forward_caffe(img, nets_caffe[0])
    out_torch1 = forward_torch(img, nets_torch[0])
    # test net2
    img = setInput(imgpath, nets_caffe[1])
    out_caffe2 = forward_caffe(img, nets_caffe[1])
    out_torch2 = forward_torch(img, nets_torch[1])
    # test net3
    img = setInput(imgpath, nets_caffe[2])
    out_caffe3 = forward_caffe(img, nets_caffe[2])
    out_torch3 = forward_torch(img, nets_torch[2])

    # more test on net3
    # lst_caffe = nets_caffe[2].blobs
    # lst_torch = nets_torch[2].output
    # diff(lst_caffe, lst_torch)
