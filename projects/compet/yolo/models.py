import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


def convolution_block(in_channels, out_channels, kernel_size, **kwargs):
    block = nn.Sequential(OrderedDict([
        ('conv', nn.Conv2d(in_channels, out_channels, kernel_size, **kwargs)),
        ('batch_norm', nn.BatchNorm2d(out_channels)),
        ('leaky', nn.LeakyReLU(0.1)),
    ]))
    return block

class Darknet(nn.Module):
    def __init__(self, image_size=416, channels=3):
        super().__init__()
        self.module_list = nn.Sequential(
            convolution_block(channels, 32, kernel_size=3, padding=1, bias=False),
            convolution_block(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            convolution_block(64, 32, kernel_size=1, bias=False),
            convolution_block(32, 64, kernel_size=3, padding=1, bias=False),
        )

if __name__ == '__main__':
    net = Darknet()
    print(net)