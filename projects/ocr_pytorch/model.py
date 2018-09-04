""" CNN model architecture for captcha
"""

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


from config import args


class Net(nn.Module):

    def __init__(self, output):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, stride=1, padding=1)     
        self.conv3 = nn.Conv2d(128, 256, 3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(256, 512, 3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.mp = nn.MaxPool2d(2)
        self.dropout = nn.Dropout2d(0.5)
        self.fc = nn.Linear(, output)
        self.bn = nn.BatchNorm2d()
    
    def forward(self, x):
        batch_size = x.size(0)
        x = self.mp(self.conv1(x))
        x = nn.MaxPool2d(2)
        x = nn.Conv2d(64, 128, [3, 3], 1)
