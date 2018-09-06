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
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, 3, stride=1, padding=1)     
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, 3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 512, 3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        self.conv5 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(512)
        self.mp = nn.MaxPool2d(2)
        self.dropout = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(512 * 4 * 4, 1024)
        self.fc2 = nn.Linear(1024, output)
        
    
    def forward(self, x):
        batch_size = x.size(0)
        x = self.mp(self.bn1(self.conv1(x)))
        x = self.mp(self.bn2(self.conv2(x)))
        x = self.mp(self.bn3(self.conv3(x)))
        x = self.bn4(self.conv4(x))
        x = self.mp(self.bn5(self.conv5(x)))
        x = x.view(batch_size, -1)
        x = F.leaky_relu(self.fc1(self.dropout(x)))
        x = self.fc2(self.dropout(x))
        return x
    