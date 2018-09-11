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
        self.conv1 = nn.Conv2d(1, 32, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, stride=1, padding=1)     
        self.conv4 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.mp = nn.MaxPool2d(2)
        self.dropout = nn.Dropout2d(0.5)
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(16 * 16 * 64, 1024)
        self.fc2 = nn.Linear(1024, output)
        
    
    def forward(self, x):
        batch_size = x.size(0)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.mp(x)
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.mp(x)
        x = x.view(batch_size, -1)
        x = self.dropout(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        x = F.softmax(x)
        return x
    