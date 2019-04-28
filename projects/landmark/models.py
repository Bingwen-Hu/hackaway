import torch
import torch.nn as nn
import torch.nn.functional as F


class LandMark(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 20, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(20, 48, kernel_size=5, stride=1, padding=2)
        self.conv3 = nn.Conv2d(48, 64, kernel_size=3, stride=1, padding=0)
        self.conv4 = nn.Conv2d(64, 80, kernel_size=3, stride=1, padding=0)
        self.pool2x2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool3x2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.fc1 = nn.Linear(1280, 512)
        self.fc2 = nn.Linear(512, 136)

    def forward(self, x):
        x = self.conv1(x)
        print('conv1', x.shape)
        x = torch.abs(torch.tanh(x))
        x = self.pool2x2(x)
        print('pool1', x.shape)

        x = self.conv2(x)
        print('conv2', x.shape)
        x = torch.abs(torch.tanh(x))
        x = self.pool2x2(x)
        print('pool2', x.shape)

        x = self.conv3(x)
        print('conv3', x.shape)
        x = torch.abs(torch.tanh(x))
        x = F.pad(x, (0, 1, 0, 1))
        x = self.pool3x2(x)
        print('pool3', x.shape)

        x = self.conv4(x)
        print('conv4', x.shape)
        x = torch.abs(torch.tanh(x))
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        x = torch.abs(torch.tanh(x))
        x = self.fc2(x)
        return x


if  __name__ == '__main__':
    data = torch.randn(1, 1, 60, 60) 
    net = LandMark()
    net.train()
    outputs = net(data)