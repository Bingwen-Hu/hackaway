import torch
import torch.nn as nn
import torch.nn.functional as F




class Net(nn.Module):
    
    def __init__(self, output):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 5, padding=2)
        self.conv2 = nn.Conv2d(32, 64, 5, padding=2)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=2)
        self.conv4 = nn.Conv2d(128, 256, 3, padding=2)
        self.mp = nn.MaxPool2d(2)
        self.dropout = nn.Dropout2d(0.5)
        self.fc = nn.Linear(7168, output)
        

    def forward(self, x):
        in_size = x.size(0)
        x = F.relu(self.mp(self.conv1(x)))
        x = F.relu(self.mp(self.conv2(x)))
        x = F.relu(self.mp(self.conv3(x)))
        x = F.relu(self.mp(self.conv4(x)))
        x = self.dropout(x)
        x = x.view(in_size, -1)
        x = self.fc(x)
        return F.logsigmoid(x)
