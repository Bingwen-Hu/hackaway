import torch
import torch.nn as nn
import torch.nn.functional as F




class Net(nn.Module):
    
    def __init__(self, output):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1,    32, 5, stride=1, padding=2) 
        self.conv2 = nn.Conv2d(32,   64, 5, stride=1, padding=2) 
        self.conv3 = nn.Conv2d(64,  128, 3, stride=1, padding=2)
        self.conv4 = nn.Conv2d(128, 256, 3, stride=1, padding=2)
        self.mp = nn.MaxPool2d(2)
        self.dropout = nn.Dropout2d(0.5)
        self.fc = nn.Linear(7168, output)
        

    def forward(self, x):
        in_size = x.size(0)                     # (bs,  1, 40, 100)
        x = F.relu(self.mp(self.conv1(x)))      # (bs, 32, 20, 50)      
        x = F.relu(self.mp(self.conv2(x)))      # (bs, 64, 10, 25)
        x = F.relu(self.mp(self.conv3(x)))      # (bs, 128, 5, 12)
        x = F.relu(self.mp(self.conv4(x)))      # (bs, 256, 2, 6)
        x = self.dropout(x)
        x = x.view(in_size, -1)
        x = self.fc(x)
        return F.logsigmoid(x)
