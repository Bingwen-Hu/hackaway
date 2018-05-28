import os
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as data
from torch.autograd import Variable
from PIL import Image




class Net(nn.Module):
    
    def __init__(self, output):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1,    32, 3, stride=1, padding=1) 
        self.conv2 = nn.Conv2d(32,   64, 3, stride=1, padding=1) 
        self.conv3 = nn.Conv2d(64,  128, 3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(128, 256, 3, stride=1, padding=1)
        self.mp = nn.MaxPool2d(2)
        self.dropout = nn.Dropout2d(0.5)
        self.fc = nn.Linear(256 * 2 * 6, output)
        

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


class Captcha(data.Dataset):

    def __init__(self, args, train=True):
        self.batch_size = args.batch_size
        self.width = args.image_width
        self.height = args.image_height
        self.charset = args.charset
        self.textlen = args.captcha_size
        
        data_dir = args.train_data_dir if train else args.test_data_dir
        self.path = [os.path.join(dir, f) for dir, _, files in 
                     os.walk(data_dir) for f in files]
        
    def __getitem__(self, index):
        return self.get_X(index), self.get_y(index)

    def __len__(self):
        return len(self.path)
    
    def get_y(self, index):
        basename = os.path.basename(self.path[index])
        text = basename[:self.textlen]
        vec = self.text2vec(text)
        return vec

    def get_X(self, index):
        """对path指向的文件处理"""
        img = Image.open(self.path[index])
        img = img.resize([self.width, self.height])
        img = img.convert('L')
        img = np.array(img) / 255
        return torch.FloatTensor(img).view(-1, self.height, self.width)
    
    def text2vec(self, text):
        def char2vec(c):
            y = np.zeros(len(self.charset))
            y[self.charset.index(c)] = 1.0
            return y
        vec = np.vstack([char2vec(c) for c in text])
        vec = vec.flatten()
        return torch.FloatTensor(vec)




def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', help="batch size of model training", type=int, default=64)
    parser.add_argument('--epoch', help="training epoch", type=int, default=100)
    parser.add_argument('--image_width', help='width of training image', type=int, default=100)
    parser.add_argument('--image_height', help='height of training image', type=int, default=40)
    parser.add_argument('--train_data_dir', help='directory of train data set', type=str, default='E:/captcha-data/dwnews/smalltrain')
    parser.add_argument('--test_data_dir', help='directory of testing', type=str, default='E:/captcha-data/dwnews/test')
    parser.add_argument('--captcha_size', help='number of captcha character', type=int, default=4)
    args = parser.parse_args()
    args.charset = 'abcdefghijklmnopqrstuvwxyz'
    return args



if __name__ == "__main__":
    
    args = parse_args()
    net = Net(len(args.charset) * args.captcha_size)
    # model.cuda()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

    train_dataset = Captcha(args, train=True)
    test_dataset = Captcha(args, train=False)

    train_loader = data.DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size)
    test_loader = data.DataLoader(test_dataset, shuffle=False, batch_size=args.batch_size)

