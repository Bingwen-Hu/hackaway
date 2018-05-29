import os
import argparse

import torch
import torch.optim as optim
import torch.utils.data as data
import torch.functional as F
from torch.autograd import Variable
import numpy as np
from PIL import Image
from vgg import vgg16


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
        img = img.convert('RGB')
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


def train(net, epoch):
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.5)

    train_dataset = Captcha(args, train=True)
    train_loader = data.DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size)
    
    net.train()
    for epoch_ in range(epoch):
        for batch_idx, (images, target) in enumerate(train_loader):
            images, target = Variable(images), Variable(target)
            optimizer.zero_grad()
            output = net(images)
            loss_fn = torch.nn.BCEWithLogitsLoss(reduce=True)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % 10 == 0:
                print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch_ + 1, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()
                ))
        test(net)

def test(net):
    # test_dataset = Captcha(args, train=False)
    # test_loader = data.DataLoader(test_dataset, shuffle=False, batch_size=args.batch_size)
    pass

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', help="batch size of model training", type=int, default=64)
    parser.add_argument('--epoch', help="training epoch", type=int, default=1000)
    parser.add_argument('--image_width', help='width of training image', type=int, default=200)
    parser.add_argument('--image_height', help='height of training image', type=int, default=60)
    parser.add_argument('--train_data_dir', help='directory of train data set', type=str, default='E:/captcha-data/sina2/gentrain')
    parser.add_argument('--test_data_dir', help='directory of testing', type=str, default='E:/captcha-data/sina2/realtest2')
    parser.add_argument('--captcha_size', help='number of captcha character', type=int, default=4)
    args = parser.parse_args()
    args.charset = 'abcdefghijklmnopqrstuvwxyz' + 'ABCDEFGHIJKLMNOPQRSTUVWXYZ' + '0123456789'
    return args


if __name__ == '__main__':
    args = parse_args()
    net = vgg16(num_classes=len(args.charset) * args.captcha_size)
    train(net, 10)