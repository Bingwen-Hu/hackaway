import os
import argparse

import torch
import torch.optim as optim
import torch.utils.data as data
import torch.functional as F
from torch.autograd import Variable
import numpy as np
from PIL import Image


from moryvgg import moryVGG16
from datasets import Captcha, test_data_helper



def train(net, epoch, optimizer):

    train_dataset = Captcha(args)
    train_loader = data.DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size)
    
    loss_fn = torch.nn.MultiLabelSoftMarginLoss()


    net.train().cuda()
    for batch_idx, (images, target) in enumerate(train_loader):
        images, target = Variable(images).cuda(), Variable(target).cuda()
        output = net(images)
        loss = loss_fn(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(images), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()
            ))



        

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', help="batch size of model training", type=int, default=32)
    parser.add_argument('--epoch', help="training epoch", type=int, default=1000)
    parser.add_argument('--image_width', help='width of training image', type=int, default=200)
    parser.add_argument('--image_height', help='height of training image', type=int, default=60)
    parser.add_argument('--train_data_dir', help='directory of train data set', type=str, default='crops')
    parser.add_argument('--test_data_dir', help='directory of testing', type=str, default='test')
    parser.add_argument('--captcha_size', help='number of captcha character', type=int, default=4)
    args = parser.parse_args()
    args.charset = 'abcdefghijklmnopqrstuvwxyz' + 'ABCDEFGHIJKLMNOPQRSTUVWXYZ' + '0123456789'
    return args


if __name__ == '__main__':
    args = parse_args()
    net = moryVGG16(num_classes=len(args.charset) * args.captcha_size)
    state_dict = torch.load('myvgg.pkl')
    net.load_state_dict(state_dict)
    
    param_optim = []
    for layer, n in net.named_children():
        for num, p in n.named_children():
            if layer == 'classifier' and num in ('3', '6'):
                for param in p.parameters():
                    param_optim.append(param)
            else:
                for param in p.parameters():
                    param.requires_grad = False
    
    optimizer = optim.SGD(param_optim, lr=0.01, momentum=0.5)


    for i in range(args.epoch):
        train(net, i+1, optimizer)
        torch.save(net.state_dict(), "model_state_dict.pkl")