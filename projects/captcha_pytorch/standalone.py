import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import torchvision.models as models
from torchvision import transforms, utils
from torch.autograd import Variable
from PIL import Image
import numpy as np
import os


charset = 'abcdefghijlkmnopqrstuvwxyz'
data_dir = "/media/mory/lisp/captcha-data/dwnews/train"
batch_size = 16
epoch = 10


def equal(np1,np2):  
    n = 0
    for i in range(np1.shape[0]):
        if (np1[i,:]==np2[i,:]):
            n += 1
    return n

class Captcha(data.Dataset):
    
    def __init__(self, data_dir, charset, transform=None):
        self.path = [os.path.join(dir, f) for dir, _, files in 
                     os.walk(data_dir) for f in files]
        self.transform = transform
        self.charset = charset
    
    def __getitem__(self, idx):
        image_path = self.path[idx]
        image = Image.open(image_path)
        image = image.convert("RGB")
        image = image.resize((160, 60))
        label = os.path.basename(image_path)[:4]
        label = self.text2vec(label)

        if self.transform is not None:
            image = self.transform(image)
        
        return image, label

    def __len__(self):
        return len(self.path)

    def text2vec(self, text):
        def char2vec(c):
            y = np.zeros(len(self.charset))
            y[self.charset.index(c)] = 1.0
            return y
        vec = np.vstack([char2vec(c) for c in text])
        vec = vec.flatten()
        return torch.FloatTensor(vec)


dataset = Captcha(data_dir=data_dir, charset=charset, transform=transforms.ToTensor())
dataloader = data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
dataset_size = len(dataset)

class ConvNet(nn.Module):

    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(32, 64, kernel_size=4, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(kernel_size=2)
        )

        self.fc1 = nn.Linear(64 * 7 * 20, 500)
        self.fc2 = nn.Linear(500, 26 * 4)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        output = self.fc1(x)
        output = self.fc2(output)
        return output

class nCrossEntropyLoss(torch.nn.Module):

    def __init__(self, n=4):
        super(nCrossEntropyLoss, self).__init__()
        self.n = n
        self.total_loss = 0
        self.loss = nn.CrossEntropyLoss()

    def forward(self, output, label):
        output_t = output[:, 0:26]
        label = Variable(torch.LongTensor(label.data.cpu().numpy()))
        label_t = label[:, 0]

        for i in range(1, self.n):
            output_t = torch.cat((output_t, output[:, 26*i:26*i+26]), 0)
            label_t = torch.cat((label_t, label[:, i]), 0)
            self.total_loss = self.loss(output_t, label_t)

        return self.total_loss

net = ConvNet()
optimizer = optim.Adam(net.parameters(), lr=0.001)
loss_func = nCrossEntropyLoss()

import copy
best_model_wts = copy.deepcopy(net.state_dict())
best_acc = 0.0

for ep in range(epoch):
    running_loss = 0.0
    running_corrects = 0.0

    for step, (inputs, labels) in enumerate(dataloader):
        pred = torch.LongTensor(batch_size, 1).zero_()
        inputs = Variable(inputs)
        labels = Variable(labels)

        optimizer.zero_grad()
        output = net(inputs) 
        loss = loss_func(output, labels)

        for i in range(4):
            pre = F.log_softmax(output[:, 10*i:10*i+10], dim=1)
            pred = torch.cat((pred, pre.data.max(1, keepdim=True)[1].cpu()), dim=1)

        loss.backward()
        optimizer.step()

        running_loss += loss.data[0] * inputs.size()[0]
        running_corrects += equal(pred.numpy()[:, 1:], labels.data.cpu().numpy().astype(int))
        print("step {}".format(step))
    epoch_loss = running_loss / dataset_size
    epoch_acc = running_corrects / dataset_size

    if epoch_acc > best_acc:
        best_acc = epoch_acc
        best_model_wts = copy.deepcopy(net.state_dict())

    if ep == epoch - 1:
        torch.save(best_model_wts, "best_model_wts.pkl")

    print("training loss: %4f, training accuracy %4f\n" % (epoch_loss, epoch_acc))

