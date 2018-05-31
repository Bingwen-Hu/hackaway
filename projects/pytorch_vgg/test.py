# -*- coding: UTF-8 -*-
import numpy as np
import torch
from torch.autograd import Variable
from toy_inception import Net

from datasets import test_data_helper
from train import parse_args

def test(net, args):

    with torch.no_grad():
        images, target = Variable(images).cuda(), Variable(target).cuda()
        images = images.view(-1, 1, args.image_height, args.image_width)
        output = net(images)
        output = output.view(-1, args.captcha_size, len(args.charset))
        target = target.view(-1, args.captcha_size, len(args.charset))
        pred_o = torch.argmax(output, dim=2)
        pred_t = torch.argmax(target, dim=2)
        same   = torch.eq(pred_o, pred_t).sum().float()
        batch  = pred_t.size()[0] * args.captcha_size * len(args.charset)
        print("accuracy is %.3f\n" % (same / batch))

        return pred_o, pred_t, same
if __name__ == '__main__':
    # restore model 
    args = parse_args()
    net = Net(len(args.charset) * args.captcha_size)
    net.load_state_dict(torch.load('state_dict.pkl'))
    print("load cnn net.")

    images, target = test_data_helper(args)
    net.eval()
    with torch.no_grad():
        images, target = Variable(images), Variable(target)
        images = images.view(-1, 1, args.image_height, args.image_width)
        predict_label = net(images)
        predict_label = predict_label.view(-1, args.captcha_size, len(args.charset))
        predict_label = torch.argmax(predict_label, dim=2)
        indece = predict_label.numpy()
        for index in indece:
            (c0, c1, c2, c3) = [args.charset[i] for i in index]
            c = '%s%s%s%s' % (c0, c1, c2, c3)
            print(c)

