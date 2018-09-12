from model import Net
from config import args
import torch.optim as optim
import torch.nn as nn
import torch
from torch.autograd import Variable

from preprocess import train_data_iterator, test_data_helper



if __name__ == '__main__':
    if args.restore:
        net = torch.load(args.model)
        print('restore model from {}'.format(args.model))
    else:
        net = Net(args.wordset_size)
    print(net)
    net.cuda()
    net.train()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters())
    best_acc = 0
    for epoch_ in range(args.epochs):
        for batch_idx, (data, target) in enumerate(train_data_iterator()):
            data, target = torch.FloatTensor(data).cuda(), torch.LongTensor(target).cuda()

            output = net(data)
            loss = loss_fn(output, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print("Train epoch: {} batch: {} loss: {}".format(epoch_, batch_idx, loss))

            if batch_idx % 100 == 0:
                net.eval()
                with torch.no_grad():
                    data, target = test_data_helper()
                    data, target = torch.FloatTensor(data).cuda(), torch.LongTensor(target).cuda()
                    output = net(data)
                    pred_t = target
                    pred_p = torch.argmax(output, 1)
                    num = pred_t.size(0)
                    sum = torch.sum(torch.eq(pred_t, pred_p))
                    accuracy = sum.item() / num
                    print('accuracy : {}'.format(accuracy))
                    if accuracy > best_acc:
                        net.cpu()
                        torch.save(net, 'best.pth')
                        best_acc = accuracy
                        print("save model. best accuracy is {}".format(best_acc))
                        net.cuda()
                net.train()
            