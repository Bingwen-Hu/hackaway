from model import Net
from config import args
import torch.optim as optim
import torch.nn as nn
import torch
from torch.autograd import Variable

from preprocess import train_data_iterator, test_data_helper



if __name__ == '__main__':
    net = Net(args.wordset_size)
    net.train()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.RMSprop(net.parameters())
    
    for epoch_ in range(args.epochs):
        for batch_idx, (data, target) in enumerate(train_data_iterator()):
            data, target = torch.FloatTensor(data), torch.FloatTensor(target)
            optimizer.zero_grad()
            output = net(data)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()
            print("Train epoch: {} batch: {} loss: {}".format(epoch_, batch_idx, loss))

            if batch_idx % 100 == 0:
                net.eval()
                with torch.no_grad():
                    data, target = test_data_helper()
                    data, target = torch.FloatTensor(data), torch.LongTensor(target)
                    output = net(data)
                    pred_t = torch.argmax(target, 1)
                    pred_p = torch.argmax(output, 1)
                    num = pred_t.size(0)
                    sum = torch.sum(torch.eq(pred_t, pred_p))
                    accuracy = sum / num
                    print('accuracy : {}'.format(accuracy))
                net.train()

