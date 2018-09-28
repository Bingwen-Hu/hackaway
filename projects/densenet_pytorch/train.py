import torch
import torch.nn as nn
from models import DenseNet
from config import config
from preprocess import train_data_iterator, test_data_helper



net = DenseNet(
    growth_rate=32,
    block_config=[3, 3, 3],
    num_classes=config.charlen * config.captlen,
    small_inputs=False,
    efficient=True,
)

net = net.cuda()

# Optimizer
optimizer = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9, nesterov=True, weight_decay=0.0001)

best_acc = config.baseline

loss_fn = torch.nn.BCEWithLogitsLoss(reduce=False)

for epoch in range(config.epochs):
    for i, (input, target) in enumerate(train_data_iterator()):
        input = torch.FloatTensor(input)
        target = torch.LongTensor(target)
        input_var = torch.autograd.Variable(input.cuda(async=True))
        target_var = torch.autograd.Variable(target.cuda(async=True))
        # feed
        output = net(input_var)
        loss = loss_fn(output, target)
        # backwwrd
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # accuracy
        if (i+1) % 100 == 0:
            net.eval()
            data, label = test_data_helper()
            data, label = Variable(data).cuda(), Variable(label).cuda()
            with torch.no_grad():
                output = net(data)
                batch_size = output.size(0)
                pred = torch.argmax(output.view(-1, config.captlen, config.charlen), dim=2)
                label = torch.argmax(label.view(-1, config.captlen, config.charlen), dim=2)
                accuracy = torch.eq(pred, label).sum().item() / (batch_size * config.captlen)
            if accuracy > best_acc:
                best_acc = accuracy
                net.save('best.pth')
                print('Make sure update baseline when you retrain the model. best accuracy: %.3f' % best_acc)
        # loss
        print('step: %s, loss: %.4f' % (i, loss))
