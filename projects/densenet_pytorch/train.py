import torch

from models import DenseNet
from config import config
from preprocess import train_data_iterator, test_data_helper



net = DenseNet(
    growth_rate=32,
    block_config=[3, 3, 3],
    num_classes=config.charlen,
    small_inputs=False,
    efficient=True,
)

net = net.cuda()

# Optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, nesterov=True, weight_decay=0.0001)

best_acc = config.baseline


for epoch in range(config.epochs):
    for i, (input, target) in enumerate(train_data_iterator()):
        input_var = torch.autograd.Variable(input.cuda(async=True))
        target_var = torch.autograd.Variable(target.cuda(async=True))
        # feed
        output = net(input_var)
        loss = torch.nn.functional.cross_entropy(output, target_var)
        # backwwrd
        optimizer.zero_grad()
        loss.backwwrd()
        optimizer.step()
        # accuracy

        # loss
        print('step: %s, loss: %.4f' % (i, loss))
