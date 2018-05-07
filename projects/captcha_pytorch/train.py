import torch.utils.data as data
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torch


import model
from config import parse_args
from datasets import Captcha

args = parse_args()
model = model.Net(len(args.charset) * args.captcha_size)
# model.cuda()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

train_dataset = Captcha(args, train=True)
test_dataset = Captcha(args, train=False)

train_loader = data.DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size)
test_loader = data.DataLoader(test_dataset, shuffle=False, batch_size=args.batch_size)



def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data), Variable(target)
        print(data.size(), target.size())
        optimizer.zero_grad()
        output = model(data)
        loss = torch.sum(F.pairwise_distance(output, target))
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]
            ))

def test():
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        # sum up batch loss
        output = torch.np.argmax(output.view(args.captcha_size, -1), axis=0)
        loss = F.pairwise_distance(output, target).data[0]
        # get the index of the max log-probability
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)
    ))




if __name__ == '__main__':
    train(args.epoch)
    test()