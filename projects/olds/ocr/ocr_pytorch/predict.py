import torch
from preprocess import test_data_helper
from config import args

net = torch.load(args.model)
net.eval()

def predict():
    data, target = test_data_helper()
    data, target = torch.FloatTensor(data), torch.LongTensor(target)
    output = net(data)
    pred = torch.argmax(output, 1)
    eqs = torch.eq(pred, target)
    accuracy = eqs.sum().item() / len(target)
    print(accuracy)

predict()