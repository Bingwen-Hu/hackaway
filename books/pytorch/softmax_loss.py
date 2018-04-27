import torch
import torch.nn as nn
from torch.autograd import Variable


# cross entropy example
import numpy as np
# one hot
# 0: 1 0 0
# 1: 0 1 0 
# 2: 0 0 1
y = np.array([1, 0, 0])
y_pred1 = np.array([0.7, 0.2, 0.1])
y_pred2 = np.array([0.1, 0.3, 0.6])
print('loss1 = ', np.sum(-y * np.log(y_pred1)))
print('loss2 = ', np.sum(-y * np.log(y_pred2)))

# softmax + CrossEntropy (logSoftmax + NLLLoss)
loss = nn.CrossEntropyLoss()

# target is of size nBatch
# each element in target has to have 0 <= value < nClasses (0-2)
# input is class, not one-hot
y = Variable(torch.LongTensor([0]), requires_grad=False)

# input is of size nBatch x nClasses = 1 x 4
# y_pred are logits (not softmax)
y_pred1 = Variable(torch.Tensor([[2.0, 1.0, 0.1]]))
y_pred2 = Variable(torch.Tensor([[0.5, 2.0, 0.3]]))

l1 = loss(y_pred1, y)
l2 = loss(y_pred2, y)

print("PyTorch Loss1 = ", l1.data, "\nPyTorch Loss2=", l2.data)


# compute more loss at one time
y = Variable(torch.LongTensor([2, 0, 1]), requires_grad=False)

y_pred1 = Variable(torch.Tensor([[0.1, 0.2, 0.9],
                                 [1.1, 0.1, 0.2],
                                 [0.2, 2.1, 0.1]]))
y_pred2 = Variable(torch.Tensor([[0.8, 0.2, 0.3],
                                 [0.2, 0.3, 0.5],
                                 [0.2, 0.2, 0.5]]))
l1 = loss(y_pred1, y)
l2 = loss(y_pred2, y)

print('Batch Loss1=', l1.data, '\nBatch Loss2=', l2.data)