import torch
from torch.autograd import Variable

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

# enable grad, we omit our gradient function
w = Variable(torch.Tensor([1.0]), requires_grad=True)

# because w is a tensor, so fn forward return a tensor
def forward(x):
    return x * w

def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) ** 2


print("predict (before training)", 4, forward(4).data[0])

for epoch in range(10):
    for x_val, y_val in zip(x_data, y_data):
        l = loss(x_val, y_val)
        # gradient
        l.backward()
        print("\tgrad: ", x_val, y_val, w.grad.data[0])
        # update weight
        w.data = w.data - 0.01 * w.grad.data
        # why set it to zero?
        w.grad.data.zero_()
    print('progress:', epoch, l.data[0])

print('predict (after training)', 4, forward(4).data[0])