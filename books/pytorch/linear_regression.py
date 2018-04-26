import torch
from torch.autograd import Variable

x_data = Variable(torch.Tensor([[1.0], [2.0], [3.0]]))
y_data = Variable(torch.Tensor([[2.0], [4.0], [6.0]]))

class Model(torch.nn.Module):

    def __init__(self):
        """In the construtor we instantiate two nn.Linear module
        one in and one out"""
        super(Model, self).__init__()
        self.linear = torch.nn.Linear(1, 1)
    
    def forward(self, x):
        """
        In the forward function we accept a Variable of input data and we must return 
        a Variable of output data. We can use Modules defined in the constructor as 
        well as arbitrary operators on Variables.
        """
        y_pred = self.linear(x)
        return y_pred

# our model
model = Model()

# construct our loss function and an Optimizer. The call to model.parameters()
# in the SGD constructor will contain the learnable parameters of the two
# nn.Linear modules which are members of the model.
criterion = torch.nn.MSELoss(size_average=False)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# training loop
for epoch in range(500):
    y_pred = model(x_data)

    # compute and print loss
    loss = criterion(y_pred, y_data)
    print(epoch, loss.data[0])

    # zero gradients, perform a backward pass, and update the weights
    optimizer.zero_grad() # mini-batch training
    loss.backward()
    optimizer.step()

# after training
hour_var = Variable(torch.Tensor([[4.0]]))
y_pred = model(hour_var)
print('predict (after training)', 4, model(hour_var).data[0][0])