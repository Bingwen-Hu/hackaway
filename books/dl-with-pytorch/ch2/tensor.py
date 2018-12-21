import torch

# scalar
x = torch.rand(1)
x.size()


# vector
temp = torch.FloatTensor([23, 24, 25, 5, 33, 42, 0, 2])
temp.size()

# matrix
import numpy as np

data = np.random.randn(100, 20)
tensor = torch.from_numpy(data)
tensor.size()

# back to numpy
data_ = tensor.numpy()
