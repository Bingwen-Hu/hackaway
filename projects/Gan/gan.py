import torch
import torchvision.transforms as transforms
import torchvision.datasets.mnist as mnist


train_data = mnist.MNIST('./mnist', train=True, 
    transform=transforms.ToTensor(), download=True)
test_data = mnist.MNIST('./mnist', train=False, 
    transform=transforms.ToTensor(), download=True)

