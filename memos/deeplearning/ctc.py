# CTC implement in pytorch
# https://blog.csdn.net/JackyTintin/article/details/79425866


import numpy as np

np.random.seed(1111)

T, V = 12, 5
m, n = 6, V

x = np.random.random([T, m])
w = np.random.random([m, n])

def softmax(logits):
    max_value = np.max(logits, axis=1, keepdims=True)
    exp = np.exp(logits - max_value)
    exp_sum = np.sum(exp, axis=1, keepdims=True)
    dist = exp / exp_sum
    return dist