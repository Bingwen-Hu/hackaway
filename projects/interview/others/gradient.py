# Buggy code
# Gradient for y = w1

import numpy as np

def gradient_descent(x, y, lr=0.01, threshold=1e-3):
    params = np.array([0.1, 0.2, 0])
    x = np.array([x[0], x[1], 1])
    y_h = np.sum(x * params)
    loss = 1 / 2 * np.square(y - y_h)
    while loss > threshold:
        params = params - lr * x * -(y - y_h)
        y_h = np.sum(x * params)
        loss = 1 / 2 * np.square(y - y_h)
        print(loss)

    return params, loss

x = [-1, 0.4]
y = 20
params, loss = gradient_descent(x, y)

# validate 
x += [1]
y_hat = sum(np.array(x) * params)

print(y, y_hat)