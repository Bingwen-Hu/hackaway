import numpy as np

def gradient_descent(x, y, lr=0.01, threshold=1e-3):
    params = np.array([0, 0, 0])
    x = np.array([x[0], x[1], 1])
    # loss = w1*x + w2*x + 1 * b
    y_h = np.sum(x * params)
    loss = np.abs(y - y_h)
    while loss > threshold:
        # dw1 = x[0]
        # dw2 = x[1]
        # db = 1
        # update
        params = params - lr * x
        y_h = np.sum(x * params)
        loss = np.abs(y - y_h)

    return params