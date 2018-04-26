x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

w = 1.0  # random value


def forward(x):
    return x * w

def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) ** 2

def gradient(x, y):
    return 2 * x * (x * w -y)


# before training 
print('predict (before training', 4, forward(4))

# training loop
for epoch in range(10):
    for x_val, y_val in zip(x_data, y_data):
        # compute gradient
        grad = gradient(x_val, y_val)
        # update delta
        w = w - 0.01 * grad
        print('\tgrad: ', x_val, y_val, round(grad, 2))
        # compute the loss
        l = loss(x_val, y_val)
    print('progress:', epoch, "w=", round(w, 2), "loss=", round(1, 2))

# after training
print('predict (after training)', '4 hours', forward(4))