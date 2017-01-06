"""
multiple regression means there are more than one factor affect the result
Given: yi = alpha + beta1 * x1 + beta2 * x2 ... betan * xn + error
the same as the simpleLinearRegression, error is ignore

here we given two vector 
beta = [alpha, beta1, beta2 ... betan]
x_i = [1, x_i1, ... x_ik]
Note that the constant can be placed either at the head or tail

New assumption:
every column of x is linearly independent or we say it is duplicated
every column of x is uncorrelated with the error we ignore
"""

import numpy as np
import random

def predict(x_i, beta):
    return np.dot(x_i, beta)

def error(x_i, y_i, beta):
    return y_i - predict(x_i, beta)

def squared_error(x_i, y_i, beta):
    return error(x_i, y_i, beta) ** 2

def squared_error_gradient(x_i, y_i, beta):
    """the gradient (with respect to beta) corresponding to the ith term"""
    return [-2 * x_ij * error(x_i, y_i, beta) for x_ij in x_i]
    
def estimate_beta(x, y):
    beta_initial = [random.random() for x_i in x[0]]
    return minimize_stochastic(squared_error,
                               squared_error_gradient,
                               x, y,
                               beta_initial,
                               0.001)

# Gradient descent

def minimize_stochastic(target_fn, gradient_fn, x, y, theta_0, alpha_0=0.01):

    data = zip(x, y)
    theta = theta_0
    alpha = alpha_0
    min_theta, min_value = None, float("Inf")
    iterations_with_no_improvement = 0

    # if we ever go 100 iterations with no improvement, we stop
    while iterations_with_no_improvement < 100:
        value = sum(target_fn(x_i, y_i, theta) for x_i, y_i in data)

        if value < min_value:
            min_theta, min_value = theta, value
            iterations_with_no_improvement = 0
            alpha = alpha_0
        else:
            iterations_with_no_improvement += 1
            alpha *= 0.9

        for x_i, y_i in in_random_order(data):
            gradient_i = gradient_fn(x_i, y_i, theta)
            theta = np.array(theta) - np.array(gradient_i) * alpha

    return min_theta

def in_random_order(data):
    indexes = [i for i, _ in enumerate(data)]
    random.shuffle(indexes)
    for i in indexes:
        yield data[i]
