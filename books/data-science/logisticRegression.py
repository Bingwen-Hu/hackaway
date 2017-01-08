"""
 The range of Linear Regression is -Inf to +Inf
 But sometimes we want some metrics range from [-1, 1] or [0, 1]
 At this moment, we can use logistic 

 Compared with linear regression, we using the following formula 
 to fit the model
     y_i = f(x_i * beta) + error_i
 where f is the logistic function.
 
 Note: In linear Regression, minimizing the sum of squared error is the same 
 as choosing the beta vector that maximized the likelihood of the data.
 Here this two aren't equivalent. So we use gradient descent to maximize the 
 likelihood directly.

 Given some beta, our model says that each y should equal 1 with probability 
 f(x * beta) and 0 with 1 - f(x * beta)

 in particular, the pdf for y_i can be written as 
     p(y_i|x_i, beta) = f(x_i * beta)^y_i * (1 - f(x_i * beta))^(1 - y_i)
 same as 
     log L(beta|x_i, y_i) = y_ilog(f(x_i*beta)) + (1-y_i)log(1-f(x_i*beta))

"""
import numpy as np
from functools import reduce
# The logistic function:
def logistic(x):
    return 1.0 / (1 + np.exp(-x)))

def logistic_prime(x):
    "derivative of logistic function"
    return logistic(x) * (1 - logistic(x))

"""
 because log is strictly increasing function, any beta that maximizes the log 
 likelihood also maximizes the likelihood, and vice versa:
"""
def logistic_log_likelihood_i(x_i, y_i, beta):
    if y_i == 1:
        return np.log(logistic(np.dot(x_i, beta)))
    else:
        return np.log(1 - logistic(np.dot(x_i, beta)))
def logistic_log_likelihood(x, y, beta):
    return sum(logistic_log_likelihood_i(x_i, y_i, beta)
               for x_i, y_i in zip(x, y))


# I can't understand
def logistic_log_partial_ij(x_i, y_i, beta, j):
    "here i is the index of the data point, j the index of the derivative"
    return (y_i - logistic(np.dot(x_i, beta))) * x_i[j]

def logistic_log_partial_i(x_i, y_i, beta):
    "the gradient of the log likelihood corresponding to the ith data point"
    
def logistic_log_gradient(x, y, beta):
    return reduce(np.add, 
                  [logistic_log_gradient_i(x_i, y_i, beta)
                   for x_i, y_i in zip(x, y)])


