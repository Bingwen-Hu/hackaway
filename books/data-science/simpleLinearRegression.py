"""
The simplest model: y = alpha + beta * x + error
the error is uncontrolled factor affected the result. we ignore it in our model
so given out ALPHA and BETA, we can compute the yi value
The predict error given by:
prediected_error = y - yi
So we just need to minimize the predict_error

# some useful numpy function
np.std(array)                       # standard deviation
np.var(array)                       # variance
np.correlate(v1, v2)                
correlation(x, y) = covariance(x, y) / (std(x) * std(y)) when std(x)*std(y)!=0
covariance(x, y) = (x - avg(x)) * (y - avg{y}) / (n - 1)
"""

import numpy as np

def predict(alpha, beta, x_i):
    return beta * x_i + alpha

def error(alpha, beta, x_i, y_i):
    return y_i - predict(alpha, beta, x_i)

def sum_of_squared_errors(alpha, beta, x, y):
    return sum(error(alpha, beta, x_i, y_i) ** 2
               for x_i, y_i in zip(x, y))

# the three function do the things can be imployed as the following one:
def numpy_sum_of_squared_errors(alpha, beta, x, y):
    error = np.array(y) - (beta * np.array(x) + alpha)
    squared_error = np.square(error)
    return np.sum(squared_error)

# least squares fit need some calculus
def least_squares_fit(x, y):
    beta = numpy_correlation(x, y) * np.std(y) / np.std(x)
    alpha = np.mean(y) - beta * np.mean(x)
    return np.double(alpha), np.double(beta)


def total_sum_of_squares(y):
    return sum((np.array(y) - np.average(y))**2)

def r_squared(alpha, beta, x, y):
    return 1.0 - (sum_of_squared_errors(alpha, beta, x, y) /
                  total_sum_of_squares(y))



# some helpful function:
def numpy_covariance(x, y):
    x_ = np.array(x) - np.average(x)
    y_ = np.array(y) - np.average(y)
    return np.dot(x_, y_) / (len(x) -1)

def numpy_correlation(x, y):
    return numpy_covariance(x, y) / (np.std(x) * np.std(y))

# NOTE: Given the beta as
# beta = correlation(x, y) * std(y) / std(x)
#      = covariance(x, y) / std(x) / std(y) * std(y) / std(x)
#      = covariance(x, y) / (std(x) * std(x))
# very strange!



# reference to the multipleRegression.py
