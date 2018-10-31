import numpy as np
import matplotlib.pyplot as plt


def fit_line(X, Y):
    n = X.shape[0]
    temp1 = np.sum(X * Y) - np.sum(X) * np.sum(Y) / n
    temp2 = np.sum(X * X) - np.sum(X) ** 2 / n
    b = temp1 / temp2
    a = (np.sum(Y) - b * np.sum(X)) / n
    return lambda x: b * x + a

if __name__ == '__main__':
    cpa_1 = np.array([3.5, 2.5, 4.0, 3.8, 2.8, 1.9, 3.2, 3.7, 2.7, 3.3])
    cpa_2 = np.array([3.3, 2.2, 3.5, 2.7, 3.5, 2.0, 3.1, 3.4, 1.9, 3.7])
    
    plt.scatter(cpa_1, cpa_2)