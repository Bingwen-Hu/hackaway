import numpy as np
import seaborn as sns

def coefficient(X, Y):
    """
                         n * sum(XY) - sum(X)sum(Y) 
    r_XY =  ---------------------------------------------------------
            sqrt([n * sum(X^2) - sum(X)^2] [n * sum(Y^2) - sum(Y)^2])
    """
    def deno(data):
        """denominator helper"""
        n = data.shape[0]
        sum_of_square = np.sum(data ** 2)
        square_of_sum = np.sum(data) ** 2
        return n * sum_of_square - square_of_sum
    n = X.shape[0]
    numerator = n * np.sum(X * Y) - np.sum(X) * np.sum(Y)
    denominator = np.sqrt(deno(X) * deno(Y))
    return numerator / denominator

def scatterplot(X, Y):
    sns.scatterplot(x=X, y=Y)
    

if __name__ == '__main__':
    X = np.array([2, 4, 5, 6, 4, 7, 8, 5, 6, 7])
    Y = np.array([3, 2, 6, 5, 3, 6, 5, 4, 4, 5])
    R_xy = coefficient(X, Y)
    scatterplot(X, Y)