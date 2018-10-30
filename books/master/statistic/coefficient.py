import numpy as np
import seaborn as sns


def coefficient(X, Y):
    """ Pearson correlation coefficient
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
    denominator = np.sqrt(deno(X)) * np.sqrt(deno(Y)) # avoid overflow
    return numerator / denominator

    

def demo():
    X = np.array([2, 4, 5, 6, 4, 7, 8, 5, 6, 7])
    Y = np.array([3, 2, 6, 5, 3, 6, 5, 4, 4, 5])
    R_xy = coefficient(X, Y)
    sns.scatterplot(x=X, y=Y)

def correlation_coefficient_test():
    marriage_quality = np.array([
        76, 81, 78, 76, 76, 78, 76, 78, 98, 88, 76, 66, 44, 
        67, 65, 59, 87, 77, 79, 85, 68, 76, 77, 98, 99, 98, 
        87, 67, 78,
    ])
    parent_children_relation = np.array([
        43, 33, 23, 34, 31, 51, 56, 43, 44, 45, 32, 33, 28,
        39, 31, 38, 21, 27, 43, 46, 41, 41, 48, 56, 55, 45, 
        68, 54, 33,
    ])
    coefficient_ = coefficient(marriage_quality, parent_children_relation)
    # left the same problem with anova.py
    # never mind
    return coefficient_

if __name__ == '__main__':
    demo()