import numpy as np


def chi_square(X):
    """
                (O - E)^2
    X^2 = sum  -----------
                    E
    """
    n = X.shape[0]
    E = np.sum(X) / n
    chi_2 = sum((X - E) ** 2 / E)
    return chi_2

if __name__ == '__main__':
    X = np.array([23, 17, 50])
    chi_2 = chi_square(X)
    print('Chi square is: {}'.format(chi_2))
    # another table lookup