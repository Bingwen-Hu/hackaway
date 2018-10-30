# why using t-dependent test?
# for example, the same group student attend a test before
# and after some training
# the key is `the same`
import numpy as np
from scipy import stats


def t_value_dependent(X, X_):
    """
    D = X - X_
                        sum(D)
    t = ---------------------------------------
        sqrt([n * sum(D^2) - sum(D)^2] / (n-1))
    """
    D = X - X_
    n = X.shape[0]
    t = np.sum(D) / np.sqrt((n * np.sum(D**2) - np.sum(D)**2) / (n - 1))
    return np.abs(t)

if __name__ == '__main__':
    X = np.array([
        3, 5, 4, 6, 5, 5, 4, 5, 3, 6, 7, 8, 
        7, 6, 7, 8, 8, 9, 9, 8, 7, 7, 6, 7, 8,
    ])
    X_ = np.array([
        7, 8, 6, 7, 8, 9, 6, 6, 7, 8, 8, 7,
        9, 10, 9, 9, 8, 8, 4, 4, 5, 6, 9, 8, 12,
    ])
    t = t_value_dependent(X, X_)
    p = 0.05
    # for we test one side, so p * 2, this is python trick
    left, right = stats.t.interval(1-p*2, df=X.shape[0]-1)
    # and we just get the positive one
    threshold = right
    if t > threshold:
        msg = "reject original hypothesis."
    else:
        msg = 'cannot reject original hypothesis'
    # so our result is
    result = 't[24] = %.2f, p < .05' % t
    print(result)