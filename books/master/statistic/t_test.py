# when use t test?
# two independent group
# sample not larger than 30
import numpy as np
from scipy import stats

def t_value(X1, X2):
    """
            (n1-1)*s1_var + (n2-1)*s2_var 
    temp1 = --------------------------------
                    n1 + n2 - 2

              n1 + n2
    temp2 = ------------
              n1 * n2

            X1_bar - X2_bar
    t = ------------------------
          sqrt(temp1 * temp2)
    """
    X1_bar = np.mean(X1)
    X2_bar = np.mean(X2)
    s1_var = np.var(X1)
    s2_var = np.var(X2)
    n1 = X1.shape[0]
    n2 = X2.shape[0]
    temp1 = ((n1 - 1) * s1_var + (n2 - 1) * s2_var) / (n1 + n2 - 2)
    temp2 = (n1 + n2) / (n1 * n2)
    t = (X1_bar - X2_bar) / np.sqrt(temp1 * temp2)
    return t

def t_value_test():
    X1 = np.array([
        7, 5, 5, 3, 4, 7, 3, 6, 1, 2, 
        10, 9, 3, 10, 2, 8, 5, 5, 8, 1, 
        2, 5, 1, 12, 8, 4, 15, 5, 3, 4,
    ])
    X2 = np.array([
        5, 3, 4, 4, 2, 3, 4, 5, 2, 5,
        4, 7, 5, 4, 6, 7, 6, 2, 8, 7, 
        8, 8, 7, 9, 9, 5, 7, 8, 6, 6,
    ])
    t = t_value(X1, X2)
    print("t_58 = %.2f, p > .05" % t)

    # so t = -0.14
    # what's the threshold when significant level is 5%?
    left, right = stats.t.interval(0.95, df=58)
    print('confident interval [%.3f, %.3f]' % (left, right))
    if left < t < right:
        msg = ("t value live in the confident interval, "
            "accept the original hypothesis")
    else:
        msg = ('t value live out of confident interval, '
            'reject the original hypothesis')
    print(msg)



if __name__ == '__main__':
    t_value_test()