# the tools I used work properly --> reliability
# the result I got by using analysis tools is what I want --> validity
from coefficient import coefficient
import numpy as np
import pandas as pd
# retest reliability
def retest_reliablity():
    """check whether a test is reliable in different period"""
    X1 = np.array([54, 67, 67, 83, 87, 89, 84, 90, 98, 65])
    X2 = np.array([56, 77, 87, 89, 89, 90, 87, 92, 99, 76])
    r = coefficient(X1, X2)
    return r
    

def parallel_forms_reliability():
    """check whether the same equality and similarity of the test
    on different parallel forms.
    """
    X1 = np.array([4, 5, 3, 6, 7, 5, 6, 4, 3, 3])
    X2 = np.array([5, 6, 5, 6, 7, 6, 7, 8, 7, 7])
    r = coefficient(X1, X2)
    return r

def internal_consisitency_reliability():
    """ using coefficient of Cronbach's
                     k       s_y^2 - sum(s_i^2)
        epsilon = ------- * --------------------
                   k - 1           s_y^2

    """
    data = {
        'proj1': [3, 4, 3, 3, 3, 4, 2, 3, 3, 3],
        'proj2': [5, 4, 4, 3, 4, 5, 5, 4, 5, 3],
        'proj3': [1, 3, 4, 5, 5, 5, 5, 4, 4, 2],
        'proj4': [4, 5, 4, 2, 4, 3, 3, 2, 4, 3],
        'proj5': [1, 3, 4, 1, 3, 2, 4, 4, 3, 2],
    }
    df = pd.DataFrame(data=data, index=range(1, 11))
    k = df.shape[1]
    s_y = df.sum(axis=1).var()
    sum_of_s_i = df.var(axis=0).sum()
    epsilon = (k / (k - 1)) * ((s_y - sum_of_s_i) / s_y)
    return epsilon


def interrater_reliability():
    """determine whether two interrater are consistent with 
    observation
    """
    X1 = np.array([1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1])
    X2 = np.array([1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1])
    r = (X1 == X2).sum() / X1.shape[0]
    return r