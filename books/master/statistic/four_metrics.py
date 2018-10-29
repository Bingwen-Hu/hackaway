# data have four difference metrics
# mean, variability, skerness, kurtosis
import numpy as np
import seaborn as sns
# define:
# if mean > median, then data show positive skewness
# if mean < median, then data show negative skewness
# if mean == median, no skewness
def skewness(data):
    """
        SK = 3 * (X_bar - M) / s
    """
    sns.distplot(data, kde=True)
    mean = np.mean(data)
    median = np.median(data)
    s = np.std(data)
    print("mean {} median, data show {} skewness".format(
        '>' if mean > median else '<',
        'positive' if mean > median else 'negative',
    ))
    sk = 3 * (mean - median) / s
    return sk


def kurtosis(data):
    """flat or steep"""
    mean = np.mean(data)
    s = np.std(data)
    k = np.sum(np.power(((data - mean) / s), 4)) / len(data) - 3
    return k

if __name__ == '__main__':
    data = np.random.randint(1, 100, 50)
    sk = skewness(data)
    k = kurtosis(data)