# normal curve:
# mean == median == mode

# how we judge something is impossible?
# We say, if P(X) < 5%, we consider it as impossible
# in a normal distribution,  P(z > 1.65) < 5%




import numpy as np
import seaborn as sns

def z_score(data):
    mean = np.mean(data)
    std = np.std(data)
    z = (data - mean) / std
    return z

if __name__ == "__main__":
    data = np.random.normal(100, 10, 100)
    data_z = z_score(data)
    mean = np.mean(data_z)
    std = np.std(data_z)
    print("Mean: %s, Std: %s" % (mean, std))