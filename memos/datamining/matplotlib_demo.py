import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def demo1():
    data = np.array([1, 1, 2, 2, 3, 3, 3, 4, 4, 5, 6, 7, 9, 10, 12, 3, 2, 4, 3, 5, 6, 7, 7, 6, 5, 44, 5, 6, 2])
    plt.hist(data, bins=7)
    plt.show()

def demo2():
    data = np.random.normal(3, 0.1, 1000)
    plt.hist(data, bins=40)
    plt.show()

def demo3():
    ints = np.random.randint(1, 100, 40)
    plt.hist(ints, bins=15)
    plt.show()
