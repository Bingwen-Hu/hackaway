import numpy as np

t1 = np.array([1,2])
t2 = np.array([
    [0, 3],
    [1, 10],
])

# normal * operation apply row-wise
t3 = t1 * t2
# t3 = array([
#   [0, 6],
#   [1, 20],
# ])
# NOTE: this also happens for +/np.add operation

# in numpy, vector has no meaning of T
# but matrix does
t1.T == t1


# perform math matrix operation
t4 = t1.dot(t2)
# t4 = array([ 2, 23 ])
# here, t1 as horizontal vector and t2 act as normal matrix

# so what if I want to perform and adding along
# column instead of row?
# Maybe you don't need that in Numpy

