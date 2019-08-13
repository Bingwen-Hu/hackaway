import numpy as np


mat = np.random.randn(100, 100)
mat = np.where(mat > 0, 1, 0)
xs, ys = np.nonzero(mat)

coordinate = np.array([ys, xs]).T


# accessor
mat = np.arange(10000).reshape(100, 20, 5)
xs = [10, 40, 30, 4, 1, 2] # < 100
ys = [0, 10, 19, 18, 2, 3] # < 20
accessor = np.zeros([4, len(xs)], dtype=np.intp)
accessor[0, :] = xs

accessor[1, :] = ys
accessor[2, :] = 1
accessor[3, :] = 2
# should work
points = mat[xs, ys, accessor[2:4, :]]
channels = [[1], [2]]
points = mat[xs, ys, channels]
print(points.shape)
# points = mat[accessor[0, :], accessor[1, :] ,accessor[2:4, :]]