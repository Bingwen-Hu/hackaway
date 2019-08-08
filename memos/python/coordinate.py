import numpy as np


mat = np.random.randn(100, 100)
mat = np.where(mat > 0, 1, 0)
xs, ys = np.nonzero(mat)

coordinate = np.array([ys, xs]).T