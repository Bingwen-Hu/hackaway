import numpy as np


d1 = np.array([
    [1, 2, 3],
    [11, 12, 13],
])
d2 = np.array([
    [6, 7, 8],
    [66, 77, 88],
])

dc = np.concatenate([d1, d2], axis=0)
dc_ = np.hstack([d1, d2])
dv = np.concatenate([d1, d2], axis=1)
dv_ = np.vstack([d1, d2])