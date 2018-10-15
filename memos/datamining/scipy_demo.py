from scipy import linalg
import numpy as np

mat = np.array([
    [2, 3, 1],
    [4, 9, 10],
    [10, 5, 6],
])

print(mat)

det = linalg.det(mat)
inv = linalg.inv(mat)
print("determinant of the matrix", det)
print("inverse of matrix\n", inv)

# singular value decomposition
comp1, comp2, comp3 = linalg.svd(mat)
print('perform svd')
print(comp1)
print(comp2)
print(comp3)


# stats module
from scipy import stats

# normal distribution with mean 3 and standard deviation 5
rvs_20 = stats.norm.rvs(3, 5, size=20)
print(rvs_20)

cdf_ = stats.beta.cdf(0.42, a=100, b=50)
print(cdf_)