# nxm方格中第k大的数

n = 7
m = 3
matrix = []
# 1   2   3
# 2   4   6
# 3   6   9
# 4   8   12
# 5   10  15
# 6   12  18
# 7   14  21
# <- -n
# ^ -m
k = 3
v = 15

maxminum = matrix[n-1][m-1]
# 暴力法
def force(n, m, k):
    matrix = [n_ * m_ for n_ in range(1, n+1) for m_ in range(1, m+1)]
    matrix.sort(reverse=True)
    return matrix[k-1]
