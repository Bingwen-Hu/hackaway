import sys
import numpy as np

# 前一行是N和M
# 接下来是N行，每行M个元素
lines = sys.stdin.readlines()
N, M = list(map(int, lines[0].split(' ')))

matrix = []
for line in lines[1:]:
    ms = list(map(int, line.split(' ')))
    matrix.append(ms)
    
matrix = np.array(matrix, dtype=np.intp)
# 假设有一个4行(N=4)3列(M=3)的矩阵
# 2 1 3
# 2 1 1
# 2 1 1
# 1 1 5


# 有六个面，上下两个面
top = N * M
down = N * M

# 前面的面积一定是每一列的最大值之和
front = sum(matrix.max(axis=0))
# 根据对称性
back = front

# 左右的面积一定是每一行的最大值之和
left = sum(matrix.max(axis=1))
right = left

# 但还有一些隐藏面积，比如第1行第2列