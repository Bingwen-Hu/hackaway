# 计算扔N个不同骰子的最大值的期望

N = 2
dices = [2, 2]
# i ii =>
# 1 2  2
# 2 1  2
# 1 1  1
# 2 2  2
# expect: (2 + 2 + 1 + 2) / 4 = 1.75
expect = 1.75


import sys
# lines = sys.stdin.readlines()
# number = int(lines[0].strip())
# dices = [int(x) for x in lines[1].strip().split()]

# 对于两个dice的数目相同的情况，最大值为
# [2i-1 for i in N], i > 0
# 组合数量为 N x N
# 期望为两者之商
def max_expection_same(N):
    maxs = [(2*i - 1) * i for i in range(1, N+1)]
    expect = sum(maxs) / (N*N)
    return expect

def max_expection_diff(N1, N2):
    maxN = N1 if N1 > N2 else N2
    minN = N1 if N1 <= N2 else N2
    res = maxN - minN
    res_nums = sum([i + minN for i in range(1, res+1)])
    res_sum = minN * res_nums
    same_sum = sum([(2*i - 1) * i for i in range(1, minN+1)])
    expect = (res_sum + same_sum) / (N1 * N2)
    return expect
    


def max_expection(dices):
    pass

