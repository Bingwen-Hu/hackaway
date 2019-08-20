# 输入木棍，切成几段，使其乘积最大

import sys
n = sys.stdin.read()
n = int(n.strip())

def cut(n):
    best = [0,1,2,3]
    if n == 0:
        return 0
    if n == 1:
        return 1
    if n == 2:
        return 2
    if n == 3:  # m > 1
        return 2
    # 对于大于3的数，每次检查所有可能的组合
    # 比如12的组合可以是
    # 11,1 10,2 9,3 8,4 7,5, 6,6
    # 选其中的最大值
    for j in range(4, n+1):
        maximum = 0
        for i in range(1,j):
            temp = best[i] * best[j-i]
            if temp > maximum:
                maximum = temp
        best.append(maximum) 
    return best[-1]

ans = cut(n)
print(ans)