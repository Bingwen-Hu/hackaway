import sys

# lines = sys.stdin.readlines()
# n, m = list(map(int, lines[0].split(' ')))
# num1 = list(map(int, lines[1].split(' ')))
# num2 = list(map(int, lines[2].split(' ')))

n, m = 5, 5
num1 = [4, 4, 1, 1, 1]
num2 = [4, 3, 0, 1, 2]

# 返回值
ret = []
# 暴力法
maximum = -1
while len(num1) > 0:
    for n1 in num1:
        for n2 in num2:
            res = (n1 + n2) % m
            if res > maximum:
                maximum = res
                xi = n1
                xj = n2

    ret.append(maximum)
    num1.remove(xi)
    num2.remove(xj)
    maximum = -1


ret = [str(i) for i in ret]
print(" ".join(ret))