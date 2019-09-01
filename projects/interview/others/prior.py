import sys

lst, N = sys.stdin.read().split(';')
lst = [int(i) for i in lst.split(',')]
N = int(N)

even = []
odd = []
for i in lst:
    if i % 2 == 0:
        even.append(i)
    else:
        odd.append(i)
even.sort(reverse=True)
odd.sort(reverse=True)

result = even + odd
result = result[:N]
result = list(map(str, result))
print(",".join(result))