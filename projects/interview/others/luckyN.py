# 幸运N串，允许最多改变两个大写字母，求得最大的N串长度
tests = [
    'NNNGN',
    'NNNNNSSNN',
    'NNNNXXNNXXNNNNNNN',
    'NNNXNNNNNNXNNNNNNXXNNNNNNNNNNNNNNNNN'
]

def lucky(string):
    # 存放最终结果
    best = 0
    # 记录可以改变的次数
    chance = 0
    # 记录回溯位置
    back = [0, 0]
    # 当前长度
    length = 0
    # 字符串索引
    i = 0
    while i < len(string):
        c = string[i]
        # 如果是N，直接加
        if c == 'N':
            length += 1
            i += 1
        # 如果不是N，但还有改变次数
        elif chance < 2:
            length += 1
            # 记录需要改变的位置，用于回溯
            back[chance] = i
            chance += 1
            i += 1
        else:
            # 否则进行回溯
            best = max(length, best)
            chance = 0
            length = 0
            # 回溯到第1次使用修改机会的位置的后一位
            i = back[0] + 1
    best = max(best, length)
    return best

for s in tests:
    best = lucky(s)
    print(best)