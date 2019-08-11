# volumn = m
# minute = t
# full = m1
# full1 = t1
# empty = m2
# empty2 = t2

line = 5
inputs = [
    [10, 2, 1, 5, 2, 5],
    [10, 2, 10, 5, 2, 5],
    [10, 2, 3, 5, 2, 5],
    [100, 100, 3, 4, 4, 3],
    [10000, 1000, 10, 5, 5, 3],
]

answers = [0, 10, 2, 3, 2495]


def pipeline(input):
    m, t, m1, t1, m2, t2 = input
    # 水量为v    
    v = 0
    # 给水管状态
    get_state = True
    # 排水管状态
    give_state = True

    for i in range(t):
        # 根据状态进行给排水
        # 两个水管同时打开
        if get_state and give_state:
            delta = m1 - m2
            if delta > 0:
                v = min(m, v + delta)
            else:
                v = max(0, v + delta)
        # 只打开了给水管
        elif get_state:
            v = min(m, v + m1)
        # 只打开了排水管
        elif give_state:
            v = max(0, v - m2)

        # 每个时间后开始先修改状态
        if (i+1) % t1 == 0:
            get_state = not get_state
        if (i+1) % t2 == 0:
            give_state = not give_state

    return v


def solution(line, inputs):
    answers = [pipeline(input) for input in inputs]
    return answers

myanswers = solution(line, inputs)
print(myanswers)