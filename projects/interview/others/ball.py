# who get red first who win
# NOTE: BUG

import sys
red, blue = sys.stdin.read().strip().split()
red, blue = int(red), int(blue)


def lastround(red, blue):
    assert red + blue <= 3
    win = red / (red + blue)
    return win

def redball(red, blue, factor=1):
    if red + blue <= 3:
        return lastround(red, blue) * factor

    # A 得红球，游戏结束
    win = red / (red + blue) * factor
    # A 不得红球，要游戏下去，B不能得红球
    a = blue / (red + blue)
    b = (blue - 1) / (red + blue - 1)
    # C 拿什么球都可以
    # C 红
    cred = red / (red + blue - 2)
    # 开始下一回合，A的获胜率
    win1 = redball(red-1, blue-2, factor=a * b * cred)
    # C 蓝
    cblue = (blue - 2) / (red + blue - 2)
    win2 = redball(red, blue-3, a * b * cblue)
    return win + (win1 + win2) * (1 - win)
    
x = redball(red, blue)
print(x)