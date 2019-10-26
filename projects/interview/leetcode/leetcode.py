# https://leetcode-cn.com/problems/container-with-most-water/
def maxArea(heights) -> int:
    """盛水最多的容器"""
    left = 0
    right = len(heights) - 1

    if heights[left] < heights[right]:
        MOVELEFT = True
        h = heights[left]
    else:
        MOVELEFT = False
        h = heights[right]
    
    area = (right - left) * h
    nleft = left + 1
    nright = right - 1
    
    while left < right:
        print(f"Left {left}  Right {right} Moveleft {MOVELEFT}")
        print(f"Area {area}  nLeft {nleft} nRight {nright}")
        if MOVELEFT:
            while heights[nleft] <= heights[left] and nleft < right:
                nleft += 1
            if nleft == right:
                return area

            if heights[nleft] < heights[right]:
                h = heights[nleft]
            else:
                h = heights[right]
            
            narea = (right - nleft) * h
            if narea > area:
                left = nleft
                area = narea
                if heights[left] < heights[right]:
                    MOVELEFT = True
                else:
                    MOVELEFT = False
            else:
                nleft -= 1
        else:
            while heights[nright] <= heights[right] and left < nright:
                nright -= 1
            if left == nright:
                return area

            if heights[left] < heights[nright]:
                h = heights[left]
            else:
                h = heights[nright]
            narea = (nright - left) * h
            if narea > area:
                right = nright
                area = narea
                if heights[left] < heights[right]:
                    MOVELEFT = True
                else:
                    MOVELEFT = False
            else:
                nright -= 1
    return area

def test_maxArea():
    val = maxArea([1,8,6,2,5,4,8,3,7])
    expect = 49
    assert val == expect


if __name__ == '__main__':
    test_maxArea()

