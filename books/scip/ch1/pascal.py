# recursive
def pascal_triangle(n):
    assert n >= 1
    if n == 1:
        return 1
    elif n == 2:
        return 1, 1
    else:
        x = pascal_helper(pascal_triangle(n-1))
        x.insert(0, 1)
        x.append(1)
        return x

def pascal_helper(lst):
    def helper():
        for pair in zip(lst, lst[1:]):
            yield sum(pair)
    return list(helper())
