# DONT_ACCEPT_TRUE_FOR_1
# NORMALIZE_WHITESPACE
# ELLIPSIS

'''
>>> False
0
>>> True
1
>>> False # doctest: +DONT_ACCEPT_TRUE_FOR_1
0
>>> True # doctest: +DONT_ACCEPT_TRUE_FOR_1
1


>>> [list(range(5)) for i in range(5)] # doctest: +NORMALIZE_WHITESPACE
[[0, 1, 2, 3, 4],
 [0, 1, 2, 3, 4],
 [0, 1, 2, 3, 4],
 [0, 1, 2, 3, 4],
 [0, 1, 2, 3, 4]]



>>> class Spam(object):
...      pass
>>> Spam()  # doctest: +ELLIPSIS
<__main__.Spam object at 0x...>
'''

if __name__ == '__main__':
    import doctest

    doctest.testmod()
