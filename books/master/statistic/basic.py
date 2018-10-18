# medium mode and mean
def median(lst):
    """ calculate the median in a list of data
    Args:
        lst (list): list of numeric data
    Returns:
        median (int)
    """
    lst.sort()
    length = len(lst)
    if length % 2 == 0:
        median = 0.5 * (lst[length//2-1] + lst[length//2])
    else:
        median = lst[length//2-1]
    return median

def mean(lst):
    sum = 0
    count = 0
    for i in lst:
        sum += i
        count += 1
    return sum / count


from collections import Counter
def mode(lst):    
    c = Counter(lst)
    most, count = c.most_common(1)[0]
    modes = [k for k in c if c[k] == count]
    return modes


def range_(lst):
    return max(lst) - min(lst)

def standard_deviation(lst):
    pass

import unittest
class BasicStatistic(unittest.TestCase):

    def test_median(self):
        lst = [2, 3, 2]
        self.assertEqual(2, median(lst))
        lst = [3, 5, 1, 4]
        self.assertEqual(3.5, median(lst))

    def test_mean(self):
        lst = [1, 4, 6, 8, 0, 12, 8, 9, 3]
        self.assertAlmostEqual(5.6, mean(lst), delta=0.1)

    def test_mode(self):
        lst = [1, 1, 2, 2, 2, 3, 3]
        self.assertEqual([2], mode(lst))
        lst = [1, 1, 2, 2, 3, 3]
        self.assertSetEqual(set([1, 2, 3]), set(mode(lst)))

    def test_range(self):
        lst = [1, 10, 2, 22, 21, 13, 3]
        self.assertEqual(22-1, range_(lst))
        
    