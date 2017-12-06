# namedtuple work as struct in C

import collections

Point = collections.namedtuple('Point', ['x', 'y', 'z'])
point = Point(1, 2, 3)

print(point)