# Chapter 9

from array import array
import math

class Vector2d:
    typecode = 'd'

    def __init__(self, x, y):
        self.__x = float(x)
        self.__y = float(y)

    @property
    def x(self):
        return self.__x

    @property
    def y(self):
        return self.__y

    # 支持迭代和解构，因此支持__repr__里的*self语法
    def __iter__(self):
        return (i for i in (self.x, self.y))


    def __repr__(self):
        class_name = type(self).__name__

        # !r 在这里表示原生打印，即元组仍是元组，字符串仍是字符串
        return '{}({!r}, {!r})'.format(class_name, *self)

    def __str__(self):
        return str(tuple(self))

    def __bytes__(self):
        return (bytes([ord(self.typecode)]) +
                bytes(array(self.typecode, self)))

    def __eq__(self, other):
        return tuple(self) == tuple(other)

    def __abs__(self):
        return math.hypot(self.x, self.y)

    def __bool__(self):
        return bool(abs(self))

    def __format__(self, fmt_spec=""):
        if fmt_spec.endswith('p'):
            fmt_spec = fmt_spec[:-1]
            coords = (abs(self), self.angle())
            outer_fmt = '<{}, {}>'
        else:
            coords = self
            outer_fmt = '({}, {})'
        components = (format(c, fmt_spec) for c in coords)
        return outer_fmt.format(*components)

    # No self argument; instead, the class itself is passed as cls.
    @classmethod
    def frombytes(cls, octets):
        typecode = chr(octets[0])
        memv = memoryview(octets[1:]).cast(typecode)
        return cls(*memv)

    def angle(self):
        return math.atan2(self.y, self.x)

    # Note on classmethod: alternative constructor

#==============================================================================
# staticmethod is a plain function happens to live in a class
#==============================================================================
class Demo:
    @classmethod
    def klassmeth(*args):
        return args
    @staticmethod
    def statmeth(*args):
        return args

print(Demo.klassmeth())
print(Demo.klassmeth('spam'))
print(Demo.statmeth())
# note: No matter how you invoke it, Demo.klassmeth receives the Demo class as the first
# argument.

# Now that we’ve seen what classmethod is good for (and that staticmethod is not very
# useful), let’s go back to the issue of object representation and see how to support for‐
# matted output

#==============================================================================
# format
#==============================================================================
brl = 1 / 2.43
"1 BRL = {rate: 0.2f} USD".format(rate=brl)
print(format(42, 'b'))
print(format(2/3, '.1%'))

from datetime import datetime
now = datetime.now()
print(format(now, "%H:%M:%S"))
print("It's now {:%I:%M %p}".format(now))
