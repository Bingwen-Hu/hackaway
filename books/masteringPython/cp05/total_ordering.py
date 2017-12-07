import functools

class Value(object):

    def __init__(self, value):
        self.value = value

    def __repr__(self):
        return '<%s[%d]>' % (self.__class__, self.value)

class Spam(Value):
    def __gt__(self, other):
        return self.value > other.value

    def __ge__(self, other):
        return self.value >= other.value

    def __lt__(self, other):
        return self.value < other.value

    def __le__(self, other):
        return self.value <= other.value

    def __eq__(self, other):
        return self.value == self.value

@functools.total_ordering
class Egg(Value):
    def __lt__(self, other):
        return self.value < other.value

    def __eq__(self, other):
        return self.value == other.value


if __name__ == '__main__':
    numbers = [4, 2, 3, 1, 4]
    spams = [Spam(n) for n in numbers]
    eggs = [Egg(n) for n in numbers]
    import pprint
    print(pprint.pformat(spams))
    print(pprint.pformat(eggs))