# basic trick, using yield
def count(start=0, step=1, stop=10):
    n = start
    while n <= stop:
        yield n
        n += step

for x in count(10, 2.5, 20):
    print(x)

class Count(object):
    def __init__(self, start=0, step=1, stop=10):
        self.n = start
        self.step = step
        self.stop = stop

    def __iter__(self):
        return self

    def __next__(self):
        n = self.n
        if n > self.stop:
            raise StopIteration()

        self.n += self.step
        return n

for x in Count(10, 2.5, 20):
    print(x)

# generator comprehensions
generator = (x ** 2 for x in range(4))
for x in generator:
    print(x)


# generator usage example
# emulate the shell command
# cat lines.txt | grep spam | sed 's/spam/bacon/g'
def cat(filename):
    for line in open(filename):
        yield line.rstrip()

def grep(sequence, search):
    for line in sequence:
        if search in line:
            yield line

def replace(sequence, search, replace):
    for line in sequence:
        yield line.replace(search, replace)

for line in replace(grep(cat('lines.txt'), 'spam'), 'spam', 'bacon'):
    print(line)


# generating from generators
import itertools

def powerest(sequence):
    for size in range(len(sequence) + 1):
        for item in itertools.combinations(sequence, size):
            yield item

# a shorten version using `yield from`
def powerest_shorten(sequence):
    for size in range(len(sequence) + 1):
        yield from itertools.combinations(sequence, size)

for result in powerest_shorten('abc'):
    print(result)


# yet another example
def flatten(sequence):
    for item in sequence:
        try:
            yield from flatten(item)
        except TypeError: # detect non-iterable objects
            yield item

print(list(flatten([1, 3, [4, 5, [0, 9, 9, [5, 6, 7], 8], 10], 12, -1])))