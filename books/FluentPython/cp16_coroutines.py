# chapter 16 Coroutines

# basic behavior of a generator used as a Coroutine

def simple_coroutine():
    """
    Usage:
        >>> my_coro = simple_coroutine()
        >>> my_coro
        >>> next(my_coro)
        -> coroutine started
        >>> my_coro.send(42)
        ......
    """
    print('-> coroutine started')
    x = yield
    print('-> coroutine received:', x)


def simple_coro2(a):
    print('-> started: a = ', a)
    b = yield a
    print('-> Received: b = ', b)
    c = yield a + b
    print('-> Received: c = ', c)



def averager():
    total = 0.0
    count = 0
    average = None
    while True:
        term = yield average
        total += term
        count += 1
        average = total / count

coro_avg = averager()
next(coro_avg)
coro_avg.send(10)
coro_avg.send(20)
coro_avg.send(5)


from functools import wraps

def coroutine(func):
    """Decorator: primes 'func' by advancing to first `yield`"""
    @wraps(func)
    def primer(*args, **kwargs):
        gen = func(*args, **kwargs)
        next(gen)
        return gen
    return primer






#==============================================================================
# Error handle
#==============================================================================
class DemoException(Exception):
    """An exception type for the demonstration"""
@coroutine
def demo_exc_handling():
    print('-> coroutine started')
    while True:
        try:
            x = yield
        except DemoException:
            print("*** DemoException handled, Continuing...")
        else:
            print('-> coroutine received: {!r}'.format(x))
    raise RuntimeError("This line should never run.")


# make it safer
@coroutine
def demo_exc_handling2():
    print('-> coroutine started')
    try:
        while True:
            try:
                x = yield
            except DemoException:
                print("*** DemoException handled, Continuing...")
            else:
                print('-> coroutine received: {!r}'.format(x))
    finally:
        print('-> coroutine ending')


#==============================================================================
# Returning a Value from a Coroutine
#==============================================================================
from collections import namedtuple

Result = namedtuple('Result', 'count average')
@coroutine
def averager():
    total = 0.0
    count = 0
    average = None
    while True:
        term = yield
        if term is None:
            break
        total += term
        count += 1
        average = total / count
    return Result(count, average)




#==============================================================================
# yield from rethink
#==============================================================================
def gen():
    for c in "AB":
        yield c
    for i in range(1, 3):
        yield i

def gen2():
    yield from "AB"
    yield from range(1, 3)


# the subgenerator
def averager():
    total = 0.0
    count = 0
    average = None
    while True:
        term = yield
        if term is None:
            break
        total += term
        count += 1
        average = total / count
    return Result(count, average)

# the delegating generator
def grouper(results, key):
    while True:
        results[key] = yield from averager()

# helper function
def report(results):
    for key, result in sorted(results.items()):
        group, unit = key.split(';')
        print('{:2} {:5} averaging {:.2f}{}'.format(
                result.count, group, result.average, unit))


data = {
        'girls;kg':
            [40.9, 38.5, 44.3, 42.2, 45.2, 41.7, 44.5, 38.0, 40.6, 44.5],
        'girls;m':
            [1.6, 1.51, 1.4, 1.3, 1.41, 1.39, 1.33, 1.46, 1.45, 1.43],
        'boys;kg':
            [39.0, 40.8, 43.2, 40.8, 43.1, 38.6, 41.4, 40.6, 36.3],
        'boys;m':
            [1.38, 1.5, 1.32, 1.25, 1.37, 1.48, 1.25, 1.49, 1.46],
            }

# the client code, a.k.a. the caller
def main(data):
    results = {}
    for key, values in data.items():
        group = grouper(results, key)
        next(group)
        for value in values:
            group.send(value)
        group.send(None)
    report(results)
main(data)