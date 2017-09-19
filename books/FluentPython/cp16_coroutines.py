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
