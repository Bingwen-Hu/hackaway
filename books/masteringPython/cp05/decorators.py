# It is time to remember decorator syntax

import functools

def eggs(function):
    @functools.wraps(function)
    def _eggs(*args, **kwargs):  # for _eggs is the function to return, so accept every kind of params
        print('%r got args: %r and kwargs: %r' % (
            function.__name__, args, kwargs
        ))
        return function(*args, **kwargs)

    return _eggs

@eggs
def spam(a, b, c):
    """The spam function Returns a * b + c"""
    return a * b + c


# functools.wraps wrap all the function properties including documents
# so whenever writing a decorator, always be sure to add functools.wrap
# the inner function.

# example of using decorators: Memoization

import functools
def memoize(function):
    function.cache = dict()

    @functools.wraps(function)
    def _memoize(*args):
        if args not in function.cache:
            function.cache[args] = function(*args)
        return function.cache[args]
    return _memoize

@memoize
def fibonacci(n):
    if n < 2:
        return n
    else:
        return fibonacci(n-1) + fibonacci(n-2)

for i in range(1, 7):
    print('fiboacci %d: %d' % (i, fibonacci(i)))



# stack decorators
def counter(function):
    function.calls = 0
    @functools.wraps(function)
    def _counter(*args, **kwargs):
        function.calls += 1
        return function(*args, **kwargs)
    return _counter

@functools.lru_cache(maxsize=3)
@counter
def fibonacci(n):
    if n < 2:
        return n
    else:
        return fibonacci(n-1) + fibonacci(n-2)

fibonacci(100)




# decorator with (optional) arguments
# Mory Note: generally, decorator accept a function as parameter and define a inner function accept
# the same parameters as the decoratored function. So a extra layer to capture extra parameter is
# needed.
import functools

def add(extra_n=1):
    "Add extra_n to the input of the decoratord function"
    # the inner function, notice that this is the actual decorator
    def _add(function):
        # the actual function that will be called
        # where functools.wraps will exists
        @functools.wraps(function)
        def _add(n):
            return function(n + extra_n)
        return _add
    return _add

@add(extra_n=2)
def eggs(n):
    return 'eggs' * n

# Yes, optional arguments is more difficult, so let's have a look
def add(*args, **kwargs):
    "add n to the input of the decorated function"

    # the default kwargs, we don't store this in kwargs
    # because we want to make sure that args and kwargs
    # can't both be filled
    default_kwargs = dict(n=1)

    # the inner function, notice that this is actually a
    # decorator itself
    def _add(function):
        # the actual function that will be called
        @functools.wraps(function)
        def __add(n):
            default_kwargs.update(kwargs)
            return function(n + default_kwargs['n'])

        return __add

    if len(args) == 1 and callable(args[0]) and not kwargs:
        # Decorator call without argument, just call it ourselves
        return _add(args[0])
    elif not args and kwargs:
        # Decorator call with arguments, this time it will
        # actomatically be executed with function as the
        # first argument
        default_kwargs.update(kwargs)
        return _add
    else:
        raise RuntimeError("This decorator only supports "
                           'keyword argument')


# decorators using class
import functools

class Debug(object):

    def __init__(self, function):
        self.function = function
        # functools.wraps for classes
        functools.update_wrapper(self, function)

    def __call__(self, *args, **kwargs):
        output = self.function(*args, **kwargs)
        print('%s(%r, %r): %r' % (self.function.__name__, args, kwargs, output ))
        return output


@Debug
def spam(eggs):
    return 'spam' * (eggs % 5)