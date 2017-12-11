import contextlib


# DIY open function
@contextlib.contextmanager
def open_context_manager(filename, mode='r'):
    fh = open(filename, mode)
    yield fh
    fh.close()

# usage:
with open_context_manager('context.py') as fh:
    print(fh.read())

# a simple one
# open is a contextmanager, so `closing` is not need in fact.
with contextlib.closing(open('context.py')) as fh:
    print(fh.read())

# the following example show that we can wrapped a function in a context without `with`
@contextlib.contextmanager
def debug(name):
    print('Debugging %r: ' % name)
    yield
    print('End of debugging %r' % name)

@debug('spam')
def spam():
    print('This is the inside of our spam function')

spam()
