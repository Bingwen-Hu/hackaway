# coroutines
# The difference between regular generators and coroutines is that coroutines don't
# simply return values to the calling function but can receive values as well.

# a basic example
def generator():
    value = yield 'spam'
    print('Generator received: %s' % value)
    yield 'Previous value: %r' % value

g = generator()
print('Result from generator: %s' % next(g))
print(g.send('eggs'))



# priming
# Since generators are lazy, you can't just send a value to a brand new generator.
# Before a value can be sent to the generator, either a result must be fetched using
# next() or a send(None) has to be issued so that the code is actually reached.
import functools

def coroutine(function):
    @functools.wraps(functools)
    def _coroutine(*args, **kwargs):
        active_coroutine = function(*args, **kwargs)
        next(active_coroutine)
        return active_coroutine
    return _coroutine

@coroutine
def spam():
    while True:
        print('waiting for yield...')
        value = yield
        print('spam received: %s' % value)

generator = spam()
generator.send('a')
generator.send('b')


# closing and throwing exception
@coroutine
def simple_coroutine():
    print('Setting up the coroutine')
    try:
        while True:
            item = yield
            print('Got item: %r' % item)
    except GeneratorExit:
        print('Normal exit')
    except Exception as e:
        print('Exception exit: %r' % e)
        raise
    finally:
        print('Any exit')

print('Creating simple coroutine')
active_coroutine = simple_coroutine()
print()

print('Sending spam')
active_coroutine.send('spam')
print()

print('Close the coroutine')
active_coroutine.close()
print()

print('Creating simple coroutine')
active_coroutine = simple_coroutine()
print()

print('Sending eggs')
active_coroutine.send('eggs')
print()

# uncomment the following three line and comment the fourth if you like.
# print('Throwing runtime error')
# active_coroutine.throw(RuntimeError, 'Oops...')
# print()
active_coroutine.close()


# bidirectional pipelines
@coroutine
def replace(search, replace):
    while True:
        item = yield
        print(item.replace(search, replace))

spam_replace = replace('spam', 'bacon')
for line in open('lines.txt'):
    spam_replace.send(line.rstrip())

# Grep sends all matching items to the target
@coroutine
def grep(target, pattern):
    """
    :param target: target generator
    :param pattern: the pattern to search
    :return:
    """
    while True:
        item = yield
        if pattern in item:
            target.send(item)


# replace does a search and replace on the items and sends it to
# the target once it's done
@coroutine
def replace(target, search, replace):
    while True:
        target.send((yield).replace(search, replace))

# Print will print the items using the provided formatstring
@coroutine
def print_(formatstring):
    while True:
        print(formatstring % (yield))

# Tee multiplexes the items to multiple targets
@coroutine
def tee(*targets):
    while True:
        item = yield
        for target in targets:
            target.send(item)


printer = print_('%s')
replacer_spam = replace(printer, 'spam', 'bacon')
replacer_eggs = replace(printer, 'spam spam', 'sausage')
branch = tee(replacer_spam, replacer_eggs)
greper = grep(branch, 'spam')
for line in open('lines.txt'):
    greper.send(line.rstrip())