# generator and context manager

import datetime
import contextlib

@contextlib.contextmanager
def timer(name):
    start_time = datetime.datetime.now()
    yield
    stop_time = datetime.datetime.now()
    print('%s took %s' % (name, stop_time - start_time))

@contextlib.contextmanager
def write_to_log(name):
    """this function writes all stdout to a file, The contextlib.redirect_stdout
        context wrapper temporarily redirects standard output to a given file
        handle, in this case the file we just opened for writing.
    """
    with open('%s.txt' % name, 'w') as fh:
        with contextlib.redirect_stdout(fh):
            with timer(name):
                yield

import time
# use the context manager as a decorator
@write_to_log('some function')
def some_function():
    time.sleep(2)
    print('This function takes a bit of time to execute...')
    print('This function takes a bit of time to execute...')
    print('Do more...')
some_function()


# another version using decorator, reduce the with stack
@contextlib.contextmanager
def write_to_log2(name):
    with contextlib.ExitStack() as stack:
        fh = stack.enter_context(open('stdout.txt', 'w'))
        stack.enter_context(contextlib.redirect_stdout(fh))
        stack.enter_context(timer(name))
        yield

@write_to_log2('some function2')
def some_function2(): # duplicate
    time.sleep(2)
    print('This function takes a bit of time to execute...')
    print('This function takes a bit of time to execute...')
    print('Do more...')
some_function2()



# a another example:
import contextlib
with contextlib.ExitStack() as stack:
    spam_fh = stack.enter_context(open('spam.txt', 'w'))
    eggs_fh = stack.enter_context(open('eggs.txt', 'w'))
    spam_bytes_written = spam_fh.write('writing to spam')
    eggs_bytes_written = eggs_fh.write('writing to eggs')
    # move the contexts to a new ExitStack and store the close method
    close_handlers = stack.pop_all().close

spam_bytes_written = spam_fh.write('still write to spam')
eggs_bytes_written = eggs_fh.write('still write to eggs')

close_handlers()
# could not write anymore