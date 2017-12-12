# Warning: test on linux is ok, on windows is not ok......

# there two bundled event loop implementations
# async.SelectorEventLoop & async.ProactorEventLoop (only on windows)

# example
# setup event loops
# loop = asyncio.ProactorEventLoop()
# asyncio.set_event_loop(loop)

import sys
import selectors

def read(fh):
    print('Got input from stdin: %r' % fh.readline())

if __name__ == '__main__':
    # Create the default selector
    selector = selectors.DefaultSelector()

    # Register the read function for the READ event on stdin
    selector.register(sys.stdin, selectors.EVENT_READ, read)

    while True:
        for key, mask in selector.select():
            # the data attribute contains the read function here
            callback = key.data
            # call it with the file object (stdin here)
            callback(key.fileobj)
