# usage
import time
import asyncio

t = time.time()

def printer(name):
    print('Started %s at %.1f' % (name, time.time() -t))
    time.sleep(0.2)
    print('Finished %s at %.1f' % (name, time.time() -t))

loop = asyncio.get_event_loop()
result = loop.call_at(loop.time() + .2, printer, 'call_at')
result = loop.call_later(0.1, printer, 'call_later')
result = loop.call_soon(printer, 'call_soon')
result = loop.call_soon_threadsafe(printer, 'call_soon_threadsafe')

result = loop.call_later(1, loop.stop)

loop.run_forever()