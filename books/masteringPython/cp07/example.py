# chapter 7
# a simple example of single-threaded parallel processing

import asyncio

async def sleeper(delay):
    await asyncio.sleep(delay)
    print('Finished sleeper with delay: %d' % delay)


loop = asyncio.get_event_loop()
results = loop.run_until_complete(asyncio.wait(
    [sleeper(1), sleeper(3), sleeper(2),]
))
print('Where I am?')
# even we start with the order of 1, 3, 2, the results is in order


# some note:
# asyncio.coroutine: decorator, not very useful now.
# asyncio.sleep: asynchronous version of time.sleep
# asyncio.get_event_loop: default event loop
# asyncio.wait: coroutine for wrapping a seq of coroutines or futures
#               and waiting for the results.