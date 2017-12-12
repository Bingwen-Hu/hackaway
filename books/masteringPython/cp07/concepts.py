# chapter 7 asyncio
# concepts of asyncio

# main concepts: coroutines and event loops
# helper class: Streams, Futures, Processes

# create_task and  ensure_future
import asyncio

async def sleeper(delay):
    await asyncio.sleep(delay)
    print('Finished sleeper with delay: %d' % delay)

# create an event loop
loop = asyncio.get_event_loop()

# create the task
result = loop.call_soon(loop.create_task, sleeper(1))

# make sure the loop stops after 2 seconds
result = loop.call_later(2, loop.stop)

# start the loop and make it run forever, or at least until the loop.stop gets
loop.run_forever()

