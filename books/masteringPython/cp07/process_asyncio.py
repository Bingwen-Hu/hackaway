# only on linux, maybe a bug

import time
import asyncio

t = time.time()


async def async_process_sleeper():
    print('Started sleep at %.1f' % (time.time() - t))
    process = await asyncio.create_subprocess_exec('sleep', '0.1')
    await process.wait()
    print('Finished sleep at %.1f' % (time.time() - t))

loop = asyncio.get_event_loop()
for i in range(1):
    task = loop.create_task(async_process_sleeper())

future = loop.call_later(1, loop.stop)
loop.run_forever()