# leverage interpreter
import asyncio


async def run_script():
    process = await asyncio.create_subprocess_shell(
        'python3',
        stdout=asyncio.subprocess.PIPE,
        stdin=asyncio.subprocess.PIPE,
    )

    # write a simple python script to the interpreter

    process.stdin.write(b'\n'.join((
        b'import math',
        b'x = 2 ** 8',
        b'print(math.sqrt(x))',
    )))

    # make sure the stdin is flushed asynchronously
    await process.stdin.drain()

    # send end of file to interpreter
    process.stdin.write_eof()

    async for out in process.stdout:
        print(out.decode('utf-8').rstrip())

    await process.wait()


if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    loop.run_until_complete(run_script())
    loop.close()