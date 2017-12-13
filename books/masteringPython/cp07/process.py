# the version without asyncio
import time
import subprocess

t = time.time()

# this function run sequentially
def process_sleeper():
    print('Started sleep at %.1f' % (time.time() - t))
    return subprocess.Popen(['sleep', '0.1'])

processes = []
for i in range(5):
    processes.append(process_sleeper())

for process in processes:
    returncode = process.wait()
    print('Finished sleep at %.1f' % (time.time() - t))
