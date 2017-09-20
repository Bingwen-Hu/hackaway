# Chapter 17 concurrency with futures
import os
import sys
import requests
from concurrent import futures

MAX_WORKERS = 20

BASE_URL = "http://flupy.org/data/flags"

DEST_DIR = "downloads/"



def save_flag(img, filename):
    path = os.path.join(DEST_DIR, filename)
    with open(path, 'wb') as fp:
        fp.write(img)


def get_flag(cc):
    url = '{}/{cc}/{cc}.gif'.format(BASE_URL, cc=cc.lower())
    resp = requests.get(url)
    return resp.content

def show(text):
    print(text, end=" ")
    sys.stdout.flush()


def download_one(cc):
    image = get_flag(cc)
    show(cc)
    save_flag(image, cc.lower() + '.gif')
    return cc

def download_many(cc_list):
    workers = min(MAX_WORKERS, len(cc_list))
    with futures.ThreadPoolExecutor(workers) as executor:
        res = executor.map(download_one, sorted(cc_list))
    return len(list(res))




def download_many2(cc_list):
    cc_list = cc_list[:5]
    with futures.ThreadPoolExecutor(max_workers=3) as executor:
        to_do = []
        for cc in sorted(cc_list):
            future = executor.submit(download_one, cc)
            to_do.append(future)
            msg = "Scheduledd for {}: {}"
            print(msg.format(cc, future))

        results = []
        for future in futures.as_completed(to_do):
            res = future.result()
            msg = '{} result: {!r}'
            print(msg.format(future, res))
            results.append(res)

    return len(results)



#==============================================================================
# experimenting with Executor.map
#==============================================================================
from time import sleep, strftime
# from concurrent import futures


def display(*args):
    print(strftime('[%H:%M:%S]'), end=" ")
    print(*args)

def loiter(n):
    msg = '{}loiter({}): doing nothing for {}s...'
    display(msg.format('\t'*n, n, n))
    sleep(n)
    msg = '{}loiter({}): done.'
    display(msg.format('\t'*n, n))
    return n * 10

def main():
    display("Script starting.")
    executor = futures.ThreadPoolExecutor(max_workers=3)
    results = executor.map(loiter, range(10))
    display('results:', results)
    display("waiting for individual results:")
    for i, result in enumerate(results):
        display('result {}: {}'.format(i, result))

main()