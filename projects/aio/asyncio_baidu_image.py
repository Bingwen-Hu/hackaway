import re
import asyncio
import aiohttp
import time

from baidu_image import get_short_uuid


START_TIME = time.time()
IMAEG_LIST = []


async def fetch(word):
    """ return html page
    word: kind
    """
    def get_urls(html):
        """ parse image from html page
        html: html page from fetch
        """
        pattern = re.compile('"objURL":"(https?:[\/a-zA-Z0-9_\-\.]+.jpg)"')
        urls = pattern.findall(html)
        # for debug
        IMAEG_LIST.append(urls)
        return urls

    print(f'start fetch task {word} at: {time.time() - START_TIME}')
    url = "https://image.baidu.com/search/index?tn=baiduimage&word={}"
    url = url.format(word)
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as resp:
            text = await resp.text()
            return get_urls(text)
            

async def download(url, name):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as resp:
            content = await resp.read()
            with open(name, 'wb') as f:
                f.write(content)


def main():
    words = [
        '写真', '美女写真', '模特', '明星写真', '街拍', '车模', 
    ]

    tasks = [asyncio.ensure_future(fetch(word)) for word in words]
    loop = asyncio.get_event_loop()
    loop.run_until_complete(asyncio.wait(tasks))


if __name__ == '__main__':
    main()