import re
import asyncio
import aiohttp
import time


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
    global IMAEG_LIST
    words = [
        '奥巴马', '特朗普', '习近平', '马云'
    ]

    tasks = [asyncio.ensure_future(fetch(word)) for word in words]
    loop = asyncio.get_event_loop()
    loop.run_until_complete(asyncio.wait(tasks))
    
    # download
    IMAEG_LIST = [url for lst in IMAEG_LIST for url in lst]
    tasks = [asyncio.ensure_future(download(url, "%s.jpg" % i)) for i, url in enumerate(IMAEG_LIST)]
    loop.run_until_complete(asyncio.wait(tasks))


if __name__ == '__main__':
    main()