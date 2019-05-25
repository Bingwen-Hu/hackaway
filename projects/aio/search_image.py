import re
import asyncio
import aiohttp
import time
from uuid import uuid1

START_TIME = time.time()
IMAEG_LIST = []

def get_urls(html):
    """ parse image from html page
    html: html page from fetch
    """
    pattern = re.compile('(https?:[\/a-zA-Z0-9_\-\.]+.j?pn?e?g)')
    urls = pattern.findall(html)
    # for debug
    IMAEG_LIST.append(urls)
    return urls


async def bing_search(word):
    print(f'start fetch task {word} at: {time.time() - START_TIME}')
    url = "https://cn.bing.com/images/search?q={}&FORM=BESBTB"
    url = url.format(word)
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as resp:
            text = await resp.text()
            return get_urls(text)

async def baidu_search(word):
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


def main(engine='baidu'):
    global IMAEG_LIST
    words = [
        '人类笑容', "青年笑容", '小孩笑容', "美女笑容", "医生笑容", "学生笑容", "护士笑容", 
        '程序员的笑容', '马云的笑容', '奥巴马笑容', '习近平笑容', '车模笑容', '清洁工的笑容',
        '妻子的笑容', '丈夫的笑容', '杀手的笑容', '黑客的笑容', '建筑师的笑容', '法师的笑容',
        '藏民的笑容', '回族的笑容', '基督教的笑容', '白种人的笑容', '蓝眼睛的人的笑容', '黄种人的笑容',
        '黑人的笑容', '空姐的笑容', '行政小哥的笑容', '人事小妹的笑容', '叔叔的笑容', '警察的笑容',
        '军人的笑容', '导演的笑容', '大学生的笑容', '暨南大学生的笑容', '农村孩子的笑容', 
        '贫困山区的人的笑容', '放牛娃的笑容', '古装女子的笑容'
    ]
    words = [
        'smile human','smile woman', 'smile man', 'smile boy', 'smile girl', 'smile student', 
        'smile teacher', 'smile wife', 'smile couple', 'very happy woman'     
    ]
    words = [
        '惊讶的妇女', '惊讶的女人', '惊讶的女生', '惊讶的小女孩', '惊讶的小男孩', '惊讶的男人', '惊讶的老人', 
        '惊讶的美国人', '惊讶的老百姓', '惊讶的大学生',  '惊讶的小孩子'
    ]
    words = [
        'surprising+woman', 'surprising+man', 'surprising+boy', 'surprising+girl', 'surprising+old+man'
    ]

    words = [
        '伤心的妇女', '伤心的女人', '伤心的女生', '伤心的小女孩', '伤心的小男孩', '伤心的男人', '伤心的老人', 
        '伤心的美国人', '伤心的老百姓', '伤心的大学生',  '伤心的小孩子'
    ]

    words = [
        'sad+woman', 'sad+man', 'sad+girl', 'sad+boy', 'sad+old+man', 'sad+woman+widow'
    ]

    words = [
        '生气的女人', '生气的女生', '生气的小女孩', '生气的小男孩', '生气的男人', '生气的老人', 
        '生气的美国人', '生气的老百姓', '生气的大学生',  '生气的小孩子'
    ]

    words = [
        'anger+woman', 'anger+man', 'anger+girl', 'anger+boy', 'anger+old+man', 'anger+woman+widow',
    ]

    words = [
        '恐惧的女人', '恐惧的女生', '恐惧的小女孩', '恐惧的小男孩', '恐惧的男人', '恐惧的老人', 
        '恐惧的美国人', '恐惧的老百姓', '恐惧的大学生',  '恐惧的小孩子'
    ]

    words = [
        'fearwoman', 'fear+man', 'fear+girl', 'fear+boy', 'fear+old+man',
    ]

    words = [
        '厌恶的表情 孩子', '厌恶的表情 女生', '厌恶的表情 小女孩', '厌恶的表情 小男孩', '厌恶的表情 男人', '厌恶的表情 老人', 
        '厌恶的表情 美国人', '厌恶的表情 老百姓', '厌恶的表情 大学生',  '厌恶的表情 小孩子',
        'disgust+woman', 'disgust+man', 'disgust+girl', 'disgust+boy', 'disgust+old+man',
    ]
    words = [
        '正常脸 男人', '正常脸 女人', '正常脸 女孩', '正常脸 男孩', '正常脸 老人', 'neutral+woman+expression',
        'neutral+man+expression', 'neutral+girl+expression', 'neutral+boy+expression', 'neutral+old+expression+chinese', 
        'neutral+woman+expression+chinese', 'neutral+man+expression+chinese',
    ]
    
    if engine == 'baidu':
        fetch_fn = baidu_search
    elif engine == 'bing':
        fetch_fn = bing_search
    else:
        raise KeyError("Select an engine in `baidu`, 'bing'")

    tasks = [asyncio.ensure_future(fetch_fn(word)) for word in words]
    loop = asyncio.get_event_loop()
    loop.run_until_complete(asyncio.wait(tasks))
    
    # download
    IMAEG_LIST = [url for lst in IMAEG_LIST for url in lst]
    tasks = [asyncio.ensure_future(download(url, f"{i}-{uuid1()}.jpg")) for i, url in enumerate(IMAEG_LIST)]
    loop.run_until_complete(asyncio.wait(tasks))


if __name__ == '__main__':
    main('baidu')