# script to download image from baidu
import requests
import re
from urllib.request import urlretrieve
from uuid import uuid1



words = [
    '写真', '美食', '美女写真', '模特', '明星写真', '动漫', '汽车', '飞机', '动物', '鸟', '街拍', '车模',
    '宗教', '艺术', '伊斯兰', '电影', '科技', '海', '光', '梦', '心', '手杖', '五谷',  
]

words = [
    '双胞胎美女', '动漫美女', '传统美女', '现代美女', '非主流', '美男子', '水果', '森林',
    '小鲜肉', '宗教人物', 
]

words = [
    '外国美女写真', '坚果', '外国车模', '韩国美女', '家具', '电影剧照'
]

words = [
    '江纬', '哺乳动物','服装美女', '蔬菜', '外国美女', '聚会', '沙滩', '泳装', 
]

def get_short_uuid():
    s = f"{uuid1()}"
    return s[:8]


def get_html(word):
    url = "https://image.baidu.com/search/index?tn=baiduimage&word={}"
    url = url.format(word)
    resp = requests.get(url)
    resp.encoding = 'UTF-8'
    text = resp.text
    return text


def get_urls(html):
    pattern = re.compile('"objURL":"(.*?.jpg)"')
    urls = pattern.findall(text)
    return urls


def download(urls):
    short_uid = get_short_uuid()
    for i, url in enumerate(urls):
        name = f"image/{i}-{short_uid}.jpg"
        try:
            urlretrieve(url, name)
        except Exception as e:
            print(e)

if __name__ == '__main__':
    for word in words:
        text = get_html(word)
        urls = get_urls(text)
        download(urls)