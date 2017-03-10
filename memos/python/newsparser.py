# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 16:45:41 2017

@author: Administrator
"""
import requests
from bs4 import BeautifulSoup
from webUtil import character2url, date2timestamp

def main(bt, et, must, either):
    url = url_constructor(bt, et, must, either)
    soup = preprocess(url)
    first, other = get_all_text(soup)
    return first, other
    
def preprocess(url, params={}):
    news = requests.get(url, params)
    news.encoding = 'UTF-8'
    soup = BeautifulSoup(news.text, "lxml")
    return soup


def get_all_text(soup):
    first_text = get_text(soup)
    other_text = next_page(soup)
    return first_text, other_text
    
def get_text(soup):
    rs = soup.find_all('h3', {'class': 'c-title'})
    texts = [r.get_text() for r in rs]
    return texts


def get_links(soup):
    tag_a = soup.find("p", id="page")
    links = tag_a.find_all('a')
    links = [link.get('href') for link in links]
    return links


def url_constructor(bt, et, must, either):
    must = '"'+must+'"'
    either = '"'+either+'"'
    urlbase = "http://news.baidu.com/ns?from=news&cl=2"
    urldate = "&bt={bt}&et={et}".format(bt=date2timestamp(bt), 
                   et=date2timestamp(et))
    keywords = "&q1={must}&submit=%B0%D9%B6%C8%D2%BB%CF%C2&q3={either}&q4=".format(must=character2url(must), either=character2url(either))
    options = "&mt=0&lm=&s=2&tn=newstitledy&ct=0&rn=50&q6="
    return urlbase+urldate+keywords+options

def next_page(soup):
    all_text = []
    base = 'http://news.baidu.com'
    for link in get_links(soup):
        url = base + link
        soup = preprocess(url)
        text = get_text(soup)
        all_text.append(text)
    return all_text
    

home = "http://news.baidu.com/advanced_news.html"

url = "http://news.baidu.com/ns?from=news&cl=2&bt=1457280000&y0=2016&m0=3&d0=7&y1=2017&m1=3&d1=7&et=1488902399&q1=%22%BA%CF%D7%F7%22&submit=%B0%D9%B6%C8%D2%BB%CF%C2&q3=%A1%B0%C1%AA%CD%A8%A1%B1&q4=&mt=0&lm=&s=2&begin_date=2016-3-7&end_date=2017-3-7&tn=newstitledy&ct=0&rn=50&q6="

url_format = "http://news.baidu.com/ns?from=news&cl=2&bt=1457280000&y0=2016&m0=3&d0=7&y1=2017&m1=3&d1=7&et=1488902399&q1={must}&submit=%B0%D9%B6%C8%D2%BB%CF%C2&q3={either}&q4=&mt=0&lm=&s=2&begin_date=2016-3-7&end_date=2017-3-7&tn=newstitledy&ct=0&rn=50&q6=".format(must=character2url('"合作"'), either=character2url('"联通"'))
better = "http://news.baidu.com/ns?word=intitle%3A%28%22%E5%90%88%E4%BD%9C%22%20%28%E2%80%9C%E8%81%94%E9%80%9A%E2%80%9D%29%29&pn=100&cl=2&ct=0&tn=newstitledy&rn=50&ie=utf-8&bt=1457280000&et=1488902399"

    
if '__main__' == __name__:
    pass