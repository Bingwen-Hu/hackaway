# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 16:45:41 2017

@author: Mory
"""
import requests
import pandas as pd
from bs4 import BeautifulSoup
from webUtil import character2url, date2timestamp

defaulttimes = [[2010, 1, 1],
             [2011, 1, 1],
             [2012, 1, 1],
             [2013, 1, 1],
             [2014, 1, 1],
             [2015, 1, 1],
             [2016, 1, 1],
             [2017, 1, 1]]


def df_split(df):
    company = df['公司'][0]
    indexfilter = df['新闻'].apply(lambda news: company in news)
    df = df[indexfilter]
    indexStrategy = df['新闻'].apply(lambda news: '战略' in news)
    strategy = df[indexStrategy] 
    nonStrategy = df[~indexStrategy]
    return strategy, nonStrategy

def all_df(times, keyword, company):
    dataframes = pd.DataFrame(columns=['公司', '起始时间', '新闻'])
    for bt, et in zip(times, times[1:]):
        df = get_df(bt, et, keyword, company)
        dataframes = dataframes.append(df, ignore_index=True)
    return dataframes
def cleaned_df(first, other):
    all_others = [o for ox in other for o in ox]
    first.extend(all_others)
    return list(set(first))

def get_df(bt, et, keyword, company):
    url = url_constructor(bt, et, keyword, company)
    soup = preprocess(url)
    first, other = get_all_text(soup)
    text = cleaned_df(first, other)
    df = news_output(text, bt, company)
    return df
    
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

def news_output(text, bt, company):
    df = pd.DataFrame(columns=['公司', '起始时间', '新闻'])
    df["新闻"] = text
    df["公司"] = company
    df["起始时间"] = str(bt[0])
    return df


if __name__ == '__main__':
    importants = ['小米', '华为', 
    '苹果', '微软', '谷歌', 
    '支付宝', '京东', '亚马逊', # 购物
    '美团', '百度外卖', # 查看市场份额
    '滴滴', '摩拜单车']
    
    
    home = "http://news.baidu.com/advanced_news.html"
