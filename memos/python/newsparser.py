# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 16:45:41 2017

@author: Administrator
"""
import time
import requests
from bs4 import BeautifulSoup

url = "http://news.baidu.com/ns?from=news&cl=2&bt=1457280000&y0=2016&m0=3&d0=7&y1=2017&m1=3&d1=7&et=1488902399&q1=%22%BA%CF%D7%F7%22&submit=%B0%D9%B6%C8%D2%BB%CF%C2&q3=%A1%B0%C1%AA%CD%A8%A1%B1&q4=&mt=0&lm=&s=2&begin_date=2016-3-7&end_date=2017-3-7&tn=newstitledy&ct=0&rn=50&q6="
better = "http://news.baidu.com/ns?word=intitle%3A%28%22%E5%90%88%E4%BD%9C%22%20%28%E2%80%9C%E8%81%94%E9%80%9A%E2%80%9D%29%29&pn=100&cl=2&ct=0&tn=newstitledy&rn=50&ie=utf-8&bt=1457280000&et=1488902399"
def main():
    base = "http://news.baidu.com/ns"
    params = {"from":'news',
              "cl":2,
              "bt":1457280000,
              "y0":2016,
              "m0":3,
              "d0":7,
              "y1":2017,
              "m1":3,
              "d1":7,
              "et":1488902399,
              "q1":"合作",
              "submit":"百度一下",
              "q3":"华为",
              "q4":"",
              "mt":0,
              "lm":"",
              "s":2,
              "begin_date":'2016-3-7',
              "end_date":'2017-3-7',
              "tn":'newstitledy',
              "ct":0,
              "rn":50,
              "q6":"",
              }
    soup = preprocess(base, params)
    texts = get_text(soup)
    return texts
    
def preprocess(url, params):
    news = requests.get(url, params)
    news.encoding = 'UTF-8'
    soup = BeautifulSoup(news.text, "lxml")
    return soup



def get_text(soup):
    rs = soup.find_all('h3', {'class': 'c-title'})
    texts = [r.get_text() for r in rs]
    return texts


def links(soup):
    tag_a = soup.find("p", id="page")
    links = tag_a.find_all('a')
    links = [link.get('href') for link in links]
    return links

def str2timestamp(date=[2017, 3, 7]):
    year, month, day = date
    a = "%s-%s-%s 00:00:00" % (year, month, day)
    timeArray = time.strptime(a, "%Y-%m-%d %H:%M:%S")
    timeStamp = int(time.mktime(timeArray))
    return timeStamp

def timestamp2str():
    import datetime
    timeStamp = 1488902399
    dateArray = datetime.datetime.utcfromtimestamp(timeStamp)
    otherStyleTime = dateArray.strftime("%Y-%m-%d %H:%M:%S")
    return otherStyleTime

if '__main__' == __name__:
    base = "http://news.baidu.com/ns"
    params = {"from":'news',
              "cl":2,
              "bt":"",
              "y0":2016,
              "m0":3,
              "d0":7,
              "y1":2017,
              "m1":3,
              "d1":7,
              "et":"",
              "q1":"合作",
              "submit":"百度一下",
              "q3":"华为",
              "q4":"",
              "mt":0,
              "lm":"",
              "s":2,
              "begin_date":'2016-3-7',
              "end_date":'2017-3-7',
              "tn":'newstitledy',
              "ct":0,
              "rn":50,
              "q6":"",
              }