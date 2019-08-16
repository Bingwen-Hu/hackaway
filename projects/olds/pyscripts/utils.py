# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 10:52:52 2017

@author: Administrator
"""

import requests
from bs4 import BeautifulSoup


def preprocess(name):
    """ 预处理，将获取name指定的网页，返回一个BeautifulSoup
    name: 字符串，百度百科姓名
    ret: soup <class BeautifulSoup>
    """
    leader = requests.get("http://baike.baidu.com/item/" + name)
    leader.encoding = 'UTF-8'
    soup = BeautifulSoup(leader.text, "lxml")
    return soup