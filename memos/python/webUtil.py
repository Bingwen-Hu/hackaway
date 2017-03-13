# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 11:35:12 2017

@author: Mory
@description: python tool for web parsing 
"""
import time
import urllib.parse as parse

def date2timestamp(date=[2017, 3, 7]):
    year, month, day = date
    a = "%s-%s-%s 00:00:00" % (year, month, day)
    timeArray = time.strptime(a, "%Y-%m-%d %H:%M:%S")
    timeStamp = int(time.mktime(timeArray))
    return timeStamp

def timestamp2str(timestamp=1488902399):
    import datetime
    dateArray = datetime.datetime.utcfromtimestamp(timestamp)
    otherStyleTime = dateArray.strftime("%Y-%m-%d %H:%M:%S")
    return otherStyleTime

def character2url(string, encode='gbk'):
    return parse.quote(string, encoding=encode)
def url2character(string, encode='gbk'):    
    return parse.unquote(string, encoding=encode)
