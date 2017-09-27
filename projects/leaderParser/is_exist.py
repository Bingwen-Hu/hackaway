# -*- coding: utf-8 -*-
""" 百度百科上都有谁？ """

from utils import preprocess

def judge(soup):
    no_result = soup.find_all('p', {"class": "sorryCont"})
    if no_result:
        print("Sorry, 百度百科上没有哦！要不你自己写一个？")
        return
    links = soup.find_all("ul", {"class": "polysemantList-wrapper"})
    if not links:
        print("百度百科有一个条目!")
    else:
        print("有多个相同的条目哦!")


if __name__ == '__main__':
    import sys
    item = sys.argv[1]
    soup = preprocess(item)
    judge(soup)