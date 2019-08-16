# -*- coding: utf-8 -*-
""" 属于百科上的什么条目呢？"""

import re

from utils import preprocess


def what_kind(soup):
    try:
        tags = soup.find("dd", {"id": "open-tag-item"})
        class_str = tags.get_text()
    except:
        return "找不到哦！自己去百度百科看一下？"
#    h2s = soup.find_all("h2", {"class": "title-text"})
#    h2s = [h2.get_text() for h2 in h2s]
    else:
        class_str = re.split(r"[\n， ]", class_str)
        class_str = [c for c in class_str if c]
    return ' '.join(class_str)


if __name__ == '__main__':
    import sys
    item = sys.argv[1]
    soup = preprocess(item)
    kinds = what_kind(soup)
    print(kinds)