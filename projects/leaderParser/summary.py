# -*- coding: utf-8 -*-
""" 获取百度百科人物简介 """

import re
from utils import preprocess

def get_summary(soup):
    """ 百度百科名目 """
    summary = soup.find("div", {"class": 'lemma-summary'})
    return summary.get_text().strip()

def clean_summary(summary):
    pattern = re.compile('\[\d\]|\xa0|\\n')
    summary_ = pattern.sub("", summary)
    return summary_

if __name__ == '__main__':
    import sys
    item = sys.argv[1] # 获取命令行参数
    # print(sys.argv)
    soup = preprocess(item)
    summary = get_summary(soup)
    summary2 = clean_summary(summary)
    print(summary2)


