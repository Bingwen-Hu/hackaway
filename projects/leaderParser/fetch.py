# -*- coding: utf-8 -*-

import re
import requests
from bs4 import BeautifulSoup

from utils import preprocess

#==============================================================================
# Startup
#==============================================================================
def main(names):
    return [main_single(name) for name in names]

def main_single(name):
    soup = preprocess(name)
    paras = strategy_choose(soup)
    filter_paras = parse_filter(paras)
    return filter_paras

#==============================================================================
# Core function: fetch function
#==============================================================================


def strategy_choose(soup):
    """ 对soup进行分析，根据是否重名来选择不同的策略
    """
    global polylinks

    links = soup.find_all("ul", {"class": "polysemantList-wrapper"})
    if not links:
        paras = unique_fetch_strategy(soup)
        return paras
    else:
        polylinks.append(links)
        paras = unique_fetch_strategy(soup)
        return paras

def unique_fetch_strategy(soup):
    paras = soup.find_all("div", {"class": "para", "label-module": "para"})
    paras = clear_paras(paras)
    return paras

def poly_fetch_strategy(links):
    links = links.find_all("a")
    links = [link.get('href') for link in links]
    paras = [poly_fetch_helper(link)
            for link in links if link.startswith("/")]
    paras = [para for para in paras if para]
    return paras

def poly_fetch_helper(link):
    url = "http://baike.baidu.com" + link
    leader = requests.get(url)
    leader.encoding = "UTF-8"
    soup = BeautifulSoup(leader.text, 'lxml')
    paras = unique_fetch_strategy(soup)
    return paras

#==============================================================================
# core function: parse function
#==============================================================================
def parse_filter(para):
    pattern = re.compile("^\d{4}..*")
    para_new = [pattern.findall(p) for p in para]
    # \u3000 need to be remove
    return [p for p in para_new if p]

def parse_spilt(para):
    comma = re.compile('[， \u3000]')
    split_para = [re.split(comma, p[0], maxsplit=1) for p in para]
    return split_para

def parse_datetime(para):
    global time_pattern
    # note sps means split points
    sps = [time_pattern.match(p).end() for p in para]
    para = [(p[:sps[i]], p[sps[i]:]) for i, p in enumerate(para)]
    return para



def spilt_result_test(spilt_para):
    problems = [p for p in spilt_para if len(p) < 2]
    print("%d fail to spilt.." % len(problems))
    return problems



#==============================================================================
# test function
#==============================================================================
def test_leaders(leaders, number=100):
    import random
    random.shuffle(leaders)
    return leaders[:number]

#==============================================================================
# helper function
#==============================================================================
def prepare(pathname="../../projects/Leaders.xls",
            column="姓名", sheet="Leaders"):
    import pandas as pd
    excel = pd.ExcelFile(pathname)
    # print(excel.sheet_names)
    leaders = excel.parse(sheet) # # pandas.DataFrame Object
    leaders = leaders[column] # pandas.Series Object
    return leaders # simple list


def clear_paras(paras):
    remove_string = re.compile(r"<[^>]+>|\xa0|\n|\[\d+(\-\d+)?\]", re.S)
    paras_new = [remove_string.sub("", p.get_text()) for p in paras]
    return paras_new

def judge(soup):
    no_result = soup.find_all('p', {"class": "sorryCont"})
    if no_result:
        print("Sorry, no result.")
        return
    links = soup.find_all("ul", {"class": "polysemantList-wrapper"})
    if not links:
        print("Unique: go the first strategy")
    else:
        print("Poly: go the second strategy")

def fetch_name(soup):
    return soup.find('h1').get_text()

def judge_name(name):
    soup = preprocess(name)
    judge(soup)


#==============================================================================
#  Poly test module
#==============================================================================
def select_description(link):
    keywords = ['省','委','书记','政','省长','主任', "市长"]
    description = link.get_text()
    score = [1 for k in keywords if k in description]
    score = sum(score)
    if score > 0:
        print("go to next test %s\n" % description) #go to next test
    else:
        print("game over: %s\n" % description)

def select_experience(link):
    href = link['href']
    url = "http://baike.baidu.com" + href
    try:
        leader = requests.get(url)
    except:
        return "error"
    leader.encoding = "UTF-8"
    soup = BeautifulSoup(leader.text, 'lxml')
    h2 = soup.find_all('h2')
    if h2:
        keywords = fetch_name(soup)+"人物履历"
        text = [h.get_text() for h in h2]
        if keywords in text:
            print("go to next test\n")
        else:
            print("Game over\n")


def classify(soup):
    try:
        tags = soup.find_all("dd", {"id": "open-tag-item"})[0]
    except:
        return ["notag"]
#    h2s = soup.find_all("h2", {"class": "title-text"})
#    h2s = [h2.get_text() for h2 in h2s]
    class_str = tags.get_text()
    class_str = re.split(r"[\n， ]", class_str)
    class_str = [c for c in class_str if c]
    return class_str




#==============================================================================
# leader basic information scrape
#==============================================================================
def get_summary(name):
	"""人物简介"""
    soup = preprocess(name)
    summary = soup.find("div", {"class": 'lemma-summary'})
    return summary.get_text()



if __name__ == "__main__":
    polylinks = []
    time_pattern = re.compile(r"^[\d\.\-━―—－ 至今年月日起后到]+，?")
    baseUrl = "http://baike.baidu.com"
    leaders_experience = {"name": "", "experience": ""}
    leaders = ["习近平", "李克强", "胡锦涛"]

