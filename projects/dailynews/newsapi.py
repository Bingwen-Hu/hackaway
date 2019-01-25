import time
import requests
import pickle

from datetime import datetime


## global settings
ACCOUNT_LIST = [
    'rgznai100', #AI科技大本营
    'naodong-open', #AI派 
    'redstonewill', #AI有道：微信号: 
    'aicvml', #我爱计算机视觉：微信号: 
    'yizhennotes', #机器学习算法与自然语言处理：微信号: 
    'zenRRan', #深度学习自然语言处理：微信号: 
    'datayx', #机器学习AI算法工程：微信号: 
    'gh_7ce29bea153b', #机器学习研究会订阅号：微信号: 
    'gh_7cbc376561eb', #语音杂谈：微信号: 
    'Jeemy110', #机器学习算法工程师：微信号: 
    'gh_61c0bdb49b7d', #AI深入浅出：微信号:
    'topitnews', #技术最前线
]


KEYWORDS = [
    'nlp', 'cnn', 'seq2seq', 'ai', 'attention',
    '深度学习', '自动驾驶', '人工智能', '注意力机制', '图像描述', '智能问答',
    '神经网络', '谷歌', '微软', '物体检测', '人脸识别', '文本识别', 
    'google', 'facebook', '语义分割', '语音合成', '语音识别',
    'pytorch', 'tensorflow', 
]


def score_result(result_lists):
    for news in result_lists:
        news['Score'] = 0
        for keyword in KEYWORDS:
            title = news['Title'].lower()
            if keyword in title:
                news['Score'] += 1
    

def write_cache():
    result_lists = parse_all_account()
    score_result(result_lists)
    result_lists.sort(key=lambda x: x['Score'], reverse=True)
    with open('cache.pk', 'wb') as f:
        pickle.dump(result_lists, f)


def read_cache():
    with open('cache.pk', 'rb') as f:
        result_lists = pickle.load(f)
    return result_lists


if __name__ == '__main__':
    write_cache()

    


