import pandas as pd
import re


data = pd.read_excel("data.xlsx")
content = data['正文']
lst = list(content)

def is_ads(message):
    words = ['足彩', '基本面', '推荐']
    for f in words:
        if f in message:
            return True
    return False

lst = [re.split("[，：“”。！？]", l) for l in lst if not is_ads(l)]
lst = [l for ls in lst for l in ls]
most = re.compile("\D\d[:-]\d\D")
c_most = [l for l in lst if re.search(most, l)]

bands = ['葡萄牙', '阿尔及利亚', '突尼斯', '法国', '德国', '巴西', '英格兰', '英国', '尼日利亚']

patterns = [
    '11打10',
    '11胜',
    '12场比赛',
    '11零封',
    '1-0击败',
    '打进',
    '打入9球',
    '1:0',
    '攻入一球',
    '助攻',
    '晋级',
    '黄牌',
    'vs',
    '首发',
    '4粒',
    '连胜',
    '2胜3负',
    '三连冠',
    '出场',
    '胜率',
    '第21轮',
    '完胜',
]