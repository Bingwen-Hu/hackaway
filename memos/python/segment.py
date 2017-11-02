import re
from jieba import tokenize

def clean(data):
    data = re.sub('[的了]', '', data)
    data = re.sub('[#@-]', '', data)
    data = re.sub('[?？！，。°]', '', data)
    data = re.sub('[“”""" ]', '', data)
    return data

with open('../data/want2.txt', encoding='utf-8') as f:
    data = f.read().strip()

data = clean(data)
data = data[6:]


lines = data.split('\n')
words = [list(tokenize(l)) for l in lines]

with open('../data/want2_segment.txt', 'w', encoding='utf-8') as f:
    for line in words:
        for w, _, _ in line:
            f.write(w)
            f.write(' ')
        f.write('\n')
