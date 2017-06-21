import numpy as np

data = np.array([[0,0,11,22,3,0,0,0,0,0],
                 [1,2,10,12,0,0,0,3,4,6],
                 [0,0,13,14,0,0,7,0,0,9],
                 [0,0,0,0,0,0,0,0,15,17],
                 [0,0,0,0,28,27,0,0,0,0],
                 [0,0,0,30,19,0,0,0,0,0]])

dataDict = {}

for i in range(len(data)):
    for j in range(len(data[0])):
        if data[i][j] == 0:
            continue
        else:
            key = data[i][j]
            dataDict.update({key: [i, j]})

def near(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    return abs(x2-x1)<=1 and abs(y1-y2)<=1

rawlst = list(dataDict.keys())

result = []

while rawlst:
    iterId = rawlst.pop()
    s = set()
    s.add(iterId)
    n = set()
    n.add(iterId)
    
    while n:
        i = n.pop()
        pos1 = dataDict.get(i)
        for id in rawlst:
            pos2 = dataDict.get(id)
            if near(pos1, pos2):
                n.add(id)
                s.add(id)
                rawlst.remove(id)
    result.append(s)
