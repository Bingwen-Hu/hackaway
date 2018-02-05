# datalink: https://archive.ics.uci.edu/ml/machine-learning-databases/00321/LD2011_2014.txt.zip

# preprocess
import numpy as np
import matplotlib.pyplot as plt
import os
import re


with open("LD2011_2014.txt") as fld:
    fld.readline()
    data = []
    cid = 250
    for line in fld:
        if line.startswith(";"):
            continue
        cols = [float(re.sub(',', '.', x))
                for x in line.strip().split(';')[1:]]
        data.append(cols[cid])

NUM_ENTRIES = 1000
plt.plot(range(NUM_ENTRIES), data[0:NUM_ENTRIES])
plt.ylabel('electricity consumption')
plt.xlabel('time (1pt = 15 mins)')
plt.show()

np.save("LD_250.npy", np.array(data))