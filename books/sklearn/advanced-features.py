# ==================== load the data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
titanic = pd.read_csv('data/titanic.csv')

print(titanic)


# ==================== feature extraction
# the following function can encode the categorical attributes to numeric.
from sklearn import feature_extraction
def one_hot_dataframe(data, cols, replace=False):
    vec = feature_extraction.DictVectorizer()
    mkdict = lambda row: dict((col, row[col]) for col in cols)
    vecData = pd.DataFrame(vec.fit_transform(data[cols].apply(mkdict, axis=1)).toarray())
    vecData.columns = vec.get_feature_names()
    vecData.index = data.index
    if replace:
        data = data.drop(cols, axis=1)
        data = data.join(vecData)
    return (data, vecData)

# ==================== feature selection
