from sklearn.preprocessing import OneHotEncoder, MultiLabelBinarizer
import numpy as np

data = np.array([
    [0, 1], 
    [2, 2],
    [4, 3],
])

encoder = OneHotEncoder()

encoder.fit(data)

data_n = np.array([
    [1, 2],
    [2, 1],
    [2, 1],
])

trans = encoder.transform(data_n).toarray()

mlb = MultiLabelBinarizer()
data = np.array(['A', 'B', 'C'])
mlb.fit(data)
mlb.transform(['A', 'B'])


# standardlize and inverse
from sklearn.preprocessing import StandardScaler

sample = np.array([
    [38.1455, 137.454],
    [42.456, 173.345],
    [32.456, 165.34],
])

scaler = StandardScaler()
std_data = scaler.fit_transform(sample)
org_data = scaler.inverse_transform(std_data)
