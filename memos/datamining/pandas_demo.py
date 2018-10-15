import pandas as pd
import numpy as np

index = ['Mory', 'Ann']
columns = ['Windy', 'Sunny', 'Snowy', 'Thundery', 'Soild', 'Lighting']
data = {
    'Mory': [2.0, 4.0, 6.0, 7.0, 6.0, 5.0],
    'Ann': [1.0, 5.0, 1.0, 1.0, 1.0, 1.0],
}


df = pd.DataFrame(index=index, columns=columns, dtype=np.float64)

for (k, v) in data.items():
    df.T[k] = v

print(df)
    

######## demo2
data = {
    'Name': ['Mory', 'Ann', 'Jenny'],
    'Dream': ['Become a leader', 'Maybe world will enlightened', 'Everyone in happiness'],
    'Level': [2.0, 5.0, 2.5]
}

df_surpass = pd.DataFrame(data=data, index=[1, 2, 3])

ann = df_surpass.iloc[1]
mory = df_surpass.iloc[0]

df_surpass.loc[4] = 'Know myself', 3.5, 'Demon'
print(df_surpass)