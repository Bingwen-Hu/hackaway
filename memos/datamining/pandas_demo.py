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

df_surpass.sort_values(by='Level', ascending=False)

surpass_type = pd.Series(
    data=['light', 'demon', 'snow', np.nan],
    index=[2, 1, 3, 4]
)

df_surpass['SType'] = surpass_type
print(df_surpass)

df_surpass['SType'].fillna('ordinary', inplace=True)
print(df_surpass)

df_surpass['Level'] = df_surpass['Level'].map(lambda x: min(5, x+1))
print(df_surpass)

# demo dummy variable
data = pd.DataFrame(columns=['weekday'])
data.weekday = [i for i in range(1, 8)] * 3
data['score'] = 1.0
# perform dummy
dummy_data = pd.get_dummies(data.weekday, prefix='weekday')
# merge two
mergedata = pd.concat([data.drop(['weekday'], axis=1), dummy_data], axis=1)

# another merge method
data.join(dummy_data)


# Excel
excel = pd.ExcelWriter('demo.xlsx')
data.to_excel(excel, 'dummy')
excel.close()

# read back
dummy = pd.read_excel('demo.xlsx', 'dummy')
print(dummy)