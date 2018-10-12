import pandas as pd


index = ['Mory', 'Ann']
columns = ['Windy', 'Sunny', 'Snowy', 'Thundery', 'Soild', 'Lighting']
data = {
    'Mory': ['0.0', '0.0', '0.0', '0.0', '0.0', '0.0'],
    'Ann': ['1.0', '1.0', '1.0', '1.0', '1.0', '1.0'],
}


df = pd.DataFrame(index=index, columns=columns)

for (k, v) in data.items():
    df.T[k] = v

print(df)
    
