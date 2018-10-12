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
    
