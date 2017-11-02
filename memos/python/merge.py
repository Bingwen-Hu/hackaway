import os
import pandas as pd

all_dirs = os.walk('2017')
data = pd.DataFrame(columns=['title', 'url', 'blog'])
for root, _, files in all_dirs:
    for file in files:
        try:
            path = os.path.join(root, file)
            excel = pd.ExcelFile(path)
            df = excel.parse(excel.sheet_names[0], names=['title', 'url', 'blog'])
            data = data.append(df)
        except Exception as e:
            print(e, path)
data.to_csv("all_data.csv", index=False)
        