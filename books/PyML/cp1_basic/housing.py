import pandas as pd


def load_housing_data():
    csv_path = 'housing.csv'
    return pd.read_csv(csv_path)

# data = load_housing_data()
# data.head(10)
# data.info()