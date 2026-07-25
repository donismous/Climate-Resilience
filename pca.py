import pandas as pd

def load_data():
    df = pd.read_csv("data/data_preprocessed/processed_data.csv")
    df = df.set_index(["Country", "Year"])
    return df

df = pd.read_csv("data/data_preprocessed/processed_data.csv")
