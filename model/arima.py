

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
import numpy as np

def load_data():
    df = pd.read_csv("../data/data_preprocessed/processed_data.csv")
    df = df.set_index(index=["Country", "Year"])
    return df
