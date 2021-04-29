import pandas as pd


def load_data_csv(path):
    data = pd.read_csv(path, index_col=0)
    return data