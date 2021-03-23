import pandas as pd
from scipy.sparse import hstack

data = pd.read_csv('salary-train.csv')

def lower_text(text: pd.Series) -> pd.Series:
    return text.str.lower()
lower_text(data)