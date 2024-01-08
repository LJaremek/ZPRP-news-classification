import pandas as pd
from nltk.tokenize import word_tokenize


def tokenize(data: pd.DataFrame, text_col: str) -> None:
    data['tokens'] = data.apply(lambda row: word_tokenize(row[text_col].lower()), axis=1)


def label_change(data: pd.DataFrame, label_col: str) -> pd.DataFrame:
    data = data.replace({label_col: {'real': 1, 'fake': 0}})
    return data
