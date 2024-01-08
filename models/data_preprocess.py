import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from collections import Counter


# Changes in DataFrame


def tokenize(data: pd.DataFrame, text_column: str) -> None:
    """
    adds extra columns with tokens
    """
    data["tokens"] = data.apply(
        lambda row: word_tokenize(row[text_column].lower()), axis=1
    )


def label_change(data: pd.DataFrame, label_column: str) -> pd.DataFrame:
    """
    changes label column from str representation to int
    """
    data = data.replace({label_column: {"real": 1, "fake": 0}})
    return data


# Tokenization


def create_corpus(token_column: pd.Series) -> Counter:
    """
    Create vocabulary of words from all texts in dataframe
    with its frequency of occurence
    """
    corpus = []
    for text in token_column:
        corpus.extend(text)
    count_words = Counter(corpus)

    return count_words.most_common()


def corpus_to_int(corpus: Counter, token_column) -> list[int]:
    """
    Changes text represantation to numbers
    """
    word_to_int = {word: n + 1 for n, (word, count) in enumerate(corpus)}
    text_int = []
    for text in token_column:
        article = [word_to_int[word] for word in text]
        text_int.append(article)
    return text_int


def pad_tokens(text_int: list[int], seq_len: int) -> np.array:
    """
    Pads tokens to given value of sequence length
    """
    ready_articles = np.zeros((len(text_int), seq_len), dtype=int)

    for n, article in enumerate(text_int):
        if len(article) <= seq_len:
            zeros = list(np.zeros(seq_len - len(article)))
            new = zeros + article
        else:
            new = article[:seq_len]
        ready_articles[n, :] = np.array(new)

    return ready_articles


def prepare_data(token_column: pd.Series, seq_len: int) -> np.array:
    """
    returns array of ints of every article from dataframe
    """
    corpus = create_corpus(token_column)
    text_int = corpus_to_int(corpus, token_column)
    data = pad_tokens(text_int, seq_len)
    return data
