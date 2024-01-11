import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from collections import Counter
import pickle as pkl

import sys

sys.path.append("path/to/parent/directory")

from config import PATH_TO_DATA, SEQ_LEN
from embeddings import create_embedding_matrix, create_word2vec_model


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


if __name__ == "__main__":
    print("Data preparing....")
    data = pd.read_csv(PATH_TO_DATA)

    data["tokens"] = data.apply(lambda row: word_tokenize(row["clear_text"]), axis=1)

    corpus = create_corpus(data["tokens"])
    tokenized_articles = prepare_data(data["tokens"], SEQ_LEN)
    vocab_size = len(corpus) + 1

    with open("../../models/pickles/corpus.pkl", "wb") as fp:
        pkl.dump(corpus, fp)

    with open("../../models/pickles/tokenized_articles.pkl", "wb") as fp:
        pkl.dump(tokenized_articles, fp)

    print("Saved tokenized articles")

    print("Creating embeddings....")

    word2vec_model = create_word2vec_model(data["tokens"], data["label"])
    embedding_matrix = create_embedding_matrix(word2vec_model, vocab_size, corpus)

    with open("../../models/pickles/embedding_matrix.pkl", "wb") as fp:
        pkl.dump(embedding_matrix, fp)

    print("Saved embedding matrix")
