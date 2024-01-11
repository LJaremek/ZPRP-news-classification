from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from gensim.models import Word2Vec
import sys

sys.path.append("/home/wpartycja/7sem/zrpr/zprp-projekt-z4/src")

from config import TEST_SIZE, RANDOM_STATE, VAL_SIZE, EMBEDDING_DIM


def get_X_train_text(text_column: pd.Series, label_column: pd.Series):
    """
    Find X_train subset on clean text from dataset
    to make word2vec model
    """
    X = text_column.to_list()
    y = label_column.to_list()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train,
        y_train,
        test_size=VAL_SIZE,
        stratify=y_train,
        random_state=RANDOM_STATE,
    )
    return X_train


def create_word2vec_model(text_column: pd.Series, label_column: pd.Series):
    """
    Creates word2vec model
    """
    X_train = get_X_train_text(text_column, label_column)
    X_train_new = [" ".join(article) for article in X_train]
    Word2vec_train_data = list(map(lambda x: x.split(), X_train_new))
    word2vec_model = Word2Vec(Word2vec_train_data, vector_size=EMBEDDING_DIM)

    return word2vec_model


def create_embedding_matrix(word2vec_model, vocab_size, corpus):
    """
    Makes embedding matrix
    """
    embedding_matrix = np.zeros((vocab_size, EMBEDDING_DIM))

    # Fill the embedding matrix with pre-trained values from word2vec
    for n, (word, token) in enumerate(corpus):
        # Check if the word is present in the word2vec model's vocabulary
        if word in word2vec_model.wv.key_to_index:
            # If the word is present, retrieve its embedding vector and add it to the embedding matrix
            embedding_vector = word2vec_model.wv[word]
            embedding_matrix[n] = embedding_vector

    return embedding_matrix
