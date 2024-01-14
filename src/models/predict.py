import os

from src.models.lstm import LSTM_Classifier
import numpy as np
import torch
import torch.nn.functional as F
import pickle as pkl
from typing import Tuple
from src.config import *

parent_dir = os.path.dirname(os.path.realpath(__file__))
PATH_TO_MODEL = f"{parent_dir}/checkpoints/lstm-best.pt"
TEXT = "In its court address today, South Africa has argued that Israel is committing genocide in Gaza, citing the large number of civilian casualties, the displacement of population, the lack of safe shelter and poor humanitarian conditions. “Genocides are never declared in advance, but this court has the benefit of the past 13 weeks of evidence that shows incontrovertibly a pattern of conduct and related intention that justifies as a plausible claim of genocidal acts,” South African lawyer Adila Hassim told the judges. In another passage, statements by leading Israeli officials including prime minister Benjamin Netanyahu, defence minister Yoav Gallant and president Isaac Herzog were cited."


def vocab_to_int_one(corpus, text):
    """
    changes one text in str to int representation
    """
    vocab_to_int = {word: n + 1 for n, (word, counter) in enumerate(corpus)}
    text_int = []
    for word in text:
        if word in vocab_to_int.keys():
            text_int.append(vocab_to_int[word])
    return text_int


def pad_tokens_one(article, seq_len):
    """
    pad tokens in one article
    """
    if len(article) <= seq_len:
        zeros = list(np.zeros(seq_len - len(article)))
        new = zeros + article
    else:
        new = article[:seq_len]
    return new


def text_preprocess(txt, corpus, seq_len):
    """
    preprocess text (from raw article to int sequence)
    """
    txt_int = vocab_to_int_one(corpus, txt)
    ready = pad_tokens_one(txt_int, seq_len)
    return ready


def predict(txt: str, use_cuda: bool = True) -> Tuple[str, float]:
    """
    predict on one given article
    """
    device = torch.device("cuda" if use_cuda else "cpu")

    print("Loading model...")

    with open(f"{parent_dir}/pickles/embedding_matrix.pkl", "rb") as fp:
        embedding_matrix = pkl.load(fp)

    with open(f"{parent_dir}/pickles/corpus.pkl", "rb") as fp:
        corpus = pkl.load(fp)

    vocab_size = len(corpus) + 1

    model = LSTM_Classifier(
        vocab_size,
        EMBEDDING_DIM,
        HIDDEN_DIM,
        NUM_CLASSES,
        LSTM_LAYERS,
        DROPOUT,
        IS_BIDIRECTIONAL,
    )

    model = model.to(device)

    # Initialize the embedding layer with the previously defined embedding matrix
    model.embedding.weight.data.copy_(torch.from_numpy(embedding_matrix))

    model.load_state_dict(
        torch.load(PATH_TO_MODEL, map_location=None if use_cuda else "cpu")
    )

    print("Model loaded successfully!")
    model.eval()

    txt_int = text_preprocess(txt, corpus, SEQ_LEN)
    test_h = model.init_hidden(1, device)
    input = torch.LongTensor([txt_int]).to(device)
    output, val_h = model(input, test_h)
    y_pred_test = int(torch.argmax(output, dim=1))

    prob = F.softmax(output, dim=-1)
    confidence = prob[0][y_pred_test] * 100

    result = "fake" if y_pred_test == 0 else "true"

    return result, float(confidence.cpu().detach().numpy())


if __name__ == "__main__":
    print(f"Prediction: {predict(TEXT)}")
