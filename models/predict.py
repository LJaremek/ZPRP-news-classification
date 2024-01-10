from lstm import LSTM_Classifier
from config import * 
import numpy as np
import torch
import torch.nn as nn
import pickle as pkl

PATH_TO_MODEL = './checkpoints/lstm-best.pt'
TEXT = 'Since the beginning of the war in Ukraine, there has been a trend in which anti-vaccine accounts have been changing their narrative towards supporting Kremlin propaganda. This phenomenon shows how strong Russian influence is. The following text gives an insight into how conspiracy groups, consciously or not, spread the disinformation generated in Moscow. Due to the lack of in-depth research in this area, we decided to produce a report in which we took a closer look at this issue, based on social media. This research was conducted in cooperation with a group of volunteers, without whom the analysis of several hundred social media accounts would not be possible. We highly appreciate their hard work! The team led by the author of this report included the following people:'


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


def predict(txt):
    """
    predict on one given article
    """
    device = torch.device("cuda")

    print("Loading model...")

    with open('./pickles/embedding_matrix1.pkl', 'rb') as fp:
        embedding_matrix = pkl.load(fp)

    print(embedding_matrix.shape)
        
    with open('./pickles/corpus.pkl', 'rb') as fp:
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

    model.load_state_dict(torch.load(PATH_TO_MODEL))

    print("Model loaded successfully!")
    model.eval()

    txt_int = text_preprocess(txt, corpus, SEQ_LEN)
    test_h = model.init_hidden(1, device)
    input = torch.LongTensor([txt_int]).to(device)
    output, val_h = model(input, test_h)
    y_pred_test = int(torch.argmax(output, dim=1))

    return "fake" if y_pred_test == 0 else "true"


if __name__ == "__main__":
    print(f'Prediction: {predict(TEXT)}')


