import torch
import numpy as np


def evaluate_model(model, device, test_loader):
    """
    Model evaulation (on whole test loader)
    """
    model.eval()
    y_pred_list = []
    y_test_list = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            test_h = model.init_hidden(labels.size(0))

            output, val_h = model(inputs, test_h)
            y_pred_test = torch.argmax(output, dim=1)
            y_pred_list.extend(y_pred_test.squeeze().tolist())
            y_test_list.extend(labels.squeeze().tolist())

    return y_pred_list, y_test_list


# Helper funtions to predict on just one article (preprocess)


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


def predict(model, device, txt):
    """
    predict on one given article
    """
    # unpack corpus
    corpus = 0
    txt_int = text_preprocess(txt, corpus)
    test_h = model.init_hidden(1)
    input = torch.LongTensor([txt_int]).to(device)
    output, val_h = model(input, test_h)
    y_pred_test = int(torch.argmax(output, dim=1))
    return "fake" if y_pred_test == 0 else "true"
