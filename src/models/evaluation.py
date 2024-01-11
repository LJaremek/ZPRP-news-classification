import torch
import numpy as np
import pickle as pkl
from lstm import LSTM_Classifier
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import classification_report
from typing import List, Tuple

import sys

sys.path.append("path/to/parent/directory")

from config import (
    PATH_TO_DATA,
    EMBEDDING_DIM,
    TEST_SIZE,
    RANDOM_STATE,
    HIDDEN_DIM,
    NUM_CLASSES,
    LSTM_LAYERS,
    DROPOUT,
    IS_BIDIRECTIONAL,
    BATCH_SIZE,
)


PATH_TO_MODEL = "../../models/checkpoints/your_ckpt.pt"
PATH_TO_DATA = "path_to_data"


def create_test_loader(path: str) -> DataLoader:
    """
    creates test loader from path
    """
    data = pd.read_csv(path)

    with open("../../models/pickles/tokenized_articles.pkl", "rb") as fp:
        tokenized_articles = pkl.load(fp)

    X = tokenized_articles
    y = data["label"].values

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    test_data = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))

    test_loader = DataLoader(
        test_data, shuffle=False, batch_size=BATCH_SIZE, drop_last=True
    )

    return test_loader


def evaluate_model(
    model, device: torch.device, test_loader: DataLoader
) -> Tuple[List[int], List[int]]:
    """
    Model evaulation (on whole test loader)
    """
    model.eval()
    y_pred_list = []
    y_test_list = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            test_h = model.init_hidden(labels.size(0), device)

            output, val_h = model(inputs, test_h)
            y_pred_test = torch.argmax(output, dim=1)
            y_pred_list.extend(y_pred_test.squeeze().tolist())
            y_test_list.extend(labels.squeeze().tolist())

    return y_pred_list, y_test_list


if __name__ == "__main__":
    print("Loading model...")

    device = torch.device("cuda")

    with open("../../models/pickles/embedding_matrix.pkl", "rb") as fp:
        embedding_matrix = pkl.load(fp)

    with open("../../models/pickles/corpus.pkl", "rb") as fp:
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

    model.eval()

    test_loader = create_test_loader(PATH_TO_DATA)

    print("Evaluating....")

    y_pred_list, y_test_list = evaluate_model(model, device, test_loader)
    print(
        "Classification Report:\n",
        classification_report(y_test_list, y_pred_list, target_names=["fake", "true"]),
    )
