# general purpose
from tqdm import tqdm
import numpy as np
import datetime
import pandas as pd
import os

# for neural network train
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
import torch
import torch.nn as nn

# docs
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report
import logging as logger
import pickle as pkl

# from our project
from lstm import LSTM_Classifier
from evaluation import evaluate_model

import sys

sys.path.append("path/to/parent/dir")

from config import *


# datetime
now = datetime.datetime.now()
time_str = now.strftime("[%m-%d]-[%H-%M]-")

# logger
log_format = "[%(levelname)s %(asctime)s %(filename)s] %(message)s"
logger.basicConfig(level=logger.INFO, format=log_format, datefmt="%Y-%m-%d %H:%M:%S")
fh = logger.FileHandler(f"../../models/log/{time_str}lstm.txt")
fh.setFormatter(logger.Formatter(log_format))
logger.getLogger().addHandler(fh)


def train(model, device, train_loader, valid_loader, criterion, optimizer):
    """
    performs training
    """
    total_step = len(train_loader)
    total_step_val = len(valid_loader)

    early_stopping_patience = 4
    early_stopping_counter = 0

    valid_acc_max = 0

    recorder = RecorderMeter(EPOCHS + 1)

    for epoch in tqdm(range(1, EPOCHS + 1)):
        train_loss, valid_loss = [], []
        train_acc, valid_acc = [], []

        y_train_list, y_val_list = [], []

        correct, correct_val = 0, 0
        total, total_val = 0, 0
        running_loss, running_loss_val = 0, 0

        # training

        model.train()

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            h = model.init_hidden(labels.size(0), device)

            model.zero_grad()

            output, h = model(inputs, h)

            loss = criterion(output, labels)
            loss.backward()

            running_loss += loss.item()

            optimizer.step()

            y_pred_train = torch.argmax(output, dim=1)
            y_train_list.extend(y_pred_train.squeeze().tolist())

            correct += torch.sum(y_pred_train == labels).item()
            total += labels.size(0)

        train_loss.append(running_loss / total_step)
        train_acc.append(100 * correct / total)

        # validation

        with torch.no_grad():
            model.eval()

            for inputs, labels in valid_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                val_h = model.init_hidden(labels.size(0), device)

                output, val_h = model(inputs, val_h)

                val_loss = criterion(output, labels)
                running_loss_val += val_loss.item()

                y_pred_val = torch.argmax(output, dim=1)
                y_val_list.extend(y_pred_val.squeeze().tolist())

                correct_val += torch.sum(y_pred_val == labels).item()
                total_val += labels.size(0)

            valid_loss.append(running_loss_val / total_step_val)
            valid_acc.append(100 * correct_val / total_val)

        # Save model if validation accuracy increases
        if np.mean(valid_acc) >= valid_acc_max:
            logger.info(
                f"Epoch {epoch+1}:Validation accuracy increased ({valid_acc_max:.6f} --> {np.mean(valid_acc):.6f}).  Saving model ..."
            )
            torch.save(model.state_dict(), f"../../models/checkpoints/{time_str}model.pt")
            valid_acc_max = np.mean(valid_acc)
            early_stopping_counter = 0
        else:
            logger.info(f"Epoch {epoch+1}:Validation accuracy did not increase")
            early_stopping_counter += 1

        # Early stopping if validation accuracy did not increase
        if early_stopping_counter > early_stopping_patience:
            logger.info("Early stopped at epoch :", epoch + 1)
            break

        # Update recorder and plot
        recorder.update(
            epoch,
            np.mean(train_loss),
            np.mean(train_acc),
            np.mean(valid_loss),
            np.mean(valid_acc),
        )
        curve_name = time_str + "cnn.png"
        recorder.plot_curve(os.path.join("../../models/log/", curve_name))

        logger.info(
            f"\tTrain_loss : {np.mean(train_loss):.4f} Val_loss : {np.mean(valid_loss):.4f}"
        )
        logger.info(
            f"\tTrain_acc : {np.mean(train_acc):.3f}% Val_acc : {np.mean(valid_acc):.3f}%"
        )


class RecorderMeter:
    """
    Computes and stores the minimum loss value and its epoch index
    Originally taken from POSTERv2: https://arxiv.org/pdf/2301.12149v2.pdf
    """

    def __init__(self, total_epoch):
        self.reset(total_epoch)

    def reset(self, total_epoch):
        self.total_epoch = total_epoch
        self.current_epoch = 0
        self.epoch_losses = np.zeros(
            (self.total_epoch, 2), dtype=np.float32
        )  # [epoch, train/val]
        self.epoch_accuracy = np.zeros(
            (self.total_epoch, 2), dtype=np.float32
        )  # [epoch, train/val]

    def update(self, idx, train_loss, train_acc, val_loss, val_acc):
        self.epoch_losses[idx, 0] = train_loss * 300
        self.epoch_losses[idx, 1] = val_loss * 300
        self.epoch_accuracy[idx, 0] = train_acc
        self.epoch_accuracy[idx, 1] = val_acc
        self.current_epoch = idx + 1

    def plot_curve(self, save_path):
        title = "the accuracy/loss curve of train/val"
        dpi = 80
        width, height = 1800, 800
        legend_fontsize = 10
        figsize = width / float(dpi), height / float(dpi)

        fig = plt.figure(figsize=figsize)
        x_axis = np.array([i for i in range(self.total_epoch)])  # epochs
        y_axis = np.zeros(self.total_epoch)

        plt.xlim(0, self.total_epoch)
        plt.ylim(0, 100)
        interval_y = 5
        interval_x = 5
        plt.xticks(np.arange(0, self.total_epoch + interval_x, interval_x))
        plt.yticks(np.arange(0, 100 + interval_y, interval_y))
        plt.grid()
        plt.title(title, fontsize=20)
        plt.xlabel("the training epoch", fontsize=16)
        plt.ylabel("accuracy", fontsize=16)

        y_axis[:] = self.epoch_accuracy[:, 0]
        plt.plot(x_axis, y_axis, color="g", linestyle="-", label="train-accuracy", lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        y_axis[:] = self.epoch_accuracy[:, 1]
        plt.plot(x_axis, y_axis, color="y", linestyle="-", label="valid-accuracy", lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        y_axis[:] = self.epoch_losses[:, 0]
        plt.plot(
            x_axis, y_axis, color="g", linestyle=":", label="train-loss-x300", lw=2
        )
        plt.legend(loc=4, fontsize=legend_fontsize)

        y_axis[:] = self.epoch_losses[:, 1]
        plt.plot(
            x_axis, y_axis, color="y", linestyle=":", label="valid-loss-x300", lw=2
        )
        plt.legend(loc=4, fontsize=legend_fontsize)

        if save_path is not None:
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
            print("Saved figure")
        plt.close(fig)


if __name__ == "__main__":
    logger.info("Initializing....")

    device = torch.device("cuda")
    data = pd.read_csv(PATH_TO_DATA)

    logger.info("Data loading....")

    with open("../../models/pickles/embedding_matrix.pkl", "rb") as fp:
        embedding_matrix = pkl.load(fp)

    with open("../../models/pickles/tokenized_articles.pkl", "rb") as fp:
        tokenized_articles = pkl.load(fp)

    with open("../../models/pickles/corpus.pkl", "rb") as fp:
        corpus = pkl.load(fp)

    vocab_size = len(corpus) + 1

    logger.info(f"Data loaded sucessfully. Vocabulary size: {vocab_size}")

    logger.info("Creating loaders...")

    X = tokenized_articles
    y = data["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train,
        y_train,
        test_size=VAL_SIZE,
        stratify=y_train,
        random_state=RANDOM_STATE,
    )

    train_data = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    test_data = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))
    valid_data = TensorDataset(torch.from_numpy(X_valid), torch.from_numpy(y_valid))

    train_loader = DataLoader(
        train_data, shuffle=True, batch_size=BATCH_SIZE, drop_last=True
    )
    valid_loader = DataLoader(
        valid_data, shuffle=False, batch_size=BATCH_SIZE, drop_last=True
    )
    test_loader = DataLoader(
        test_data, shuffle=False, batch_size=BATCH_SIZE, drop_last=True
    )

    logger.info("Building model......")

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
    criterion = nn.NLLLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=5e-6)

    # Initialize the embedding layer with the previously defined embedding matrix
    model.embedding.weight.data.copy_(torch.from_numpy(embedding_matrix))
    # Allow the embedding matrix to be fine-tuned to better adapt to our dataset and get higher accuracy
    model.embedding.weight.requires_grad = True

    logger.info("Begin training......")
    train(model, device, train_loader, valid_loader, criterion, optimizer)

    logger.info("EVALUATION")

    y_pred_list, y_test_list = evaluate_model(model, device, test_loader)
    logger.info(
        "Classification Report:\n")
    logger.info(
        classification_report(y_test_list, y_pred_list, target_names=["fake", "true"]),
    )
