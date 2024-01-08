import numpy as np
import torch
from config import EPOCHS


def train(model, device, train_loader, valid_loader, criterion, optimizer):
    total_step = len(train_loader)
    total_step_val = len(valid_loader)

    early_stopping_patience = 4
    early_stopping_counter = 0

    valid_acc_max = 0

    for epoch in range(EPOCHS):
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

            h = model.init_hidden(labels.size(0))

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

                val_h = model.init_hidden(labels.size(0))

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
            torch.save(model.state_dict(), "./state_dict.pt")
            print(
                f"Epoch {epoch+1}:Validation accuracy increased ({valid_acc_max:.6f} --> {np.mean(valid_acc):.6f}).  Saving model ..."
            )
            valid_acc_max = np.mean(valid_acc)
            early_stopping_counter = 0
        else:
            print(f"Epoch {epoch+1}:Validation accuracy did not increase")
            early_stopping_counter += 1

        # Early stopping if validation accuracy did not increase
        if early_stopping_counter > early_stopping_patience:
            print("Early stopped at epoch :", epoch + 1)
            break

        print(
            f"\tTrain_loss : {np.mean(train_loss):.4f} Val_loss : {np.mean(valid_loss):.4f}"
        )
        print(
            f"\tTrain_acc : {np.mean(train_acc):.3f}% Val_acc : {np.mean(valid_acc):.3f}%"
        )
