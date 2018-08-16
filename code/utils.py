import os
import random
import numpy as np

import torch
import torch.nn as nn

from sklearn.metrics import log_loss, accuracy_score, roc_auc_score
from tqdm import tqdm

import data_loader
import model


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def to_np(x):
    return x.data.cpu().numpy()


def report_metrics(metrics):
    report = " ; ".join(
        "{}: {:06.4f}".format(k, v) for k, v in metrics.items())
    return report


def evaluate(model, criterion, dataloader, device):
    model.eval()

    probas = []
    labels = []
    # compute metrics over the dataset
    with torch.no_grad():
        for i, (batch_inputs, batch_labels) in enumerate(tqdm(dataloader)):
            # move to GPU if available
            batch_inputs = batch_inputs.to(device)
            batch_labels = batch_labels.to(device)  # shape: (batch_size,)

            # predict softmax probabilities
            batch_probas = model.predict_proba(
                batch_inputs)  # shape: (batch_size, 2)

            # collect predictions
            probas.append(to_np(batch_probas))
            labels.append(to_np(batch_labels))

    probas = np.vstack(probas)
    labels = np.concatenate(labels)

    # compute all metrics after one epoch
    metrics = {
        "loss": log_loss(labels, probas[:, 1]),
        "accuracy": accuracy_score(labels, probas.argmax(1)),
        "AUC": roc_auc_score(labels, probas[:, 1])
    }
    return probas, labels, metrics


def train_one_epoch(model, criterion, optimizer, dataloader, device):
    model.train()

    for i, (batch_inputs, batch_labels) in enumerate(dataloader):
        # move to GPU if available
        batch_inputs = batch_inputs.to(device)
        batch_labels = batch_labels.to(device)

        optimizer.zero_grad()

        outputs = model(batch_inputs)
        batch_loss = criterion(outputs, batch_labels)
        batch_loss.backward()

        optimizer.step()


def train(model, train_dl, valid_dl, criterion, optimizer, n_epochs, device,
          model_path):

    best_valid_AUC = float("-inf")

    for epoch in tqdm(range(n_epochs)):
        train_one_epoch(model, criterion, optimizer, train_dl, device)

        _, _, train_metrics = evaluate(model, criterion, train_dl, device)
        _, _, valid_metrics = evaluate(model, criterion, valid_dl, device)

        print("- Train metrics", report_metrics(train_metrics))
        print("- Valid metrics", report_metrics(valid_metrics))

        valid_AUC = valid_metrics["AUC"]
        if valid_AUC > best_valid_AUC:
            best_valid_AUC = valid_AUC
            torch.save(model.state_dict(), model_path)
