import os
import random
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn

from sklearn.metrics import log_loss, accuracy_score, roc_curve, auc, average_precision_score
from sklearn.preprocessing import label_binarize
from tqdm import tqdm

import data_loader
import model

def multi_auc(labels, probas, classes=11):
    # mask = labels[labels!=0]
    # labels = labels[mask]
    # probas = probas[mask]
    
    labels = label_binarize(labels, classes=range(classes))
    fpr, tpr, _ = roc_curve(labels.ravel(), probas.ravel())
    roc_auc = auc(fpr, tpr)
    return roc_auc

def mAP_cal(labels, probas, classes=11):
    # mask = labels[labels!=0]
    # labels = labels[mask]
    # probas = probas[mask]
    
    labels = label_binarize(labels, classes=range(classes))
    mAP = average_precision_score(labels.ravel(), probas.ravel())
    return mAP


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
            
    # print(len(probas))

    probas = np.concatenate(probas)
    labels = np.concatenate(labels)
    metric = 0.7 * multi_auc(labels, probas) + 0.3 * mAP_cal(labels, probas)

    # compute all metrics after one epoch
    metrics = {
        "loss": log_loss(labels, probas),
        "accuracy": accuracy_score(labels, probas.argmax(1)),
        "AUC": metric
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
    best_valid_acc = float("-inf")
    best_valid_loss = float("inf")

    for epoch in range(n_epochs):
        train_one_epoch(model, criterion, optimizer, train_dl, device)

        _, _, train_metrics = evaluate(model, criterion, train_dl, device)
        _, _, valid_metrics = evaluate(model, criterion, valid_dl, device)

        print("- Train metrics", report_metrics(train_metrics))
        print("- Valid metrics", report_metrics(valid_metrics))

        valid_AUC = valid_metrics["AUC"]
        valid_acc = valid_metrics["accuracy"]
        valid_loss = valid_metrics["loss"]
        # if valid_AUC > best_valid_AUC:
        #     print('model updated !!!')
        #     best_valid_AUC = valid_AUC
        #     torch.save(model.state_dict(), model_path)
        # if valid_acc > best_valid_acc:
        #     print('model updated !!!')
        #     best_valid_acc = valid_acc
        #     torch.save(model.state_dict(), model_path)
        if valid_loss < best_valid_loss:
            print('model updated !!!')
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), model_path)

def pd_entry_gen(probas, dl):
    dcs = ['norm', 
           'defect_1', 
           'defect_2', 
           'defect_3', 
           'defect_4', 
           'defect_5', 
           'defect_6', 
           'defect_7', 
           'defect_8',
           'defect_9',
           'defect_10']

    filenames = [Path(p).parts[-1] for p in dl.dataset._dataset.im_paths]

    entry_fns = []
    entry_probas = []
    for i in range(len(probas)):
        entry_fns.append([filenames[i] + '|' +  dc for dc in dcs])
        entry_probas.append(probas[i])

    entry_fns = np.concatenate(entry_fns)
    entry_probas = np.concatenate(entry_probas)
    return entry_probas, entry_fns
