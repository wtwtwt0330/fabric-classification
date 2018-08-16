import argparse
import random
import pandas as pd
from pathlib import Path
import numpy as np

import torch
from torch.utils.data import DataLoader

from data_loader import XuelangDataset, TransformDataset, get_image_transform
from model import ResNetClassifier
from utils import evaluate, set_random_seed, pd_entry_gen

parser = argparse.ArgumentParser()
parser.add_argument(
    "--data_path",
    default="../data/test_a.csv",
    type=Path,
    help="Path of test data")
parser.add_argument(
    "--model_dir",
    default="../data/experiments/resnet34/",
    type=Path,
    help="Directory for model weights and submission file")
parser.add_argument("--seed", default=0, type=int, help="Random seed")

def main():
    args = parser.parse_args()

    set_random_seed(args.seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = ResNetClassifier(n_class=11,freeze_features=False)
    model.load_state_dict(torch.load(args.model_dir / "best.pth"))
    model.to(device)

    test_dl = DataLoader(
        TransformDataset(
            XuelangDataset(args.data_path), get_image_transform(500, False)),
        batch_size=32,
        num_workers=0,
        shuffle=False)

    test_aug_dl = DataLoader(
        TransformDataset(
            XuelangDataset(args.data_path), get_image_transform(500, True)),
        batch_size=32,
        num_workers=0,
        shuffle=False)

    preds = []
    # evaluate on test set
    pred, _, _ = evaluate(model, torch.nn.CrossEntropyLoss(), test_dl, device)
    entry_probas, entry_fns =  pd_entry_gen(pred, test_dl)
    preds.append(entry_probas)

    # evaluate on test set with augmentation
    for _ in range(4):
        pred, _, _ = evaluate(model, torch.nn.CrossEntropyLoss(), test_aug_dl,
                              device)
        entry_probas, entry_fns =  pd_entry_gen(pred, test_aug_dl)
        preds.append(entry_probas)

    entry_probas_l = np.array(preds)
    avr_probas_entry = np.mean(entry_probas_l,axis=0)

    df = pd.DataFrame(
        {
            "filename|defect": entry_fns,
            "probability": avr_probas_entry,
        },
        columns=["filename|defect", "probability"])
    df.to_csv(args.model_dir / "submission.csv", index=False)
    print("submission.csv generated!!!")


if __name__ == "__main__":
    main()

