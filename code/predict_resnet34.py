import argparse
import random
import pandas as pd
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from data_loader import XuelangDataset, TransformDataset, get_image_transform
from model import ResNetClassifier
from utils import evaluate, set_random_seed

parser = argparse.ArgumentParser()
parser.add_argument(
    "--data_path",
    default="../data/test_b.csv",
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
    model = ResNetClassifier()
    model.load_state_dict(torch.load(args.model_dir / "best.pth"))
    model.to(device)

    test_dl = DataLoader(
        TransformDataset(
            XuelangDataset(args.data_path), get_image_transform(299, False)),
        batch_size=32,
        num_workers=0,
        shuffle=False)

    test_aug_dl = DataLoader(
        TransformDataset(
            XuelangDataset(args.data_path), get_image_transform(299, True)),
        batch_size=32,
        num_workers=0,
        shuffle=False)

    preds = []
    # evaluate on test set
    pred, _, _ = evaluate(model, torch.nn.CrossEntropyLoss(), test_dl, device)
    preds.append(pred)
    # evaluate on test set with augmentation
    for _ in range(4):
        pred, _, _ = evaluate(model, torch.nn.CrossEntropyLoss(), test_aug_dl,
                              device)
        preds.append(pred)

    pred = sum(preds) / 5
    df = pd.DataFrame(
        {
            "filename":
            [Path(p).parts[-1] for p in test_dl.dataset._dataset.im_paths],
            "probability":
            pred[:, 1]
        },
        columns=["filename", "probability"])
    df.to_csv(args.model_dir / "submission.csv", index=False)


if __name__ == "__main__":
    main()
