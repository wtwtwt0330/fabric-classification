import argparse
import random
from pathlib import Path

import torch

from model import ResNetClassifier
from utils import train, set_random_seed
from data_loader import fetch_dataloader

parser = argparse.ArgumentParser()
parser.add_argument(
    "--data_dir", default="../data", type=Path, help="Directory of data")
parser.add_argument(
    "--model_dir",
    default="../data/experiments/resnet34",
    type=Path,
    help="Directory for model weights and submission file")
parser.add_argument("--seed", default=0, type=int, help="Random seed")


def main():
    args = parser.parse_args()

    set_random_seed(args.seed)


    model_dir = args.model_dir
    model_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = ResNetClassifier().to(device)
    crit = torch.nn.CrossEntropyLoss()

    # stage 1
    lr = 1e-5
    size = 224
    epochs = 5
    batch_size = 32
    train_dl, valid_dl, _, _ = fetch_dataloader(
        args.data_dir, size, batch_size, num_workers=4)
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr)
    train(model, train_dl, valid_dl, crit, optimizer, epochs, device,
          model_dir / "best_224.pth")

    model.load_state_dict(torch.load(model_dir / "best_224.pth"))
    # stage 2
    lr = 5e-6
    size = 299
    epochs = 15
    batch_size = 32
    train_dl, valid_dl, _, _ = fetch_dataloader(
        args.data_dir, size, batch_size, num_workers=0)
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr)
    train(model, train_dl, valid_dl, crit, optimizer, epochs, device,
          model_dir / "best.pth")


if __name__ == "__main__":
    main()