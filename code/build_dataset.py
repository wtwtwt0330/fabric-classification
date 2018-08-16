import argparse
import random
import pprint
import pandas as pd

from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument(
    "--data_dir", default="data/round1", type=Path, help="Directory of images")
parser.add_argument(
    "--seed",
    default=0,
    type=int,
    help="Random seed for train/validation split")

BACKGROUND = "正常"


def get_label_table(train_paths):
    """
    Should be replaced by an official one in competition round2
    """
    labels_in_path = [p.parts[-2] for p in train_paths]
    defects = set(l for l in labels_in_path if l != BACKGROUND)
    labels = [BACKGROUND] + sorted(defects)
    return {l: idx for idx, l in enumerate(labels)}


if __name__ == "__main__":
    args = parser.parse_args()

    # Get paths of trainig/testing images
    data_dir = Path(args.data_dir)
    train_jpgs = sorted(data_dir.glob("*train*/*/*.jpg"))
    test_a_jpgs = sorted(data_dir.glob("*test_a*/*.jpg"))
    test_b_jpgs = sorted(data_dir.glob("*test_b*/*.jpg"))

    # Train/validation split
    random.seed(args.seed)
    random.shuffle(train_jpgs)

    split = int(0.8 * len(train_jpgs))
    jpg_sets = {
        "train": train_jpgs[:split],
        "valid": train_jpgs[split:],
        "test_a": test_a_jpgs,
        "test_b": test_b_jpgs
    }

    # Get a table to transform image labels to their indices
    label2idx = get_label_table(train_jpgs)
    pprint.pprint(label2idx)

    # Save image info to csv files for data loading
    for dataset, jpgs in jpg_sets.items():
        if "test" not in dataset:
            idxs = [label2idx[p.parts[-2]] for p in jpgs]
            is_defect = [1 if idx > 0 else 0 for idx in idxs]
        else:
            # pseudo labels for simplicity
            idxs = [0] * len(jpgs)
            is_defect = [random.randint(0, 1) for _ in jpgs]
        df = pd.DataFrame(
            {
                "filename": [str(p) for p in jpgs],
                "defect_code": idxs,
                "is_defect": is_defect
            },
            columns=["filename", "defect_code", "is_defect"])
        df.to_csv(data_dir / f"{dataset}.csv", index=False)
