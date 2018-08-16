import argparse
import random
import pprint
import pandas as pd
import numpy as np

from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument(
    "--data_dir", default="data/round1", type=Path, help="Directory of images")
parser.add_argument(
    "--norm_ratio", default=0.4, type=float, help="ratio of normal samples during training phase")
parser.add_argument(
    "--seed",
    default=0,
    type=int,
    help="Random seed for train/validation split")

BACKGROUND = "正常"
defect_dict = {'正常': ['norm', 0],
               '扎洞': ['defect_1', 1],
               '毛斑': ['defect_2', 2],
               '擦洞': ['defect_3', 3],
               '毛洞': ['defect_4', 4],
               '织稀': ['defect_5', 5],
               '吊经': ['defect_6', 6],
               '缺经': ['defect_7', 7],
               '跳花': ['defect_8', 8],
               '污渍': ['defect_9', 9],
               '油渍': ['defect_9', 9],
               }


def get_label_table(train_paths):
    """
    Should be replaced by an official one in competition round2
    """
    labels_in_path = [p.parts[-2] for p in train_paths]
    defects = set(l for l in labels_in_path if l != BACKGROUND)
    labels = [BACKGROUND] + sorted(defects)
    label_dict = {}
    for l in labels:
        if l in defect_dict.keys():
            k, v = l, defect_dict[l][1]
        else:
            k, v = l, 10
        label_dict[k] = v
    return label_dict


if __name__ == "__main__":
    args = parser.parse_args()

    # Get paths of trainig/testing images
    data_dir = Path(args.data_dir)
    train_jpgs = sorted(data_dir.glob("*round1*/*/*.jpg"))
    test_a_jpgs = sorted(data_dir.glob("*round2_test_a*/*.jpg"))
    test_b_jpgs = sorted(data_dir.glob("*round2_test_b*/*.jpg"))

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
            idxs = [random.randint(0, 10) for _ in jpgs]
            is_defect = [random.randint(0, 1) for _ in jpgs]
        df = pd.DataFrame(
            {
                "filename": [str(p) for p in jpgs],
                "defect_code": idxs,
                "is_defect": is_defect
            },
            columns=["filename", "defect_code", "is_defect"])
        length = len(df)

        if "test" not in dataset:
            split = int((1 - args.norm_ratio) * length)
            used_idx = np.random.permutation(length)[:split]
            df = df.drop(used_idx)
        print('{} phase length of df changed from {} to {}'.format(dataset,length, len(df)))
        df.to_csv(data_dir / f"{dataset}.csv", index=False)
