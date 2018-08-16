import argparse
import pandas as pd

from datetime import datetime
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument(
    "--submission1",
    default="../data/experiments/resnet34/submission.csv",
    type=Path,
    help="Path of a submission file")
parser.add_argument(
    "--submission2",
    default="../data/experiments/vgg16/submission.csv",
    type=Path,
    help="Path of a submission file")


def main():
    args = parser.parse_args()

    submission1 = pd.read_csv(args.submission1)
    submission2 = pd.read_csv(args.submission2)
    for fn1, fn2 in zip(submission1.filename, submission2.filename):
        assert fn1 == fn2
    submission_ensemble = pd.DataFrame(
        {
            "filename|defect":
            submission1.filename,
            "probability":
            (submission1.probability.values + submission2.probability.values) /
            2
        },
        columns=["filename|defect", "probability"])

    submission_ensemble.to_csv(
        f"../submit/submit_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        index=False)


if __name__ == "__main__":
    main()
