import pandas as pd

from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class XuelangDataset(Dataset):
    def __init__(self, csv_path):
        df = pd.read_csv(csv_path)
        self.im_paths = df["filename"]
        self.labels = df["defect_code"]

    def __len__(self):
        return len(self.im_paths)

    def __getitem__(self, idx):
        im = Image.open(self.im_paths[idx])
        lb = self.labels[idx]
        return im, lb


class TransformDataset(Dataset):
    """
    For easier visualization in jupyter notebook.
    """

    def __init__(self, dataset, transform):
        self._dataset = dataset
        self._transform = transform

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, index):
        im, lb = self._dataset[index]
        return self._transform(im), lb


def get_image_transform(size=224, do_aug=True):
    _trans = []
    if size:
        _trans.append(transforms.Resize(size))
    if do_aug:
        _trans.append(transforms.RandomHorizontalFlip())
        _trans.append(transforms.RandomVerticalFlip())
    _trans.append(transforms.ToTensor())
    _trans.append(
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    return transforms.Compose(_trans)


def fetch_dataloader(csv_dir, im_size, batch_size, num_workers=1):
    dataloaders = []
    for dataset in ["train", "valid", "test_a", "test_b"]:
        csv = Path(csv_dir) / f"{dataset}.csv"
        if dataset == "train":
            dl = DataLoader(
                TransformDataset(
                    XuelangDataset(csv), get_image_transform(im_size, True)),
                batch_size=batch_size,
                num_workers=num_workers,
                shuffle=True)
        else:
            dl = DataLoader(
                TransformDataset(
                    XuelangDataset(csv), get_image_transform(im_size, False)),
                batch_size=batch_size,
                num_workers=num_workers,
                shuffle=False)
        dataloaders.append(dl)

    return dataloaders


if __name__ == "__main__":
    dls = fetch_dataloader("../data", 224, 4)
    x, y = next(iter(dls[0]))
    print(x.shape, y)
    print([len(dl.dataset) for dl in dls])