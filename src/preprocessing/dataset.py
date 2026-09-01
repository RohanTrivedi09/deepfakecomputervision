"""Dataset/dataloader construction for the train/valid/test splits.

data/processed/<split>/<class>/*.jpg already matches the layout torchvision's
ImageFolder expects, so we reuse it directly rather than writing a custom Dataset.
"""
from pathlib import Path

import torch
from torchvision import datasets, transforms

from src.common.paths import DATA_PROCESSED

IMG_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_transforms(train: bool) -> transforms.Compose:
    if train:
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def build_dataloaders(data_dir: Path = DATA_PROCESSED, batch_size: int = 32, num_workers: int = 2):
    train_ds = datasets.ImageFolder(data_dir / "train", transform=build_transforms(train=True))
    valid_ds = datasets.ImageFolder(data_dir / "valid", transform=build_transforms(train=False))
    test_ds = datasets.ImageFolder(data_dir / "test", transform=build_transforms(train=False))

    # ImageFolder assigns indices alphabetically: fake=0, real=1. Every script
    # that loads a checkpoint relies on this same ordering being saved alongside it.
    assert train_ds.classes == ["fake", "real"], f"unexpected class order: {train_ds.classes}"

    loaders = {
        "train": torch.utils.data.DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers
        ),
        "valid": torch.utils.data.DataLoader(
            valid_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
        ),
        "test": torch.utils.data.DataLoader(
            test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
        ),
    }
    return loaders, train_ds.classes
