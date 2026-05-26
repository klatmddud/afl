from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from scripts.runtime.config import deep_get


def build_image_transforms(train: bool) -> transforms.Compose:
    if train:
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )


def build_imagefolder_loaders(config: dict[str, Any]) -> tuple[DataLoader, DataLoader, int]:
    train_root = Path(str(deep_get(config, "data.train_images", ""))).expanduser()
    val_root = Path(str(deep_get(config, "data.val_images", ""))).expanduser()
    if not train_root.exists():
        raise FileNotFoundError(f"Training image root does not exist: {train_root}")
    if not val_root.exists():
        raise FileNotFoundError(f"Validation image root does not exist: {val_root}")

    train_dataset = datasets.ImageFolder(train_root, transform=build_image_transforms(train=True))
    val_dataset = datasets.ImageFolder(val_root, transform=build_image_transforms(train=False))
    if train_dataset.classes != val_dataset.classes:
        raise ValueError("Train and validation ImageFolder class lists differ.")

    batch_size = int(deep_get(config, "loader.batch_size", 16))
    num_workers = int(deep_get(config, "loader.num_workers", 0))
    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=bool(deep_get(config, "loader.shuffle", True)),
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader, len(train_dataset.classes)

