from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset
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


def build_loaders(config: dict[str, Any]) -> tuple[DataLoader, DataLoader, int]:
    dataset_name = str(deep_get(config, "data.name", deep_get(config, "data.dataset", "imagefolder"))).lower()
    if dataset_name in {"imagefolder", "folder"}:
        return build_imagefolder_loaders(config)
    if dataset_name in {"cifar10", "cifar-10"}:
        return build_cifar_loaders(config, datasets.CIFAR10, "CIFAR-10")
    if dataset_name in {"cifar100", "cifar-100"}:
        return build_cifar_loaders(config, datasets.CIFAR100, "CIFAR-100")
    raise ValueError(f"Unsupported dataset: {dataset_name}")


def build_imagefolder_loaders(config: dict[str, Any]) -> tuple[DataLoader, DataLoader, int]:
    train_root = Path(str(deep_get(config, "data.train_images", ""))).expanduser()
    val_root = Path(str(deep_get(config, "data.val_images", ""))).expanduser()
    if not train_root.exists():
        raise FileNotFoundError(f"Training image root does not exist: {train_root}")
    if not val_root.exists():
        raise FileNotFoundError(f"Validation image root does not exist: {val_root}")

    train_dataset = datasets.ImageFolder(train_root, transform=build_image_transforms(train=True))
    val_dataset = datasets.ImageFolder(val_root, transform=build_image_transforms(train=False))
    return build_dataset_loaders(train_dataset, val_dataset, config)


def build_cifar_loaders(
    config: dict[str, Any],
    dataset_class: type[datasets.CIFAR10] | type[datasets.CIFAR100],
    dataset_label: str,
) -> tuple[DataLoader, DataLoader, int]:
    root_value = str(deep_get(config, "data.root", "")).strip()
    if not root_value:
        raise ValueError(f"data.root is required when data.name or data.dataset is {dataset_label}.")

    root = Path(root_value).expanduser()
    download = bool(deep_get(config, "data.download", False))
    if not root.exists() and not download:
        raise FileNotFoundError(f"{dataset_label} root does not exist: {root}")

    train_dataset = dataset_class(
        root=str(root),
        train=True,
        download=download,
        transform=build_image_transforms(train=True),
    )
    val_dataset = dataset_class(
        root=str(root),
        train=False,
        download=download,
        transform=build_image_transforms(train=False),
    )
    return build_dataset_loaders(train_dataset, val_dataset, config)


def build_dataset_loaders(
    train_dataset: Dataset,
    val_dataset: Dataset,
    config: dict[str, Any],
) -> tuple[DataLoader, DataLoader, int]:
    train_classes = getattr(train_dataset, "classes", None)
    val_classes = getattr(val_dataset, "classes", None)
    if train_classes is None or val_classes is None:
        raise ValueError("Classification datasets must expose a classes attribute.")
    if train_classes != val_classes:
        raise ValueError("Train and validation class lists differ.")

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
    return train_loader, val_loader, len(train_classes)
