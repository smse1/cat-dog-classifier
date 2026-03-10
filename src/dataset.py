from pathlib import Path
from typing import Tuple

from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def get_transforms(image_size: int = 224) -> Tuple[transforms.Compose, transforms.Compose]:
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    return train_transform, val_transform


def check_dataset_structure(data_root: str = "data") -> None:
    root = Path(data_root)
    expected_dirs = [
        root / "train" / "cat",
        root / "train" / "dog",
        root / "val" / "cat",
        root / "val" / "dog",
        root / "test",
    ]

    print("Checking dataset structure...")
    for directory in expected_dirs:
        status = "exists" if directory.exists() else "missing"
        print(f"{directory}: {status}")


def get_dataloaders(
    data_root: str = "data",
    image_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, datasets.ImageFolder, datasets.ImageFolder]:
    train_transform, val_transform = get_transforms(image_size=image_size)

    train_dir = Path(data_root) / "train"
    val_dir = Path(data_root) / "val"

    train_dataset = datasets.ImageFolder(root=str(train_dir), transform=train_transform)
    val_dataset = datasets.ImageFolder(root=str(val_dir), transform=val_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, val_loader, train_dataset, val_dataset