from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from src.dataset import check_dataset_structure, get_dataloaders
from src.model import get_model


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()

    running_loss = 0.0
    running_correct = 0
    total_samples = 0

    progress_bar = tqdm(loader, desc="Training", leave=False)

    for images, labels in progress_bar:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        running_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)

        progress_bar.set_postfix({
            "loss": f"{loss.item():.4f}"
        })

    epoch_loss = running_loss / total_samples
    epoch_acc = running_correct / total_samples

    return epoch_loss, epoch_acc


def validate(model, loader, criterion, device):
    model.eval()

    running_loss = 0.0
    running_correct = 0
    total_samples = 0

    with torch.no_grad():
        progress_bar = tqdm(loader, desc="Validation", leave=False)

        for images, labels in progress_bar:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            running_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)

    epoch_loss = running_loss / total_samples
    epoch_acc = running_correct / total_samples

    return epoch_loss, epoch_acc


def main():
    # ====== basic config ======
    data_root = "data"
    image_size = 224
    batch_size = 16
    num_workers = 0
    learning_rate = 1e-4
    num_epochs = 1
    use_pretrained = True

    # ====== check environment ======
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ====== check dataset ======
    check_dataset_structure(data_root=data_root)

    # ====== dataloader ======
    train_loader, val_loader, train_dataset, val_dataset = get_dataloaders(
        data_root=data_root,
        image_size=image_size,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Class mapping: {train_dataset.class_to_idx}")

    # ====== model ======
    model = get_model(num_classes=2, use_pretrained=use_pretrained)
    model = model.to(device)

    # ====== loss and optimizer ======
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # ====== checkpoint folder ======
    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    best_val_acc = 0.0

    # ====== training loop ======
    for epoch in range(num_epochs):
        print(f"\nEpoch [{epoch + 1}/{num_epochs}]")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        val_loss, val_acc = validate(
            model, val_loader, criterion, device
        )

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint_path = checkpoint_dir / "best_model.pth"
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Best model saved to: {checkpoint_path}")

    print(f"\nTraining finished. Best validation accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()