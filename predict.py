from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from src.model import get_model


class TestImageDataset(Dataset):
    def __init__(self, test_dir: str = "data/test", image_size: int = 224):
        self.test_dir = Path(test_dir)

        self.image_paths = sorted(
            [
                p for p in self.test_dir.iterdir()
                if p.is_file() and p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
            ],
            key=self._sort_key
        )

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

    @staticmethod
    def _sort_key(path: Path):
        stem = path.stem
        return int(stem) if stem.isdigit() else stem

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        return image, image_path.stem


def main():
    test_dir = "data/test"
    checkpoint_path = "checkpoints/best_model.pth"
    submission_path = "submission.csv"
    image_size = 224
    batch_size = 32
    num_workers = 0
    use_pretrained = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    test_dir_path = Path(test_dir)
    checkpoint_file = Path(checkpoint_path)

    if not test_dir_path.exists():
        raise FileNotFoundError(f"Test directory not found: {test_dir}")

    if not checkpoint_file.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    test_dataset = TestImageDataset(test_dir=test_dir, image_size=image_size)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    print(f"Test samples: {len(test_dataset)}")

    model = get_model(num_classes=2, use_pretrained=use_pretrained)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model = model.to(device)
    model.eval()

    ids = []
    labels = []

    with torch.no_grad():
        for images, batch_ids in test_loader:
            images = images.to(device, non_blocking=True)

            outputs = model(images)
            preds = torch.argmax(outputs, dim=1).cpu().tolist()

            # 你的训练映射是 {'cat': 0, 'dog': 1}
            ids.extend([int(x) if str(x).isdigit() else x for x in batch_ids])
            labels.extend(preds)

    submission_df = pd.DataFrame({
        "id": ids,
        "label": labels,
    })

    submission_df = submission_df.sort_values(by="id").reset_index(drop=True)
    submission_df.to_csv(submission_path, index=False)
    print(f"Submission saved to: {submission_path}")
    print(submission_df.head())


if __name__ == "__main__":
    main()