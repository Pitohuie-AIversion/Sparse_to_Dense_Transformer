from typing import List, Optional, Tuple, Any

import torch
from torch.utils.data import DataLoader, Dataset, Subset


class CustomSubset(Subset):
    def __init__(self, dataset, indices, transform=None):
        super().__init__(dataset, indices)
        self.transform = transform

    def __getitem__(self, idx):
        in_press, pressure, time_step = self.dataset[self.indices[idx]]
        if self.transform:
            in_press, pressure = self.transform((in_press, pressure))
        return in_press, pressure, time_step
from torchvision.transforms import Compose

from . import transforms
from .dataset import PressureDataset


def collate_fn(batch: List[Optional[Any]]) -> Optional[Any]:
    """Custom collate function to filter out None samples from a batch.

    Args:
        batch: A list of samples, where some might be None.

    Returns:
        A collated batch tensor or None if the batch is empty after filtering.
    """
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    return torch.utils.data.dataloader.default_collate(batch)


def get_loaders(
    data_path: str, batch_size: int, use_augmentation: bool = False, crop_size: Optional[Tuple[int, int]] = None
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Creates and returns data loaders for training, validation, and testing.

    Args:
        data_path: The path to the dataset file.
        batch_size: The number of samples per batch.
        use_augmentation: Whether to apply data augmentation to the training set.

    Returns:
        A tuple containing the training, validation, and test data loaders.
    """
    train_transform = None
    if use_augmentation:
        train_transform = Compose(
            [
                transforms.RandomFlip(),
                transforms.RandomRotate(),
                transforms.AddGaussianNoise(std=0.01),
                transforms.RandomCrop(crop_size if crop_size else (128, 128)),
            ]
        )

    full_dataset = PressureDataset(data_path)

    train_size = int(0.7 * len(full_dataset))
    valid_size = int(0.2 * len(full_dataset))
    test_size = len(full_dataset) - train_size - valid_size

    indices = torch.randperm(len(full_dataset)).tolist()

    train_indices = indices[:train_size]
    valid_indices = indices[train_size:train_size + valid_size]
    test_indices = indices[train_size + valid_size:]

    train_dataset = CustomSubset(full_dataset, train_indices, transform=train_transform)
    valid_dataset = CustomSubset(full_dataset, valid_indices, transform=None)
    test_dataset = CustomSubset(full_dataset, test_indices, transform=None)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )

    return train_loader, valid_loader, test_loader
