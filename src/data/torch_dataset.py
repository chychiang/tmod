"""Utility functions for working with Torch datasets and dataloaders."""
from pathlib import Path

import fiftyone.utils.torch as fout
import torchvision
from torch.utils.data import DataLoader
import torch



def make_tmod_dataloader(image_paths: Path, sample_ids: list[int],
                         batch_size: int, num_workers: int= 4,
                         shuffle: bool= True) -> DataLoader:
    """Create a Torch dataloader for the given image paths."""
    # mean = [0.485, 0.456, 0.406]
    # std = [0.229, 0.224, 0.225]

    # mean = [5.7129e-08, 4.7380e-08, 1.2681e-07]
    # std = [5.5260e-07, 5.4554e-07, 5.4854e-07]
    transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Resize((224, 224)),
            # torchvision.transforms.Normalize(mean, std),
        ]
    )
    dataset = fout.TorchImageDataset(
        image_paths, sample_ids=sample_ids, transform=transforms
    )
    return DataLoader(dataset, batch_size=batch_size, 
                      num_workers=num_workers, shuffle=shuffle), transforms


def get_mean_std(loader) -> tuple[torch.Tensor, torch.Tensor, float, float]:
    # Compute the mean and standard deviation per channel of all pixels in the dataset
    num_pixels = 0
    mean = torch.zeros(3)
    std = torch.zeros(3)
    min_val, max_val = float("inf"), float("-inf")
    for images, _ in loader:
        batch_size, num_channels, height, width = images.shape
        num_pixels += batch_size * height * width
        mean += images.mean(dim=(0, 2, 3))
        std += images.std(dim=(0, 2, 3))
        min_val = min(min_val, images.min())
        max_val = max(max_val, images.max())

    mean /= num_pixels
    std /= num_pixels

    return mean, std, min_val, max_val
