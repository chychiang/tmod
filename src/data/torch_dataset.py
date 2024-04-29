"""Utility functions for working with Torch datasets and dataloaders."""
from pathlib import Path

import fiftyone.utils.torch as fout
import torchvision
from torch.utils.data import DataLoader



def make_tmod_dataloader(image_paths: Path, sample_ids: list[int],
                         batch_size: int, num_workers: int= 4,
                         shuffle: bool= True) -> DataLoader:
    """Create a Torch dataloader for the given image paths."""
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2023, 0.1994, 0.2010]
    transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean, std),
        ]
    )
    dataset = fout.TorchImageDataset(
        image_paths, sample_ids=sample_ids, transform=transforms
    )
    return DataLoader(dataset, batch_size=batch_size, 
                      num_workers=num_workers, shuffle=shuffle)
