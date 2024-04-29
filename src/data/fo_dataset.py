"""Module to load and merge COCO datasets."""

from pathlib import Path

import fiftyone as fo


def get_dataset(dataset_root: Path, annotation_fp: Path):
    """Load a COCO dataset from the given directory and annotation file."""
    dataset = fo.Dataset.from_dir(
        dataset_type=fo.types.COCODetectionDataset,
        data_path=dataset_root,
        labels_path=annotation_fp,
        include_id=True,
    )
    return dataset


def merge_datasets(dataset_list: list[fo.Dataset]) -> fo.Dataset:
    """
    Combine multiple datasets into a single dataset without changing any
    content.
    """
    combined_dataset = fo.Dataset()  # Create an empty dataset to merge into
    for dataset in dataset_list:
        combined_dataset.merge_samples(dataset)
    return combined_dataset


def load_all_splits(data_dir: Path, 
                    subdir_names: list[str]) -> list[fo.Dataset]:
    """Load all splits of a COCO dataset from the given parent directory."""
    datasets = []
    for split in subdir_names:
        dataset_root = data_dir / split
        annotation_fp = dataset_root / "_annotations.coco.json"
        dataset = get_dataset(dataset_root, annotation_fp)
        datasets.append(dataset)
    return datasets


def load_all_splits_as_one(data_dir: Path,
                           subdir_names: list[str]) -> fo.Dataset:
    """
    Load all splits of a COCO dataset from the given parent directory and 
    merge them into one.
    """
    datasets = load_all_splits(data_dir, subdir_names)
    return merge_datasets(datasets)

def main():
    """Main function for testing the dataset module."""    
    dataset_root = Path("data")
    dataset = load_all_splits_as_one(dataset_root, ["train", "valid", "test"])
    print(f"Loaded dataset with {len(dataset)} samples.")

if __name__ == "__main__":
    main()
