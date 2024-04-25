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


def main():
    """Main function for testing the dataset module."""    
    datasets = []
    for split in ["train", "valid", "test"]:
        dataset_root = Path(f"data/{split}")
        annotation_fp = dataset_root / "_annotations.coco.json"
        dataset = get_dataset(dataset_root, annotation_fp)
        datasets.append(dataset)

    combined_dataset = merge_datasets(datasets)
    for sample in combined_dataset:
        print("=" * 80)
        print(sample.id)
        print(sample.filepath)


if __name__ == "__main__":
    main()
