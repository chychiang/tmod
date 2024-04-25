"""
Utility functions to parse user tags from COCO annotation file and add them 
to the dataset.
"""
import json
from pathlib import Path

import fiftyone as fo

def parse_user_tags(json_fp: Path) -> dict:
    """
    Parse the user tags from the COCO annotation file and return a dictionary.
    """
    # Read the JSON file
    with open(json_fp, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Initialize a dictionary to store image IDs and corresponding user_tags
    image_user_tags_dict = {}

    # Iterate over the images
    for image in data['images']:
        image_id = image['file_name'].split('.')[2]
        user_tags = image.get('extra', {}).get('user_tags', [])
        image_user_tags_dict[image_id] = user_tags
    return image_user_tags_dict


def add_user_tags(dataset: fo.Dataset, tags: dict) -> fo.Dataset:
    """Add classification labels to the dataset based on user tags."""
    error_count = 0
    dataset_with_tags = dataset.clone()
    for sample in dataset_with_tags:
        unique_id = Path(sample.filepath).stem.split('.')[2]
        try:
            sample["ground_truth"] = fo.Classification(label=tags[unique_id][0])
            sample.save()
        except KeyError:
            error_count += 1
    # Print number of error if any
    if error_count > 0:
        print(f"Failed to add tags to {error_count} samples")
        return dataset_with_tags, False
    return dataset_with_tags, True
