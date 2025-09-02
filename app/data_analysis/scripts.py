import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path
import json
import shutil
import os
import random

BASE_DIR = Path(__file__).resolve().parent.parent.parent
IMAGE_SRC_PATH = BASE_DIR / 'dataset' / 'color'
DATA_PATH = BASE_DIR / 'app' / 'data' / 'classes.json'
IMAGE_DEST_PATH = BASE_DIR / 'app' / 'data' / 'images'

def remove_zone_identifier_files(image_src_path=IMAGE_SRC_PATH):
    """
    Remove all files ending with ':Zone.Identifier' from the dataset directories.
    """
    for item in image_src_path.iterdir():
        for file in item.iterdir():
            if ":Zone.Identifier" in file.name:
                os.remove(file)

def generate_class_summary_json(image_src_path=IMAGE_SRC_PATH, data_path=DATA_PATH):
    """
    Generate a JSON file summarizing the dataset.
    Each entry contains the plant name, status, and image count for that class.
    """
    data = []
    for item in image_src_path.iterdir():
        name, status = item.name.split('___')
        data.append({
            "Plant": name,
            "Status": status,
            "Count": len(list(item.iterdir()))
        })
    data_path.write_text(json.dumps(data, indent=2))

def copy_sample_images_per_class(image_src_path=IMAGE_SRC_PATH, image_dest_path=IMAGE_DEST_PATH, samples_per_class=3):
    """
    For each class, copy a random sample of images (default 3) into a corresponding folder in the destination path.
    Skips files with ':Zone.Identifier' in the name and non-file entries.
    """
    for item in image_src_path.iterdir():
        dest_dir = image_dest_path / item.name
        dest_dir.mkdir(parents=True, exist_ok=True)
        # Collect all valid image files
        valid_files = [
            file for file in item.iterdir()
            if file.is_file() and ":Zone.Identifier" not in file.name
        ]
        # Randomly sample up to samples_per_class images
        sample_files = random.sample(valid_files, min(samples_per_class, len(valid_files)))
        for file in sample_files:
            try:
                shutil.copy(file, dest_dir / file.name)
            except Exception:
                continue

def split_dataset_train_test(
    image_src_path=IMAGE_SRC_PATH,
    base_dir=BASE_DIR,
    test_ratio=0.3
):
    """
    Split the dataset into training and testing sets.
    Creates 'train' and 'test' folders in the dataset directory, with subfolders for each class.
    Randomly assigns 30% of images to 'test' and 70% to 'train' for each class.
    """
    image_train_path = base_dir / 'dataset' / 'train'
    image_test_path = base_dir / 'dataset' / 'test'

    # Create train and test directories
    image_train_path.mkdir(parents=True, exist_ok=True)
    image_test_path.mkdir(parents=True, exist_ok=True)

    for class_dir in image_src_path.iterdir():
        if not class_dir.is_dir():
            continue

        # Create class subdirectories in train and test
        train_class_dir = image_train_path / class_dir.name
        test_class_dir = image_test_path / class_dir.name
        train_class_dir.mkdir(parents=True, exist_ok=True)
        test_class_dir.mkdir(parents=True, exist_ok=True)

        # List all valid image files
        image_files = [
            file for file in class_dir.iterdir()
            if file.is_file() and ":Zone.Identifier" not in file.name
        ]

        # Shuffle and split
        random.shuffle(image_files)
        split_idx = int(len(image_files) * (1 - test_ratio))
        train_files = image_files[:split_idx]
        test_files = image_files[split_idx:]

        # Copy files
        for file in train_files:
            shutil.copy(file, train_class_dir / file.name)
        for file in test_files:
            shutil.copy(file, test_class_dir / file.name)

# Example usage:
split_dataset_train_test()
# remove_zone_identifier_files()
# generate_class_summary_json()
# copy_sample_images_per_class()