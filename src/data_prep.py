
import os
from pathlib import Path
import shutil
from sklearn.model_selection import train_test_split

# Base path in Google Drive
BASE = Path('/content/drive/MyDrive/lung-cancer-detection-fastai')

ORIG_DATA = BASE / 'data/raw/LC25000'
TARGET_BASE = BASE / 'data/interim'
SPLIT_RATIOS = (0.7, 0.2, 0.1)  # train, valid, test

# New binary classes
LABEL_MAP = {
    'lung_n': 'benign',
    'lung_aca': 'malignant',
    'lung_scc': 'malignant'
}

def create_dirs():
    for split in ['train', 'valid', 'test']:
        for cls in ['benign', 'malignant']:
            (TARGET_BASE/split/cls).mkdir(parents=True, exist_ok=True)

def split_and_copy():
    grouped_files = {'benign': [], 'malignant': []}

    # Collect files and group by new label
    for orig_class, new_label in LABEL_MAP.items():
        files = list((ORIG_DATA/orig_class).glob("*.jpeg"))
        grouped_files[new_label].extend(files)

    for label, files in grouped_files.items():
        train_files, temp = train_test_split(files, test_size=1 - SPLIT_RATIOS[0], random_state=42)
        valid_files, test_files = train_test_split(
            temp, test_size=SPLIT_RATIOS[2] / (SPLIT_RATIOS[1] + SPLIT_RATIOS[2]), random_state=42
        )

        for f in train_files:
            shutil.copy(f, TARGET_BASE/'train'/label/f.name)
        for f in valid_files:
            shutil.copy(f, TARGET_BASE/'valid'/label/f.name)
        for f in test_files:
            shutil.copy(f, TARGET_BASE/'test'/label/f.name)

    print("✅ Binary data split complete: train/valid/test with benign/malignant.")

if __name__ == "__main__":
    create_dirs()
    split_and_copy()
