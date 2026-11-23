"""
Kaggle Dataset Acquisition Script for:
"No-Fire and Fire Detection in UAV collected images with LightWeight Deep-Learning"

This script provides utilities to download and prepare forest fire and no-fire UAV datasets:
1. FLAME: Aerial Fire Dataset (Kaggle)
2. Forest Fire Dataset (Kaggle)
3. Kaggle Wildfire Prediction / Aerial UAV images
"""

import os
import sys
import shutil
import urllib.request
import zipfile

DATASET_ROOT = os.path.join(os.path.dirname(__file__), "static", "dataset")
FIRE_DIR = os.path.join(DATASET_ROOT, "fire")
NOFIRE_DIR = os.path.join(DATASET_ROOT, "no_fire")

KAGGLE_DATASETS_INFO = [
    {
        "name": "FLAME Aerial Fire Dataset",
        "kaggle_ref": "deepseas/flame-dataset",
        "description": "Aerial drone images capturing forest fire and non-fire terrain with thermal and RGB visual spectrum.",
        "url": "https://www.kaggle.com/datasets/deepseas/flame-dataset"
    },
    {
        "name": "Forest Fire Detection Dataset (UAV & Aerial)",
        "kaggle_ref": "phylake1337/fire-dataset",
        "description": "Standardized Fire and No-Fire images curated for CNN and lightweight DL classification.",
        "url": "https://www.kaggle.com/datasets/phylake1337/fire-dataset"
    },
    {
        "name": "FLAME 3 Computer Vision Wildfire Subset",
        "kaggle_ref": "rosebrendat/flame3-cv-subset",
        "description": "Drone aerial imagery specifically prepared for lightweight model evaluation and cross-validation.",
        "url": "https://www.kaggle.com/datasets/rosebrendat/flame3-cv-subset"
    }
]

def print_instructions():
    print("=" * 70)
    print("   UAV FOREST FIRE DATASET ACQUISITION (KAGGLE)")
    print("=" * 70)
    print("\nTarget Dataset Configuration (from Research Framework):")
    print(" - Total Images: 2,400 UAV Aerial Images")
    print(" - Class Distribution: 1,200 Fire Images | 1,200 No-Fire Images")
    print(" - Image Resolution: Standardized to 224x224 pixels")
    print(" - Validation Strategy: 3-Fold Cross-Validation\n")
    print("Recommended Kaggle Datasets:")
    for idx, ds in enumerate(KAGGLE_DATASETS_INFO, 1):
        print(f" {idx}. {ds['name']}")
        print(f"    Kaggle: {ds['kaggle_ref']} ({ds['url']})")
        print(f"    Info:   {ds['description']}\n")
    
    print("-" * 70)
    print("METHOD 1: Using the Kaggle CLI (if configured with kaggle.json):")
    print("  pip install kaggle")
    print("  kaggle datasets download -d deepseas/flame-dataset -p ./temp_dataset --unzip")
    print("\nMETHOD 2: Using Python kagglehub:")
    print("  pip install kagglehub")
    print("  python -c \"import kagglehub; path = kagglehub.dataset_download('deepseas/flame-dataset'); print(path)\"")
    print("\nMETHOD 3: Web Download via Kaggle UI:")
    print("  1. Visit: https://www.kaggle.com/datasets/deepseas/flame-dataset")
    print("  2. Click 'Download' (ZIP)")
    print("  3. Extract 1,200 Fire images into: static/dataset/fire/")
    print("  4. Extract 1,200 No-Fire images into: static/dataset/no_fire/")
    print("=" * 70)

if __name__ == "__main__":
    print_instructions()
