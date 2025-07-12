"""
Configuration settings for the CadQuery Code Generator project.
This file contains all the important settings and parameters.
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
TRAINING_DIR = PROJECT_ROOT / "training"
EVALUATION_DIR = PROJECT_ROOT / "evaluation"
UTILS_DIR = PROJECT_ROOT / "utils"

# Dataset settings
DATASET_NAME = "CADCODER/GenCAD-Code"
DATASET_CACHE_DIR = PROJECT_ROOT / "cache"  # Local cache for dataset
NUM_PROC = 4  # Number of processes for data loading (reduce if you have less RAM)

# Model settings
BASELINE_MODEL_NAME = "baseline_cadquery_generator"
ENHANCED_MODEL_NAME = "enhanced_cadquery_generator"

# Training settings
BATCH_SIZE = 32  # Start small, increase if you have more memory
LEARNING_RATE = 1e-4  # Standard learning rate for most models
NUM_EPOCHS = 10  # Start with fewer epochs for testing
VALIDATION_SPLIT = 0.2  # 20% of data for validation

# Evaluation settings
IOU_PITCH = 0.05  # Voxel pitch for IOU calculation
MAX_EVAL_SAMPLES = 1000  # Limit evaluation samples for faster testing

# Hardware settings
USE_GPU = False  # Set to True if you have a GPU
DEVICE = "cuda" if USE_GPU else "cpu"

# Logging settings
LOG_LEVEL = "INFO"
SAVE_MODELS = True
SAVE_PLOTS = True

# Create directories if they don't exist
for directory in [DATA_DIR, MODELS_DIR, TRAINING_DIR, EVALUATION_DIR, DATASET_CACHE_DIR]:
    directory.mkdir(exist_ok=True)

print(f"Project root: {PROJECT_ROOT}")
print(f"Using device: {DEVICE}")
print(f"Dataset cache: {DATASET_CACHE_DIR}") 