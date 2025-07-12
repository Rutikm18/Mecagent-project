"""
Dataset loader for the CadQuery code generation project.
This module handles loading and preprocessing the 147K image-code pairs.
"""

import sys
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from utils.config import DATASET_NAME, DATASET_CACHE_DIR, NUM_PROC

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from datasets import load_dataset, Dataset, DatasetDict
    import numpy as np
    from PIL import Image
except ImportError as e:
    logger.error(f"Missing required packages: {e}")
    logger.info("Please install required packages: pip install datasets pillow numpy")
    raise

class CadQueryDatasetLoader:
    """
    A class to handle loading and preprocessing the CadQuery dataset.
    
    This dataset contains 147K pairs of images and corresponding CadQuery code.
    """
    
    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Initialize the dataset loader.
        
        Args:
            cache_dir: Directory to cache the dataset (optional)
        """
        self.cache_dir = cache_dir or DATASET_CACHE_DIR
        self.dataset = None
        self.train_data = None
        self.test_data = None
        
        # Create cache directory if it doesn't exist
        self.cache_dir.mkdir(exist_ok=True)
        
        logger.info(f"Dataset loader initialized with cache dir: {self.cache_dir}")
    
    def load_dataset(self, split: Optional[str] = None) -> DatasetDict:
        """
        Load the CadQuery dataset from HuggingFace.
        
        Args:
            split: Dataset split to load ('train', 'test', or None for both)
            
        Returns:
            DatasetDict containing the loaded dataset
        """
        try:
            logger.info(f"Loading dataset: {DATASET_NAME}")
            
            if split:
                # Load specific split
                self.dataset = load_dataset(
                    DATASET_NAME,
                    split=split,
                    cache_dir=str(self.cache_dir),
                    num_proc=NUM_PROC
                )
                logger.info(f"Loaded {split} split with {len(self.dataset)} samples")
            else:
                # Load both train and test splits
                self.dataset = load_dataset(
                    DATASET_NAME,
                    split=["train", "test"],
                    cache_dir=str(self.cache_dir),
                    num_proc=NUM_PROC
                )
                logger.info(f"Loaded dataset with {len(self.dataset['train'])} train and {len(self.dataset['test'])} test samples")
            
            return self.dataset
            
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            raise
    
    def get_sample_data(self, num_samples: int = 5) -> List[Dict]:
        """
        Get a small sample of the dataset for testing.
        
        Args:
            num_samples: Number of samples to return
            
        Returns:
            List of sample data
        """
        if self.dataset is None:
            self.load_dataset()
        
        if isinstance(self.dataset, DatasetDict):
            # Get samples from train split
            samples = self.dataset['train'].select(range(min(num_samples, len(self.dataset['train']))))
        else:
            # Single dataset
            samples = self.dataset.select(range(min(num_samples, len(self.dataset))))
        
        return samples
    
    def explore_dataset(self) -> Dict:
        """
        Explore the dataset structure and provide basic statistics.
        
        Returns:
            Dictionary with dataset information
        """
        if self.dataset is None:
            self.load_dataset()
        
        info = {}
        
        if isinstance(self.dataset, DatasetDict):
            info['splits'] = list(self.dataset.keys())
            for split_name, split_data in self.dataset.items():
                info[f'{split_name}_size'] = len(split_data)
                info[f'{split_name}_features'] = list(split_data.features.keys())
        else:
            info['size'] = len(self.dataset)
            info['features'] = list(self.dataset.features.keys())
        
        logger.info("Dataset exploration completed")
        return info
    
    def preprocess_sample(self, sample: Dict) -> Dict:
        """
        Preprocess a single sample from the dataset.
        
        Args:
            sample: Raw sample from the dataset
            
        Returns:
            Preprocessed sample
        """
        # This is a basic preprocessing function
        # You can extend this based on your model's needs
        
        processed_sample = {
            'image': sample.get('image', None),
            'code': sample.get('code', ''),
            'id': sample.get('id', 'unknown')
        }
        
        # Add any additional preprocessing here
        # For example:
        # - Resize images
        # - Tokenize code
        # - Normalize data
        
        return processed_sample
    
    def get_data_for_training(self, split: str = 'train', max_samples: Optional[int] = None) -> List[Dict]:
        """
        Get preprocessed data ready for training.
        
        Args:
            split: Which split to use ('train' or 'test')
            max_samples: Maximum number of samples to return (for testing)
            
        Returns:
            List of preprocessed samples
        """
        if self.dataset is None:
            self.load_dataset()
        
        # Get the appropriate split
        if isinstance(self.dataset, DatasetDict):
            data = self.dataset[split]
        else:
            data = self.dataset
        
        # Limit samples if specified
        if max_samples:
            data = data.select(range(min(max_samples, len(data))))
        
        # Preprocess all samples
        processed_data = []
        for i, sample in enumerate(data):
            processed_sample = self.preprocess_sample(sample)
            processed_data.append(processed_sample)
            
            if i % 1000 == 0:
                logger.info(f"Preprocessed {i} samples...")
        
        logger.info(f"Preprocessing completed. Total samples: {len(processed_data)}")
        return processed_data

def main():
    """Test the dataset loader."""
    print("Testing CadQuery Dataset Loader")
    print("=" * 40)
    
    # Initialize loader
    loader = CadQueryDatasetLoader()
    
    # Explore dataset
    info = loader.explore_dataset()
    print(f"Dataset info: {info}")
    
    # Get sample data
    samples = loader.get_sample_data(num_samples=3)
    print(f"\nSample data structure:")
    for i, sample in enumerate(samples):
        print(f"Sample {i+1}:")
        print(f"  Keys: {list(sample.keys())}")
        print(f"  Code length: {len(sample.get('code', ''))}")
        if 'image' in sample:
            print(f"  Image type: {type(sample['image'])}")
        print()

if __name__ == "__main__":
    main() 