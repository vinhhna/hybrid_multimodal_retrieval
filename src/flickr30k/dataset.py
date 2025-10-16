"""
Dataset module for Flickr30K dataset.

This module provides classes and functions for loading and processing
the Flickr30K image-caption dataset.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from PIL import Image
import warnings


class Flickr30KDataset:
    """
    Flickr30K Dataset handler.
    
    This class provides methods to load, process, and access the Flickr30K dataset,
    which contains 31,000 images with 5 captions each.
    
    Attributes:
        images_dir (Path): Directory containing image files
        captions_file (Path): Path to the CSV file with captions
        df (pd.DataFrame): DataFrame containing all captions
        num_images (int): Number of unique images
        num_captions (int): Total number of captions
    """
    
    def __init__(
        self,
        images_dir: str = "data/images",
        captions_file: str = "data/results.csv",
        auto_load: bool = True
    ):
        """
        Initialize the Flickr30K dataset.
        
        Args:
            images_dir: Directory containing the image files
            captions_file: Path to the CSV file with captions
            auto_load: Whether to automatically load the captions on initialization
        """
        self.images_dir = Path(images_dir)
        self.captions_file = Path(captions_file)
        self.df: Optional[pd.DataFrame] = None
        self.num_images: int = 0
        self.num_captions: int = 0
        
        # Verify paths exist
        if not self.captions_file.exists():
            warnings.warn(f"Captions file not found: {self.captions_file}")
        
        if not self.images_dir.exists():
            warnings.warn(f"Images directory not found: {self.images_dir}")
        
        # Auto-load if specified
        if auto_load and self.captions_file.exists():
            self.load_captions()
    
    def load_captions(self) -> pd.DataFrame:
        """
        Load and preprocess the captions CSV file.
        
        The CSV file uses pipe delimiter with spaces (' | ').
        Column names are standardized to: image_name, comment_number, caption
        
        Returns:
            DataFrame with processed captions
        """
        print(f"Loading captions from: {self.captions_file}")
        
        # Load CSV with pipe delimiter
        self.df = pd.read_csv(self.captions_file, sep='|', engine='python')
        
        # Clean whitespace from column names and values
        self.df.columns = self.df.columns.str.strip()
        self.df = self.df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
        
        # Standardize column names
        self.df.columns = ['image_name', 'comment_number', 'caption']
        
        # Calculate statistics
        self.num_images = self.df['image_name'].nunique()
        self.num_captions = len(self.df)
        
        print(f"✓ Loaded {self.num_captions:,} captions for {self.num_images:,} images")
        
        return self.df
    
    def get_captions(self, image_name: str) -> List[str]:
        """
        Get all captions for a specific image.
        
        Args:
            image_name: Name of the image file (e.g., '1000092795.jpg')
            
        Returns:
            List of caption strings for the image
        """
        if self.df is None:
            raise ValueError("Dataset not loaded. Call load_captions() first.")
        
        captions = self.df[self.df['image_name'] == image_name]['caption'].tolist()
        return captions
    
    def get_image(self, image_name: str) -> Optional[Image.Image]:
        """
        Load an image from the dataset.
        
        Args:
            image_name: Name of the image file
            
        Returns:
            PIL Image object, or None if image not found
        """
        image_path = self.images_dir / image_name
        
        if not image_path.exists():
            warnings.warn(f"Image not found: {image_path}")
            return None
        
        return Image.open(image_path)
    
    def get_random_sample(self, seed: Optional[int] = None) -> Tuple[str, List[str]]:
        """
        Get a random image with its captions.
        
        Args:
            seed: Random seed for reproducibility
            
        Returns:
            Tuple of (image_name, list of captions)
        """
        if self.df is None:
            raise ValueError("Dataset not loaded. Call load_captions() first.")
        
        if seed is not None:
            np.random.seed(seed)
        
        unique_images = self.df['image_name'].unique()
        image_name = np.random.choice(unique_images)
        captions = self.get_captions(image_name)
        
        return image_name, captions
    
    def get_statistics(self) -> Dict[str, any]:
        """
        Get statistical information about the dataset.
        
        Returns:
            Dictionary containing various statistics
        """
        if self.df is None:
            raise ValueError("Dataset not loaded. Call load_captions() first.")
        
        # Calculate caption lengths
        caption_lengths = self.df['caption'].str.len()
        caption_word_counts = self.df['caption'].str.split().str.len()
        captions_per_image = self.df.groupby('image_name').size()
        
        stats = {
            'num_images': self.num_images,
            'num_captions': self.num_captions,
            'avg_captions_per_image': captions_per_image.mean(),
            'min_captions_per_image': captions_per_image.min(),
            'max_captions_per_image': captions_per_image.max(),
            'avg_caption_length': caption_lengths.mean(),
            'avg_caption_words': caption_word_counts.mean(),
            'min_caption_words': caption_word_counts.min(),
            'max_caption_words': caption_word_counts.max(),
        }
        
        return stats
    
    def search_captions(
        self,
        keyword: str,
        case_sensitive: bool = False,
        max_results: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Search for captions containing a keyword.
        
        Args:
            keyword: The keyword to search for
            case_sensitive: Whether the search should be case-sensitive
            max_results: Maximum number of results to return
            
        Returns:
            DataFrame with matching captions
        """
        if self.df is None:
            raise ValueError("Dataset not loaded. Call load_captions() first.")
        
        if case_sensitive:
            mask = self.df['caption'].str.contains(keyword, na=False)
        else:
            mask = self.df['caption'].str.contains(keyword, case=False, na=False)
        
        results = self.df[mask]
        
        if max_results is not None:
            results = results.head(max_results)
        
        return results
    
    def get_unique_images(self) -> List[str]:
        """
        Get list of all unique image names.
        
        Returns:
            List of image filenames
        """
        if self.df is None:
            raise ValueError("Dataset not loaded. Call load_captions() first.")
        
        return self.df['image_name'].unique().tolist()
    
    def __len__(self) -> int:
        """Return the number of images in the dataset."""
        return self.num_images
    
    def __repr__(self) -> str:
        """String representation of the dataset."""
        if self.df is None:
            return f"Flickr30KDataset(not loaded)"
        return f"Flickr30KDataset({self.num_images:,} images, {self.num_captions:,} captions)"
