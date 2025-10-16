"""
Visualization module for Flickr30K dataset.

This module provides functions for displaying images, captions,
and creating various visualizations for the dataset.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, List, Tuple, TYPE_CHECKING
from PIL import Image

# Import dataset class for type hints
if TYPE_CHECKING:
    from .dataset import Flickr30KDataset


def display_image_with_captions(
    image_name: str,
    dataset: Optional["Flickr30KDataset"] = None,
    images_dir: Optional[str] = None,
    captions: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (10, 8)
) -> None:
    """
    Display an image with all its captions.
    
    Args:
        image_name: Name of the image file
        dataset: Flickr30KDataset instance (if available)
        images_dir: Directory containing images (used if dataset not provided)
        captions: List of captions (if not using dataset)
        figsize: Figure size tuple (width, height)
    """
    # Load image
    if dataset is not None:
        img = dataset.get_image(image_name)
        if captions is None:
            captions = dataset.get_captions(image_name)
    else:
        if images_dir is None:
            images_dir = "data/images"
        img_path = Path(images_dir) / image_name
        if not img_path.exists():
            print(f"Error: Image not found at {img_path}")
            return
        img = Image.open(img_path)
    
    if img is None:
        print(f"Error: Could not load image {image_name}")
        return
    
    # Display image
    plt.figure(figsize=figsize)
    plt.imshow(img)
    plt.axis('off')
    plt.title(f"Image: {image_name}", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    # Print captions
    if captions:
        print(f"\nCaptions for {image_name}:")
        for i, caption in enumerate(captions, 1):
            print(f"{i}. {caption}")
    else:
        print(f"\nNo captions available for {image_name}")


def display_random_samples(
    dataset: "Flickr30KDataset",
    n_samples: int = 3,
    seed: Optional[int] = 42,
    figsize: Tuple[int, int] = (10, 8)
) -> None:
    """
    Display random sample images with their captions.
    
    Args:
        dataset: Flickr30KDataset instance
        n_samples: Number of samples to display
        seed: Random seed for reproducibility
        figsize: Figure size for each image
    """
    if seed is not None:
        np.random.seed(seed)
    
    unique_images = dataset.get_unique_images()
    
    # Check if images are available
    sample_img = dataset.get_image(unique_images[0])
    has_images = sample_img is not None
    
    if not has_images:
        print("⚠️  Images not found in directory. Showing captions only.")
        print(f"Download images to: {dataset.images_dir}\n")
    
    # Sample random images
    sample_images = np.random.choice(
        unique_images,
        size=min(n_samples, len(unique_images)),
        replace=False
    )
    
    for image_name in sample_images:
        if has_images:
            display_image_with_captions(
                image_name=image_name,
                dataset=dataset,
                figsize=figsize
            )
        else:
            # Show captions only
            captions = dataset.get_captions(image_name)
            print(f"\n{'='*80}")
            print(f"Image: {image_name}")
            for i, caption in enumerate(captions, 1):
                print(f"  {i}. {caption}")
        
        print()  # Add spacing between samples


def plot_caption_statistics(
    dataset: "Flickr30KDataset",
    figsize: Tuple[int, int] = (15, 5)
) -> None:
    """
    Plot statistical visualizations of caption properties.
    
    Args:
        dataset: Flickr30KDataset instance
        figsize: Figure size tuple
    """
    if dataset.df is None:
        print("Error: Dataset not loaded")
        return
    
    df = dataset.df
    
    # Calculate statistics
    df['caption_length'] = df['caption'].str.len()
    df['caption_word_count'] = df['caption'].str.split().str.len()
    
    # Create subplots
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Caption word count distribution
    axes[0].hist(df['caption_word_count'], bins=50, edgecolor='black', alpha=0.7)
    axes[0].set_xlabel('Number of Words', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title('Distribution of Caption Word Counts', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # Caption character length distribution
    axes[1].hist(df['caption_length'], bins=50, edgecolor='black', alpha=0.7, color='orange')
    axes[1].set_xlabel('Number of Characters', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)
    axes[1].set_title('Distribution of Caption Lengths', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def print_dataset_statistics(dataset: "Flickr30KDataset") -> None:
    """
    Print comprehensive statistics about the dataset.
    
    Args:
        dataset: Flickr30KDataset instance
    """
    stats = dataset.get_statistics()
    
    print("=" * 60)
    print("DATASET STATISTICS")
    print("=" * 60)
    print(f"Total unique images:          {stats['num_images']:>12,}")
    print(f"Total captions:               {stats['num_captions']:>12,}")
    print(f"Average captions per image:   {stats['avg_captions_per_image']:>12.2f}")
    print(f"Min captions per image:       {stats['min_captions_per_image']:>12}")
    print(f"Max captions per image:       {stats['max_captions_per_image']:>12}")
    print()
    print(f"Average caption length:       {stats['avg_caption_length']:>12.2f} chars")
    print(f"Average words per caption:    {stats['avg_caption_words']:>12.2f}")
    print(f"Min words per caption:        {stats['min_caption_words']:>12}")
    print(f"Max words per caption:        {stats['max_caption_words']:>12}")
    print("=" * 60)


def display_search_results(
    results_df,
    keyword: str,
    max_display: int = 5,
    show_images: bool = False,
    dataset: Optional["Flickr30KDataset"] = None
) -> None:
    """
    Display search results in a formatted way.
    
    Args:
        results_df: DataFrame with search results
        keyword: The search keyword
        max_display: Maximum number of results to display
        show_images: Whether to show images (requires dataset)
        dataset: Flickr30KDataset instance (needed if show_images=True)
    """
    print(f"Found {len(results_df)} captions containing '{keyword}'")
    print(f"\nShowing first {min(max_display, len(results_df))} results:\n")
    
    for idx, (_, row) in enumerate(results_df.head(max_display).iterrows(), 1):
        print(f"{idx}. Image: {row['image_name']}")
        print(f"   Caption: {row['caption']}\n")
        
        if show_images and dataset is not None:
            img = dataset.get_image(row['image_name'])
            if img is not None:
                plt.figure(figsize=(6, 4))
                plt.imshow(img)
                plt.axis('off')
                plt.title(f"{row['image_name']}", fontsize=10)
                plt.tight_layout()
                plt.show()
