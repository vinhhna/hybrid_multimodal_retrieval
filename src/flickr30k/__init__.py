"""
Flickr30K Dataset Package

A modular package for loading, processing, and visualizing the Flickr30K dataset
for hybrid multimodal retrieval tasks.
"""

__version__ = "0.1.0"
__author__ = "Your Name"

from .dataset import Flickr30KDataset
from .utils import load_config, get_project_root
from .visualization import display_image_with_captions, display_random_samples

__all__ = [
    'Flickr30KDataset',
    'load_config',
    'get_project_root',
    'display_image_with_captions',
    'display_random_samples',
]
