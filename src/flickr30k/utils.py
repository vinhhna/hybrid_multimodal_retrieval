"""
Utility functions for the Flickr30K package.

This module provides helper functions for configuration management,
file handling, and other common tasks.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional


def get_project_root() -> Path:
    """
    Get the root directory of the project.
    
    Returns:
        Path object pointing to the project root
    """
    # Assuming this file is in src/flickr30k/utils.py
    # Go up 2 levels to reach project root
    return Path(__file__).parent.parent.parent


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config file. If None, loads default config.
        
    Returns:
        Dictionary containing configuration
    """
    if config_path is None:
        # Load default config
        project_root = get_project_root()
        config_path = project_root / "configs" / "default.yaml"
    else:
        config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config


def save_config(config: Dict[str, Any], config_path: str) -> None:
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary
        config_path: Path where to save the config
    """
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)
    
    print(f"✓ Configuration saved to: {config_path}")


def format_number(num: int) -> str:
    """
    Format a number with thousand separators.
    
    Args:
        num: Number to format
        
    Returns:
        Formatted string
    """
    return f"{num:,}"


def get_data_paths(config: Optional[Dict[str, Any]] = None) -> Dict[str, Path]:
    """
    Get standardized data paths from config or defaults.
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        Dictionary with 'data_dir', 'images_dir', 'captions_file' paths
    """
    if config is None:
        project_root = get_project_root()
        data_dir = project_root / "data"
    else:
        data_dir = Path(config.get('data', {}).get('data_dir', 'data'))
    
    paths = {
        'data_dir': data_dir,
        'images_dir': data_dir / "images",
        'captions_file': data_dir / "results.csv"
    }
    
    return paths


def check_data_availability() -> Dict[str, bool]:
    """
    Check which dataset components are available.
    
    Returns:
        Dictionary indicating availability of each component
    """
    paths = get_data_paths()
    
    availability = {
        'data_dir': paths['data_dir'].exists(),
        'images_dir': paths['images_dir'].exists(),
        'captions_file': paths['captions_file'].exists(),
        'has_images': False,
    }
    
    # Check if images directory has any images
    if availability['images_dir']:
        image_files = list(paths['images_dir'].glob("*.jpg"))
        availability['has_images'] = len(image_files) > 0
        availability['num_images'] = len(image_files)
    
    return availability


def print_data_status() -> None:
    """Print the current status of the dataset."""
    availability = check_data_availability()
    
    print("=" * 60)
    print("DATASET STATUS")
    print("=" * 60)
    
    status_icon = lambda x: "✓" if x else "✗"
    
    print(f"{status_icon(availability['data_dir'])} Data directory exists")
    print(f"{status_icon(availability['images_dir'])} Images directory exists")
    print(f"{status_icon(availability['captions_file'])} Captions file exists")
    
    if availability.get('has_images'):
        print(f"✓ Found {availability['num_images']:,} image files")
    else:
        print(f"✗ No images found")
    
    print("=" * 60)
    
    if not all([availability['data_dir'], availability['captions_file'], availability['has_images']]):
        print("\n⚠️  Dataset incomplete. Run scripts/download_flickr30k.py to download.")
