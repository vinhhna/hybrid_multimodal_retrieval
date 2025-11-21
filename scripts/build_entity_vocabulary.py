#!/usr/bin/env python

"""
Build the Flickr30K entity vocabulary and context JSON files.

This script loads the Flickr30K dataset, runs entity extraction on all captions,
and builds a vocabulary with context mappings (entity -> images, captions).

Usage (from repo root):
    python -m scripts.build_entity_vocabulary
"""

from __future__ import annotations

import sys
import yaml
from pathlib import Path
from typing import Any, Dict, List

# Add project root to path if needed
project_root = Path(__file__).resolve().parent.parent
if project_root not in sys.path:
    sys.path.insert(0, str(project_root))

from src.flickr30k.dataset import Flickr30KDataset
from src.graph.entities import build_entity_vocabulary


class DatasetAdapter:
    """
    Adapter to make Flickr30KDataset compatible with build_entity_vocabulary.
    
    The entity builder expects a dataset that supports:
      - len(dataset): returns number of images
      - dataset[i]: returns dict with keys: image_id, captions
    
    This adapter wraps the existing Flickr30KDataset to provide that interface.
    """
    
    def __init__(self, flickr_dataset: Flickr30KDataset):
        """
        Initialize the adapter.
        
        Args:
            flickr_dataset: Loaded Flickr30KDataset instance.
        """
        self.dataset = flickr_dataset
        if self.dataset.df is None:
            raise ValueError("Dataset must be loaded before creating adapter")
        
        # Get unique image names (our iteration basis)
        self.image_names = self.dataset.get_unique_images()
    
    def __len__(self) -> int:
        """Return the number of images."""
        return len(self.image_names)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get an item by index.
        
        Args:
            idx: Index of the image (0 to len-1).
        
        Returns:
            Dictionary with keys:
              - image_id: str (image filename)
              - captions: List[str] (all captions for this image)
        """
        if idx < 0 or idx >= len(self.image_names):
            raise IndexError(f"Index {idx} out of range [0, {len(self.image_names)})")
        
        image_name = self.image_names[idx]
        captions = self.dataset.get_captions(image_name)
        
        return {
            "image_id": image_name,
            "captions": captions,
        }


def main() -> None:
    """
    Main function to build entity vocabulary.
    
    Steps:
      1. Instantiate Flickr30KDataset
      2. Wrap it in DatasetAdapter
      3. Configure entity builder
      4. Call build_entity_vocabulary
      5. Print summary statistics
    """
    print("=" * 70)
    print("FLICKR30K ENTITY VOCABULARY BUILDER")
    print("=" * 70)
    
    # Determine data paths
    # Try Kaggle path first, fall back to local
    kaggle_data_root = Path("/kaggle/input/flickr30k/data")
    local_data_root = project_root / "data"
    
    if kaggle_data_root.exists():
        data_root = kaggle_data_root
        print(f"\n✓ Using Kaggle data: {data_root}")
    else:
        data_root = local_data_root
        print(f"\n✓ Using local data: {data_root}")
    
    images_dir = data_root / "images"
    captions_path = data_root / "results.csv"
    
    # Instantiate dataset
    print(f"\n📂 Loading Flickr30K dataset...")
    print(f"  Images dir: {images_dir}")
    print(f"  Captions: {captions_path}")
    
    dataset = Flickr30KDataset(
        images_dir=str(images_dir),
        captions_file=str(captions_path),
        auto_load=True,
    )
    
    # Wrap in adapter
    print(f"\n🔄 Creating dataset adapter...")
    adapted_dataset = DatasetAdapter(dataset)
    print(f"✓ Adapter ready: {len(adapted_dataset)} images")
    
    # Load configuration from YAML
    config_path = project_root / "config" / "entity_graph.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    print(f"\n📄 Loading configuration from: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    
    if "entity_graph" not in cfg:
        raise KeyError(
            f"Config file {config_path} must define an 'entity_graph' section."
        )
    
    entity_cfg = cfg["entity_graph"]
    print(f"\n⚙️  Configuration (from config/entity_graph.yaml):")
    print(f"  min_df: {entity_cfg.get('min_df')}")
    print(f"  max_samples: {entity_cfg.get('max_samples')}")
    print(f"  vocab_path: {entity_cfg.get('vocab_path')}")
    print(f"  context_path: {entity_cfg.get('context_path')}")
    
    # Build entity vocabulary
    print(f"\n" + "=" * 70)
    print("BUILDING ENTITY VOCABULARY")
    print("=" * 70)
    
    entity_vocab, entity_context = build_entity_vocabulary(adapted_dataset, cfg)
    
    # Print final statistics
    print(f"\n" + "=" * 70)
    print("FINAL STATISTICS")
    print("=" * 70)
    print(f"\nFinal vocabulary size: {len(entity_vocab)}")
    
    if entity_vocab:
        print("\nTop 10 entities by df_caption:")
        by_df = sorted(
            entity_vocab.items(),
            key=lambda kv: kv[1].df_caption,
            reverse=True,
        )[:10]
        for name, stats in by_df:
            print(
                f"  {name!r}: id={stats.id}, "
                f"df_caption={stats.df_caption}, "
                f"df_image={stats.df_image}, "
                f"cf={stats.cf}"
            )
    
    print(f"\n" + "=" * 70)
    print("✅ ENTITY VOCABULARY BUILD COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
