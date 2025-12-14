#!/usr/bin/env python

"""
Build the Flickr30K entity vocabulary and context JSON files.

This script loads the Flickr30K dataset, runs entity extraction on all captions,
and builds a vocabulary with context mappings (entity -> images, captions).

Usage (from repo root):
    python -m scripts.build_entity_vocabulary
"""

from __future__ import annotations

import argparse
import sys
import yaml
from pathlib import Path
from typing import Any, Dict, List

# Add project root to path if needed
project_root = Path(__file__).resolve().parent.parent
if project_root not in sys.path:
    sys.path.insert(0, str(project_root))

from src.flickr30k.dataset import Flickr30KDataset
from src.graph.entities import (
    build_entity_vocabulary,
    build_entity_embeddings_and_meta,
    save_entity_embeddings_and_meta,
)
from src.graph.config import load_entity_graph_config, get_entity_graph_config
from src.data.splits import load_split_ids, forbid_test_split


class DatasetAdapter:
    """
    Adapter to make Flickr30KDataset compatible with build_entity_vocabulary.
    
    The entity builder expects a dataset that supports:
      - len(dataset): returns number of images
      - dataset[i]: returns dict with keys: image_id, captions
    
    This adapter wraps the existing Flickr30KDataset to provide that interface.
    """
    
    def __init__(self, flickr_dataset: Flickr30KDataset, split_ids: List[str] | None = None):
        """
        Initialize the adapter.
        
        Args:
            flickr_dataset: Loaded Flickr30KDataset instance.
            split_ids: Optional list of image IDs to filter by (for train/val/test split).
        """
        self.dataset = flickr_dataset
        if self.dataset.df is None:
            raise ValueError("Dataset must be loaded before creating adapter")
        
        # Get unique image names (our iteration basis)
        all_image_names = self.dataset.get_unique_images()
        
        # Filter by split if provided
        if split_ids is not None:
            split_set = set(split_ids)
            self.image_names = [img for img in all_image_names if img in split_set]
            print(f"  ✓ Filtered to {len(self.image_names):,} images from {len(all_image_names):,} total (split filter)")
        else:
            self.image_names = all_image_names
    
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
      1. Parse arguments and enforce split guardrails
      2. Load split IDs
      3. Instantiate Flickr30KDataset
      4. Wrap it in DatasetAdapter with split filter
      5. Configure entity builder
      6. Call build_entity_vocabulary
      7. Optionally build entity embeddings and metadata (if enabled in config)
      8. Print summary statistics
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Build entity vocabulary from Flickr30K dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--split",
        type=str,
        required=True,
        choices=["train", "val", "test"],
        help="Dataset split to use (train/val/test)"
    )
    args = parser.parse_args()
    
    # Enforce leakage prevention: forbid test split for KG artifact building
    forbid_test_split(args.split, context="Entity vocabulary build scripts must not run on test")
    
    print("=" * 70)
    print("FLICKR30K ENTITY VOCABULARY BUILDER")
    print("=" * 70)
    print(f"\n🎯 Split: {args.split}")
    
    # Load split IDs
    print(f"\n📋 Loading {args.split} split manifest...")
    try:
        split_ids = load_split_ids(args.split)
        print(f"  ✓ Loaded {len(split_ids):,} image IDs for {args.split} split")
    except FileNotFoundError as e:
        print(f"\n✗ Error: {e}")
        print("\nPlease run: python scripts/write_karpathy_splits.py --dataset-root <path>")
        sys.exit(1)
    
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
    
    # Wrap in adapter with split filter
    print(f"\n🔄 Creating dataset adapter (filtering by {args.split} split)...")
    adapted_dataset = DatasetAdapter(dataset, split_ids=split_ids)
    print(f"✓ Adapter ready: {len(adapted_dataset):,} images (filtered from {dataset.num_images:,} total)")
    
    # Load configuration from YAML using the new helper
    config_path = project_root / "configs" / "entity_graph.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    print(f"\n📄 Loading configuration from: {config_path}")
    cfg = load_entity_graph_config(config_path)
    
    if "entity_graph" not in cfg:
        raise KeyError(
            f"Config file {config_path} must define an 'entity_graph' section."
        )
    
    entity_cfg = get_entity_graph_config(cfg)
    print(f"\n⚙️  Configuration (from configs/entity_graph.yaml):")
    print(f"  min_df: {entity_cfg.get('min_df')}")
    print(f"  max_samples: {entity_cfg.get('max_samples')}")
    print(f"  vocab_path: {entity_cfg.get('vocab_path')}")
    print(f"  context_path: {entity_cfg.get('context_path')}")
    print(f"  build_entity_embeddings: {entity_cfg.get('build_entity_embeddings')}")
    
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
    
    # Build entity embeddings and metadata if enabled
    build_embeddings = entity_cfg.get("build_entity_embeddings", False)
    
    if build_embeddings and len(entity_vocab) > 0:
        print(f"\n" + "=" * 70)
        print("BUILDING ENTITY EMBEDDINGS AND METADATA")
        print("=" * 70)
        
        # Initialize CLIP text encoder
        print("\n📦 Loading CLIP text encoder...")
        try:
            from src.retrieval.bi_encoder import BiEncoder
            
            # Use the same CLIP model as in retrieval
            text_encoder = BiEncoder(model_name='ViT-B/32', device='cuda')
            print(f"  ✓ Loaded model: {text_encoder.model_name}")
            print(f"  ✓ Device: {text_encoder.device}")
        except Exception as e:
            print(f"  ✗ Failed to load CLIP encoder: {e}")
            print("  Skipping entity embeddings")
            build_embeddings = False
        
        if build_embeddings:
            # Build embeddings and metadata
            try:
                embeddings, entity_meta = build_entity_embeddings_and_meta(
                    text_encoder=text_encoder,
                    entity_vocab=entity_vocab,
                    cfg=cfg,
                )
                
                # Resolve output paths from config
                embeddings_path = Path(entity_cfg.get(
                    "entity_embeddings_path",
                    "data/entities/entity_embeddings.pt"
                ))
                meta_path = Path(entity_cfg.get(
                    "entity_meta_path",
                    "data/entities/entity_meta.json"
                ))
                
                # Make paths absolute if they're relative
                if not embeddings_path.is_absolute():
                    embeddings_path = project_root / embeddings_path
                if not meta_path.is_absolute():
                    meta_path = project_root / meta_path
                
                # Save artifacts
                save_entity_embeddings_and_meta(
                    embeddings=embeddings,
                    entity_meta=entity_meta,
                    embeddings_path=embeddings_path,
                    meta_path=meta_path,
                )
                
                print(f"\n✓ Entity embeddings and metadata saved successfully")
                print(f"  Embeddings: {embeddings_path}")
                print(f"  Metadata: {meta_path}")
                
            except Exception as e:
                print(f"\n✗ Failed to build entity embeddings: {e}")
                import traceback
                traceback.print_exc()
    else:
        if not build_embeddings:
            print(f"\n⏭️  Skipping entity embeddings (build_entity_embeddings=False)")
        elif len(entity_vocab) == 0:
            print(f"\n⏭️  Skipping entity embeddings (empty vocabulary)")
    
    print(f"\n" + "=" * 70)
    print("✅ ENTITY VOCABULARY BUILD COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
