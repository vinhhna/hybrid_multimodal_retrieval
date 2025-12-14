#!/usr/bin/env python

"""
Generate Karpathy split manifests for Flickr30K dataset.

This script parses a Karpathy-style dataset JSON and generates three manifest files:
- data/splits/karpathy_train.json (29,000 images)
- data/splits/karpathy_val.json (1,000 images)
- data/splits/karpathy_test.json (1,000 images)

Each manifest is a JSON array of image filenames (e.g., ['1000092795.jpg', ...]).

Usage:
    python scripts/write_karpathy_splits.py --dataset-root D:\\datasets\\flickr30k
    python scripts/write_karpathy_splits.py --dataset-root $FLICKR30K_ROOT --force
    python scripts/write_karpathy_splits.py  # reads from FLICKR30K_ROOT env var
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Add repo root to path
repo_root = Path(__file__).resolve().parent.parent
if repo_root not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.data.splits import (
    generate_karpathy_manifests,
    default_manifest_dir,
    manifest_path,
    SplitName,
)


def main() -> None:
    """Generate Karpathy split manifests from dataset JSON."""
    parser = argparse.ArgumentParser(
        description="Generate Karpathy split manifests for Flickr30K",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/write_karpathy_splits.py --dataset-root D:\\datasets\\flickr30k
  python scripts/write_karpathy_splits.py --dataset-root $FLICKR30K_ROOT --force
  
Environment variables:
  FLICKR30K_ROOT: Default dataset root if --dataset-root not provided
        """
    )
    
    parser.add_argument(
        "--dataset-root",
        type=str,
        help="Root directory containing Karpathy dataset JSON (default: from FLICKR30K_ROOT env var)"
    )
    
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output directory for manifests (default: data/splits from repo root)"
    )
    
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing manifests if they already exist"
    )
    
    args = parser.parse_args()
    
    # Determine dataset root
    if args.dataset_root:
        dataset_root = Path(args.dataset_root)
    elif "FLICKR30K_ROOT" in os.environ:
        dataset_root = Path(os.environ["FLICKR30K_ROOT"])
        print(f"Using FLICKR30K_ROOT from environment: {dataset_root}")
    else:
        print("ERROR: No dataset root specified", file=sys.stderr)
        print("\nProvide dataset root via:", file=sys.stderr)
        print("  1. --dataset-root argument", file=sys.stderr)
        print("  2. FLICKR30K_ROOT environment variable", file=sys.stderr)
        sys.exit(1)
    
    if not dataset_root.exists():
        print(f"ERROR: Dataset root does not exist: {dataset_root}", file=sys.stderr)
        sys.exit(1)
    
    # Determine output directory
    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        out_dir = default_manifest_dir(repo_root)
    
    print(f"\n{'=' * 70}")
    print("KARPATHY SPLIT MANIFEST GENERATOR")
    print(f"{'=' * 70}\n")
    print(f"Dataset root: {dataset_root}")
    print(f"Output dir:   {out_dir}")
    print()
    
    # Check if manifests already exist
    if not args.force:
        existing = []
        missing = []
        splits_tuple: tuple[SplitName, ...] = ("train", "val", "test")
        
        for split_name in splits_tuple:
            manifest_file = manifest_path(split_name, out_dir)
            if manifest_file.exists():
                existing.append((split_name, manifest_file))
            else:
                missing.append(split_name)
        
        if existing and not missing:
            # All three exist - skip entirely
            print("All manifests already exist:")
            for split_name, f in existing:
                print(f"  - {f}")
            print("\nUse --force to overwrite existing manifests.")
            print("Skipping generation.")
            return
        elif existing:
            # Partial state - require --force
            print("WARNING: Partial manifest state detected:")
            print(f"  Existing: {', '.join(s for s, _ in existing)}")
            print(f"  Missing: {', '.join(missing)}")
            print("\nUse --force to regenerate all manifests (recommended for consistency).")
            print("Skipping generation.")
            return
    
    # Generate manifests
    try:
        splits = generate_karpathy_manifests(dataset_root, out_dir)
    except Exception as e:
        print(f"\nERROR: Failed to generate manifests", file=sys.stderr)
        print(f"{type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Print summary
    print(f"\n{'=' * 70}")
    print("GENERATION COMPLETE")
    print(f"{'=' * 70}\n")
    
    total = 0
    splits_tuple: tuple[SplitName, ...] = ("train", "val", "test")
    for split_name in splits_tuple:
        count = len(splits[split_name])
        total += count
        manifest_file = manifest_path(split_name, out_dir)
        print(f"{split_name:5s}: {count:5,} images -> {manifest_file}")
    
    print(f"\nTotal: {total:5,} images")
    print()


if __name__ == "__main__":
    main()
