"""
Karpathy split utilities for Flickr30K dataset.

This module provides functions to load, validate, and manage train/val/test splits
following the Karpathy split convention for Flickr30K.

Expected split sizes (Flickr30K Karpathy - canonical):
- train: 29,000 images
- val: 1,014 images
- test: 1,000 images
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Literal

# Type alias for split names
SplitName = Literal["train", "val", "test"]

# Expected Flickr30K Karpathy split sizes (canonical)
EXPECTED_SIZES = {
    "train": 29000,
    "val": 1014,  # Canonical Karpathy split uses 1014 validation images
    "test": 1000,
}


def get_repo_root(start: Path | None = None) -> Path:
    """
    Detect the repository root directory.
    
    Searches upward from start directory (or current working directory)
    for markers like setup.py, .git, or specific directories.
    
    Args:
        start: Starting directory for search (default: current working directory)
    
    Returns:
        Path to repository root (falls back to current working directory if not found)
    """
    if start is None:
        start = Path.cwd()
    else:
        start = Path(start).resolve()
    
    # Markers that indicate repo root
    markers = ["setup.py", ".git", "src", "scripts", "configs"]
    
    current = start
    while current != current.parent:  # Stop at filesystem root
        # Check for at least 2 markers to avoid false positives
        found_markers = sum(1 for marker in markers if (current / marker).exists())
        if found_markers >= 2:
            return current
        current = current.parent
    
    # Fallback to current working directory
    return Path.cwd()


def default_manifest_dir(repo_root: Path) -> Path:
    """
    Get the default directory for split manifest files.
    
    Args:
        repo_root: Repository root directory
    
    Returns:
        Path to data/splits/ directory
    """
    return repo_root / "data" / "splits"


def manifest_path(split: SplitName, manifest_dir: Path) -> Path:
    """
    Get the path to a split manifest file.
    
    Args:
        split: Split name (train, val, or test)
        manifest_dir: Directory containing manifest files
    
    Returns:
        Path to karpathy_{split}.json
    """
    return manifest_dir / f"karpathy_{split}.json"


def load_split_ids(split: SplitName, manifest_dir: Path | None = None) -> list[str]:
    """
    Load image IDs for a specific split from manifest file.
    
    Args:
        split: Split name (train, val, or test)
        manifest_dir: Directory containing manifest files (default: data/splits from repo root)
    
    Returns:
        List of image IDs (filenames like '1000092795.jpg')
    
    Raises:
        FileNotFoundError: If manifest file does not exist
        ValueError: If manifest file is invalid
    """
    if manifest_dir is None:
        repo_root = get_repo_root()
        manifest_dir = default_manifest_dir(repo_root)
    else:
        manifest_dir = Path(manifest_dir)
    
    manifest_file = manifest_path(split, manifest_dir)
    
    if not manifest_file.exists():
        raise FileNotFoundError(
            f"Split manifest not found: {manifest_file}\n"
            f"Run scripts/write_karpathy_splits.py to generate manifests."
        )
    
    try:
        with open(manifest_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in manifest file {manifest_file}: {e}")
    
    if not isinstance(data, list):
        raise ValueError(f"Manifest file must contain a JSON array, got {type(data).__name__}")
    
    if not all(isinstance(item, str) for item in data):
        raise ValueError("All items in manifest must be strings")
    
    return data


def write_split_ids(
    split: SplitName,
    ids: list[str],
    manifest_dir: Path | None = None
) -> Path:
    """
    Write image IDs for a split to manifest file.
    
    Args:
        split: Split name (train, val, or test)
        ids: List of image IDs to write
        manifest_dir: Directory for manifest files (default: data/splits from repo root)
    
    Returns:
        Path to written manifest file
    """
    if manifest_dir is None:
        repo_root = get_repo_root()
        manifest_dir = default_manifest_dir(repo_root)
    else:
        manifest_dir = Path(manifest_dir)
    
    # Create directory if it doesn't exist
    manifest_dir.mkdir(parents=True, exist_ok=True)
    
    manifest_file = manifest_path(split, manifest_dir)
    
    # Write sorted IDs for determinism
    sorted_ids = sorted(ids)
    
    with open(manifest_file, 'w', encoding='utf-8') as f:
        json.dump(sorted_ids, f, indent=2)
    
    return manifest_file


def generate_karpathy_manifests(
    dataset_root: Path,
    out_dir: Path,
    karpathy_json: Path | None = None,
) -> dict[SplitName, list[str]]:
    """
    Generate train/val/test split manifests from Karpathy-style dataset JSON.
    
    Expected JSON structure:
        {
            "images": [
                {
                    "filename": "1000092795.jpg",
                    "split": "train",
                    ...
                },
                ...
            ]
        }
    
    Alternative field names are also supported:
    - "dataset" instead of "images" at top level
    - "file_name", "img", "image" instead of "filename"
    
    Args:
        dataset_root: Root directory containing Karpathy dataset JSON (used for search if karpathy_json not provided)
        out_dir: Output directory for manifest files
        karpathy_json: Optional explicit path to Karpathy JSON file (bypasses search logic)
    
    Returns:
        Dictionary mapping split names to lists of image IDs
    
    Raises:
        FileNotFoundError: If no Karpathy dataset JSON found (when karpathy_json not provided)
        ValueError: If JSON structure is invalid
    """
    dataset_root = Path(dataset_root)
    
    # Determine JSON source: explicit path or search
    if karpathy_json is not None:
        # Explicit JSON path provided
        dataset_json = Path(karpathy_json)
        if not dataset_json.exists():
            raise FileNotFoundError(
                f"Karpathy JSON not found at specified path: {dataset_json}\n"
                f"Please check the --karpathy-json argument."
            )
        print(f"Loading Karpathy dataset from (explicit): {dataset_json}")
    else:
        # Search for Karpathy dataset JSON (common filenames)
        candidate_names = [
            "dataset_flickr30k.json",
            "dataset.json",
            "flickr30k_karpathy.json",
            "karpathy_splits.json",
        ]
        
        dataset_json = None
        for candidate in candidate_names:
            candidate_path = dataset_root / candidate
            if candidate_path.exists():
                dataset_json = candidate_path
                break
        
        if dataset_json is None:
            raise FileNotFoundError(
                f"No Karpathy dataset JSON found in {dataset_root}\n"
                f"Expected one of: {', '.join(candidate_names)}\n"
                f"Tip: Use --karpathy-json to specify an explicit path."
            )
        
        print(f"Loading Karpathy dataset from (search): {dataset_json}")
    
    try:
        with open(dataset_json, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {dataset_json}: {e}")
    
    # Extract images list (try multiple common structures)
    images = data.get("images")
    if images is None:
        images = data.get("dataset", {}).get("images")
    if images is None:
        raise ValueError(f"No 'images' array found in {dataset_json}")
    
    if not isinstance(images, list):
        raise ValueError(f"'images' must be an array, got {type(images).__name__}")
    
    # Build split buckets
    buckets: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    
    for img in images:
        if not isinstance(img, dict):
            continue
        
        # Extract split
        split = img.get("split")
        if split not in buckets:
            continue
        
        # Extract filename (try multiple field names)
        filename = (
            img.get("filename") or
            img.get("file_name") or
            img.get("img") or
            img.get("image")
        )
        
        if not filename:
            continue
        
        # Normalize to image_id (filename like '1000092795.jpg')
        # Use Path.name to strip any directory prefix (e.g., 'flickr30k_images/123.jpg' -> '123.jpg')
        # This ensures compatibility with repo's bare-filename convention
        #
        # ASSUMPTION: Standard Karpathy JSONs provide filenames WITH extensions (.jpg).
        # If your JSON has extensionless IDs (e.g., integer 1000092795 or string "1000092795"),
        # this will produce IDs without extensions, which may cause FileNotFoundError downstream
        # when the dataset loader tries to access images on disk.
        image_id = Path(str(filename)).name
        
        buckets[split].append(image_id)
    
    # Validate extracted splits
    train_ids = buckets["train"]
    val_ids = buckets["val"]
    test_ids = buckets["test"]
    
    # Defensive check: Warn if many IDs lack extensions
    all_ids = train_ids + val_ids + test_ids
    ids_without_ext = [img_id for img_id in all_ids if not Path(img_id).suffix]
    if ids_without_ext:
        pct = 100 * len(ids_without_ext) / len(all_ids)
        if pct > 10:  # If more than 10% lack extensions, this is likely a problem
            print(f"\n⚠️  WARNING: {len(ids_without_ext):,} / {len(all_ids):,} ({pct:.1f}%) image IDs lack file extensions", file=sys.stderr)
            print(f"   Examples: {ids_without_ext[:3]}", file=sys.stderr)
            print(f"   This may cause FileNotFoundError when loading images.", file=sys.stderr)
            print(f"   Check if your Karpathy JSON provides filenames WITH extensions (.jpg).\n", file=sys.stderr)
    
    validate_splits(train_ids, val_ids, test_ids)
    
    # Write manifests
    result = {}
    for split_name in ["train", "val", "test"]:
        ids = buckets[split_name]
        manifest_file = write_split_ids(split_name, ids, out_dir)  # type: ignore
        result[split_name] = ids  # type: ignore
        print(f"✓ Wrote {len(ids)} {split_name} IDs to {manifest_file}")
    
    return result  # type: ignore


def validate_splits(
    train_ids: list[str],
    val_ids: list[str],
    test_ids: list[str],
    verbose: bool = True
) -> None:
    """
    Validate that splits are disjoint and have expected sizes.
    
    Args:
        train_ids: List of training image IDs
        val_ids: List of validation image IDs
        test_ids: List of test image IDs
        verbose: Whether to print validation success message (default: True)
    
    Raises:
        ValueError: If validation fails
    """
    train_set = set(train_ids)
    val_set = set(val_ids)
    test_set = set(test_ids)
    
    # Check uniqueness within each split
    if len(train_set) != len(train_ids):
        raise ValueError(f"Train split has duplicates: {len(train_ids)} IDs, {len(train_set)} unique")
    if len(val_set) != len(val_ids):
        raise ValueError(f"Val split has duplicates: {len(val_ids)} IDs, {len(val_set)} unique")
    if len(test_set) != len(test_ids):
        raise ValueError(f"Test split has duplicates: {len(test_ids)} IDs, {len(test_set)} unique")
    
    # Check disjointness
    train_val_overlap = train_set & val_set
    train_test_overlap = train_set & test_set
    val_test_overlap = val_set & test_set
    
    if train_val_overlap:
        raise ValueError(f"Train and val overlap: {len(train_val_overlap)} images")
    if train_test_overlap:
        raise ValueError(f"Train and test overlap: {len(train_test_overlap)} images")
    if val_test_overlap:
        raise ValueError(f"Val and test overlap: {len(val_test_overlap)} images")
    
    # Check expected sizes
    size_errors = []
    for split_name, ids, expected in [
        ("train", train_ids, EXPECTED_SIZES["train"]),
        ("val", val_ids, EXPECTED_SIZES["val"]),
        ("test", test_ids, EXPECTED_SIZES["test"]),
    ]:
        if len(ids) != expected:
            size_errors.append(f"{split_name}: expected {expected}, got {len(ids)}")
    
    if size_errors:
        raise ValueError(f"Split sizes do not match Karpathy convention:\n" + "\n".join(size_errors))
    
    if verbose:
        print("✓ Split validation passed:")
        print(f"  - Train: {len(train_ids):,} images")
        print(f"  - Val: {len(val_ids):,} images")
        print(f"  - Test: {len(test_ids):,} images")
        print(f"  - All splits are disjoint")


def forbid_test_split(split: str, *, context: str = "This operation") -> None:
    """
    Enforce leakage prevention by forbidding test split in certain contexts.
    
    This function should be called in scripts that build KG artifacts or
    perform calibration to prevent test set leakage.
    
    Args:
        split: The split name being used
        context: Description of the operation for error message
    
    Raises:
        SystemExit: If split is 'test'
    """
    if split.lower() == "test":
        print(f"\n{'=' * 70}", file=sys.stderr)
        print(f"ERROR: Test split forbidden", file=sys.stderr)
        print(f"{'=' * 70}", file=sys.stderr)
        print(f"\n{context} must not use the test split.", file=sys.stderr)
        print(f"\nThis is a leakage prevention guardrail.", file=sys.stderr)
        print(f"Only train and val splits are allowed for:", file=sys.stderr)
        print(f"  - Building knowledge graph artifacts", file=sys.stderr)
        print(f"  - Building entity vocabularies", file=sys.stderr)
        print(f"  - Calibration and hyperparameter tuning", file=sys.stderr)
        print(f"\nThe test split should only be used for final evaluation.", file=sys.stderr)
        print(f"{'=' * 70}\n", file=sys.stderr)
        sys.exit(1)
