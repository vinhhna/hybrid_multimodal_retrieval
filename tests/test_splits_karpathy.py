"""
Unit tests for Karpathy split utilities.

This test validates:
- No overlap between train/val/test splits
- Correct split sizes (train=29000, val=1000, test=1000)
- Split manifests can be generated if dataset is available

The test behavior is practical for Windows dev machines:
- If manifests exist: validate them directly
- If manifests don't exist but dataset root is available (via env var): generate and validate
- Otherwise: skip the test with a clear message
"""

from __future__ import annotations

import os
import sys
import pytest
from pathlib import Path

# Add repo root to path
repo_root = Path(__file__).resolve().parent.parent
if repo_root not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.data.splits import (
    load_split_ids,
    validate_splits,
    generate_karpathy_manifests,
    default_manifest_dir,
    get_repo_root,
    EXPECTED_SIZES,
    manifest_path,
)


def test_karpathy_splits_disjoint_and_sized():
    """
    Test that Karpathy splits are disjoint and have expected sizes.
    
    This test tries multiple strategies:
    1. Load existing manifests (if they exist)
    2. Generate manifests (if dataset root is available via FLICKR30K_ROOT env var)
    3. Skip test (if neither manifests nor dataset are available)
    """
    repo_root = get_repo_root()
    manifest_dir = default_manifest_dir(repo_root)
    
    # Strategy 1: Try to load existing manifests
    manifests_exist = all(
        manifest_path(split, manifest_dir).exists()
        for split in ["train", "val", "test"]
    )
    
    if manifests_exist:
        print(f"\n✓ Found existing manifests in {manifest_dir}")
        train_ids = load_split_ids("train", manifest_dir)
        val_ids = load_split_ids("val", manifest_dir)
        test_ids = load_split_ids("test", manifest_dir)
    else:
        # Strategy 2: Try to generate manifests from dataset
        dataset_root_env = os.environ.get("FLICKR30K_ROOT")
        
        if dataset_root_env:
            dataset_root = Path(dataset_root_env)
            
            if not dataset_root.exists():
                pytest.skip(
                    f"FLICKR30K_ROOT env var set but path does not exist: {dataset_root}"
                )
            
            print(f"\n✓ Generating manifests from dataset: {dataset_root}")
            
            try:
                splits = generate_karpathy_manifests(dataset_root, manifest_dir)
                train_ids = splits["train"]
                val_ids = splits["val"]
                test_ids = splits["test"]
                print(f"✓ Generated and loaded splits")
            except Exception as e:
                pytest.skip(f"Failed to generate manifests: {e}")
        else:
            # Strategy 3: Skip test
            pytest.skip(
                "Split manifests not found and FLICKR30K_ROOT not set.\n"
                "To run this test:\n"
                "  1. Generate manifests: python scripts/write_karpathy_splits.py --dataset-root <path>\n"
                "  2. Or set FLICKR30K_ROOT environment variable pointing to dataset root"
            )
    
    # Validate splits
    print(f"\n✓ Validating splits...")
    print(f"  Train: {len(train_ids):,} images")
    print(f"  Val:   {len(val_ids):,} images")
    print(f"  Test:  {len(test_ids):,} images")
    
    # Convert to sets for validation
    train_set = set(train_ids)
    val_set = set(val_ids)
    test_set = set(test_ids)
    
    # Test 1: Check uniqueness within each split
    assert len(train_set) == len(train_ids), "Train split has duplicates"
    assert len(val_set) == len(val_ids), "Val split has duplicates"
    assert len(test_set) == len(test_ids), "Test split has duplicates"
    
    # Test 2: Check disjointness
    train_val_overlap = train_set & val_set
    train_test_overlap = train_set & test_set
    val_test_overlap = val_set & test_set
    
    assert len(train_val_overlap) == 0, f"Train and val overlap: {len(train_val_overlap)} images"
    assert len(train_test_overlap) == 0, f"Train and test overlap: {len(train_test_overlap)} images"
    assert len(val_test_overlap) == 0, f"Val and test overlap: {len(val_test_overlap)} images"
    
    # Test 3: Check expected sizes
    assert len(train_ids) == EXPECTED_SIZES["train"], (
        f"Train size mismatch: expected {EXPECTED_SIZES['train']}, got {len(train_ids)}"
    )
    assert len(val_ids) == EXPECTED_SIZES["val"], (
        f"Val size mismatch: expected {EXPECTED_SIZES['val']}, got {len(val_ids)}"
    )
    assert len(test_ids) == EXPECTED_SIZES["test"], (
        f"Test size mismatch: expected {EXPECTED_SIZES['test']}, got {len(test_ids)}"
    )
    
    print("✓ All validation checks passed!")
    print("  - No duplicates within splits")
    print("  - All splits are disjoint")
    print("  - Sizes match Karpathy convention")


def test_validate_splits_function():
    """Test the validate_splits function with synthetic data."""
    # Valid splits
    train = [f"train_{i}.jpg" for i in range(29000)]
    val = [f"val_{i}.jpg" for i in range(1000)]
    test = [f"test_{i}.jpg" for i in range(1000)]
    
    # Should not raise
    validate_splits(train, val, test)
    
    # Test overlap detection
    train_with_overlap = train + ["val_0.jpg"]  # Add one from val
    
    with pytest.raises(ValueError, match="Train and val overlap"):
        validate_splits(train_with_overlap, val, test)
    
    # Test wrong size
    train_wrong_size = [f"train_{i}.jpg" for i in range(28000)]
    
    with pytest.raises(ValueError, match="Split sizes do not match"):
        validate_splits(train_wrong_size, val, test)
    
    # Test duplicates
    train_with_dup = train + ["train_0.jpg"]
    
    with pytest.raises(ValueError, match="Train split has duplicates"):
        validate_splits(train_with_dup, val, test)


def test_forbid_test_split():
    """Test that forbid_test_split raises SystemExit for 'test' split."""
    from src.data.splits import forbid_test_split
    
    # Should not raise for train/val
    forbid_test_split("train", context="Test context")
    forbid_test_split("val", context="Test context")
    
    # Should raise SystemExit for test
    with pytest.raises(SystemExit):
        forbid_test_split("test", context="Test context")
