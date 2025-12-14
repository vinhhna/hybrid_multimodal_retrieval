"""
Detection storage and caching utilities.

Handles saving/loading detection results to avoid repeated inference.

Phase 5 Day 1: Add sharded storage API for raw detections.
Phase 5 Day 3: Implement Parquet-based shard writing/reading.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional

from .detector_base import Detection
from .types import RawDetection


# ============================================================================
# Phase 4 DetectionStorage (legacy, kept for compatibility)
# ============================================================================

class DetectionStorage:
    """
    Manages storage and retrieval of detection results (Phase 4 legacy).
    
    Phase 5: Use shard-based storage functions instead for raw detections.
    """
    
    def __init__(self, storage_dir: str):
        """
        Initialize detection storage.
        
        Args:
            storage_dir: Directory to store detection cache files
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
    
    def save_detections(self, image_id: str, detections: List[Detection]) -> None:
        """
        Save detections for an image.
        
        Phase 5 TODO: Implement serialization logic.
        
        Args:
            image_id: Unique image identifier
            detections: List of Detection objects
        """
        pass
    
    def load_detections(self, image_id: str) -> Optional[List[Detection]]:
        """
        Load cached detections for an image.
        
        Phase 5 TODO: Implement deserialization logic.
        
        Args:
            image_id: Unique image identifier
            
        Returns:
            List of Detection objects if cached, None otherwise
        """
        return None
    
    def has_cached(self, image_id: str) -> bool:
        """
        Check if detections are cached for an image.
        
        Args:
            image_id: Unique image identifier
            
        Returns:
            True if cached, False otherwise
        """
        cache_file = self.storage_dir / f"{image_id}.pkl"
        return cache_file.exists()
    
    def clear_cache(self) -> int:
        """
        Clear all cached detections.
        
        Returns:
            Number of cache files deleted
        """
        count = 0
        for cache_file in self.storage_dir.glob("*.pkl"):
            cache_file.unlink()
            count += 1
        return count


# ============================================================================
# Phase 5 Sharded Storage API (new, for raw detections)
# ============================================================================

def raw_shard_path(out_dir: Path, shard_id: int, model_id: str) -> Path:
    """
    Compute the path for a raw detection shard file.
    
    Phase 5 Day 1: Implement deterministic naming convention.
    Phase 5 Day 3: Use in write_raw_shard() and iter_raw_shards().
    
    Naming convention: raw_detections_{model_id}_shard_{shard_id:05d}.parquet
    Example: raw_detections_owlvit-base_shard_00042.parquet
    
    Args:
        out_dir: Directory to store raw detection shards
        shard_id: Shard index (0-based)
        model_id: Detector model identifier (e.g., "owlvit-base")
    
    Returns:
        Absolute Path to the shard file.
    
    Example:
        >>> from pathlib import Path
        >>> p = raw_shard_path(Path("data/vision/raw_shards"), 42, "owlvit-base")
        >>> print(p.name)
        raw_detections_owlvit-base_shard_00042.parquet
    """
    filename = f"raw_detections_{model_id}_shard_{shard_id:05d}.parquet"
    return out_dir / filename


def write_raw_shard(path: Path, rows: List[RawDetection]) -> None:
    """
    Write a shard of raw detections to a Parquet file.
    
    Phase 5 Day 1: Stub (raises NotImplementedError).
    Phase 5 Day 3: Implement Parquet writing with pyarrow/pandas.
    
    Implementation plan (Day 3):
      1. Convert RawDetection objects to pandas DataFrame
      2. Write to Parquet with pyarrow engine
      3. Use appropriate compression (snappy or zstd)
      4. Ensure deterministic column order for consistency
    
    Args:
        path: Output path for the Parquet shard file
        rows: List of RawDetection objects to write
    
    Raises:
        NotImplementedError: Always on Day 1 (deferred to Day 3).
    
    Example:
        >>> detections = [RawDetection(...), RawDetection(...)]
        >>> path = raw_shard_path(Path("data/vision/raw_shards"), 0, "owlvit-base")
        >>> write_raw_shard(path, detections)  # Writes to Parquet on Day 3+
    """
    raise NotImplementedError(
        "write_raw_shard() implementation deferred to Phase 5 Day 3. "
        "Day 1 provides only the function signature for scaffolding. "
        "Full Parquet writing will be implemented with pyarrow/pandas on Day 3."
    )


def iter_raw_shards(dir_path: Path, model_id: Optional[str] = None) -> List[Path]:
    """
    Iterate over raw detection shard files in a directory.
    
    Phase 5 Day 1: Implement glob + sorted (safe, no IO).
    Phase 5 Day 3: Use in postprocessing pipeline.
    
    Args:
        dir_path: Directory containing raw detection shards
        model_id: Optional model ID filter (only return shards for this model).
                  If None, return all raw_detections_*.parquet files.
    
    Returns:
        Sorted list of shard file paths (sorted by filename for deterministic order).
    
    Example:
        >>> shards = iter_raw_shards(Path("data/vision/raw_shards"), "owlvit-base")
        >>> print(len(shards))  # e.g., 50 shards
        >>> for shard_path in shards:
        ...     # Process shard (Day 3+)
        ...     pass
    """
    if not dir_path.exists():
        return []
    
    if model_id:
        pattern = f"raw_detections_{model_id}_shard_*.parquet"
    else:
        pattern = "raw_detections_*_shard_*.parquet"
    
    shards = list(dir_path.glob(pattern))
    return sorted(shards)  # Deterministic order by filename
