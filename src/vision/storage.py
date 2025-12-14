"""
Detection storage and caching utilities.

Handles saving/loading detection results to avoid repeated inference.

Phase 5 Day 1: Add sharded storage API for raw detections.
Phase 5 Day 3: Implement Parquet-based shard writing/reading with two tables
               (raw and postprocessed) and deterministic schema.
"""

from __future__ import annotations

import os
import re
import time
import warnings
from enum import Enum
from pathlib import Path
from typing import List, Dict, Any, Optional, Literal, Iterator
import pandas as pd
import numpy as np

from .detector_base import Detection
from .types import RawDetection


# ============================================================================
# Phase 5 Day 3: Schema definitions and types
# ============================================================================

SCHEMA_VERSION: str = "v1"


class PhraseType(str, Enum):
    """Standardized phrase type enumeration."""
    OBJECT = "object"
    ATTRIBUTE = "attribute"
    COMPOSED = "composed"
    NULL = "null"


# Type alias for table names
DetectionTableName = Literal["raw", "post"]

# Allowed phrase type values for validation
ALLOWED_PHRASE_TYPES = {"object", "attribute", "composed", "null"}


def parse_shard_index(path: Path) -> Optional[int]:
    """
    Extract shard index from filename, handling both new and legacy naming.
    
    Supports:
    - New: {table}__{split}__{model}__v1__shard000042.parquet
    - New with worker: {table}__{split}__{model}__v1__shard000042_w1.parquet
    - Legacy: raw_detections_{model}_shard_00042.parquet
    
    Args:
        path: Shard file path
    
    Returns:
        Shard index integer, or None if not parseable
    
    Example:
        >>> parse_shard_index(Path("raw__train__owlvit__v1__shard000042.parquet"))
        42
        >>> parse_shard_index(Path("raw__train__owlvit__v1__shard000042_w1.parquet"))
        42
        >>> parse_shard_index(Path("raw_detections_owlvit_shard_00042.parquet"))
        42
    """
    # Try new naming: __shard000042.parquet or __shard000042_w1.parquet
    m = re.search(r"__shard(\d{6})(?:_w\d+)?\.parquet$", path.name)
    if m:
        return int(m.group(1))
    
    # Try legacy naming: _shard_00042.parquet
    m2 = re.search(r"_shard_(\d+)\.parquet$", path.name)
    if m2:
        return int(m2.group(1))
    
    return None


# Raw detections schema: minimal detection output
RAW_SCHEMA_COLUMNS = {
    "schema_version": "string",
    "image_id": "string",
    "entity_id": "int64",
    "phrase_type": "string",
    "model_id": "string",
    "conf_raw": "float64",
    "x1": "float64",
    "y1": "float64",
    "x2": "float64",
    "y2": "float64",
}

# Postprocessed detections schema: adds postprocessing fields
POST_SCHEMA_COLUMNS = {
    **RAW_SCHEMA_COLUMNS,
    "noise_floor": "float64",
    "tau_eff": "float64",
    "conf_eff": "float64",
    "is_present": "bool",
}


# ============================================================================
# Phase 4 DetectionStorage (legacy, kept for compatibility)
# ============================================================================

class DetectionStorage:
    """
    Manages storage and retrieval of detection results (Phase 4 legacy).
    
    .. deprecated:: Phase 5
        Use ParquetShardWriter and related functions instead.
        This class is maintained for backward compatibility only.
    
    Phase 5: Use shard-based storage functions instead for raw detections.
    """
    
    def __init__(self, storage_dir: str):
        """
        Initialize detection storage.
        
        Args:
            storage_dir: Directory to store detection cache files
        
        .. deprecated:: Phase 5
            Use ParquetShardWriter instead.
        """
        warnings.warn(
            "DetectionStorage is deprecated. Use ParquetShardWriter instead.",
            DeprecationWarning,
            stacklevel=2
        )
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
    Compute the path for a raw detection shard file (legacy naming).
    
    .. deprecated:: Phase 5 Day 3
        Use make_shard_filename() and list_shards() instead.
        This function uses the legacy naming convention for backward compatibility.
    
    Args:
        out_dir: Directory to store raw detection shards
        shard_id: Shard index (0-based)
        model_id: Detector model identifier (e.g., "owlvit-base")
    
    Returns:
        Absolute Path to the shard file using legacy naming.
    
    Example:
        >>> from pathlib import Path
        >>> p = raw_shard_path(Path("data/vision/raw_shards"), 42, "owlvit-base")
        >>> print(p.name)
        raw_detections_owlvit-base_shard_00042.parquet
    """
    warnings.warn(
        "raw_shard_path() is deprecated. "
        "Use make_shard_filename() with list_shards() instead.",
        DeprecationWarning,
        stacklevel=2
    )
    filename = f"raw_detections_{model_id}_shard_{shard_id:05d}.parquet"
    return out_dir / filename


def write_raw_shard(path: Path, rows: List[RawDetection]) -> None:
    """
    Write a shard of raw detections to a Parquet file (legacy wrapper).
    
    .. deprecated:: Phase 5 Day 3
        Use ParquetShardWriter instead for resume-safe, validated writes.
    
    Args:
        path: Output path for the Parquet shard file (will be written atomically)
        rows: List of RawDetection objects to write
    
    Raises:
        ValueError: If rows cannot be converted to valid DataFrame
        ImportError: If pyarrow is not installed
    """
    warnings.warn(
        "write_raw_shard() is deprecated. "
        "Use ParquetShardWriter instead for resume-safe, validated writes.",
        DeprecationWarning,
        stacklevel=2
    )
    
    if not rows:
        # Empty list - create empty DataFrame with schema
        df = pd.DataFrame(columns=list(RAW_SCHEMA_COLUMNS.keys()))
        df = coerce_detection_df(df, table="raw")
    else:
        # Convert RawDetection objects to DataFrame
        data_dicts = []
        for row in rows:
            if hasattr(row, "__dict__"):
                data_dicts.append(row.__dict__)
            elif hasattr(row, "_asdict"):
                data_dicts.append(row._asdict())
            else:
                raise ValueError(
                    f"Cannot convert RawDetection object to dict: {type(row)}"
                )
        
        df = pd.DataFrame(data_dicts)
        df = coerce_detection_df(df, table="raw")
    
    # Validate
    validate_detection_df(df, table="raw", require_finite=True)
    
    # Write atomically
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(".parquet.tmp")
    
    try:
        df.to_parquet(temp_path, engine="pyarrow", compression="snappy", index=False)
        temp_path.replace(path)
    except ImportError as e:
        raise ImportError(
            "pyarrow is required for Parquet writing. "
            "Install with: pip install pyarrow"
        ) from e
    except Exception:
        # Clean up temp file on error
        if temp_path.exists():
            temp_path.unlink()
        raise


def iter_raw_shards(dir_path: Path, model_id: Optional[str] = None) -> List[Path]:
    """
    Iterate over raw detection shard files in a directory (legacy naming).
    
    .. deprecated:: Phase 5 Day 3
        Use list_shards() instead for deterministic naming.
    
    Args:
        dir_path: Directory containing raw detection shards
        model_id: Optional model ID filter (only return shards for this model).
                  If None, return all raw_detections_*.parquet files.
    
    Returns:
        Sorted list of shard file paths (sorted by filename for deterministic order).
    """
    warnings.warn(
        "iter_raw_shards() is deprecated. "
        "Use list_shards('raw', split, model_id) instead.",
        DeprecationWarning,
        stacklevel=2
    )
    
    if not dir_path.exists():
        return []
    
    if model_id:
        pattern = f"raw_detections_{model_id}_shard_*.parquet"
    else:
        pattern = "raw_detections_*_shard_*.parquet"
    
    shards = list(dir_path.glob(pattern))
    return sorted(shards)


# ============================================================================
# Phase 5 Day 3: Validation helpers
# ============================================================================

def validate_detection_df(
    df: pd.DataFrame,
    table: DetectionTableName,
    *,
    require_finite: bool = True
) -> None:
    """
    Validate a detection DataFrame against the required schema.
    
    Args:
        df: DataFrame to validate
        table: Table type ("raw" or "post")
        require_finite: If True, numeric columns must be finite (no NaN/inf)
    
    Raises:
        ValueError: If validation fails with actionable error message
    
    Notes:
        - schema_version must exist and equal SCHEMA_VERSION
        - phrase_type must be in ALLOWED_PHRASE_TYPES
        - All required columns must exist (even if DataFrame is empty)
        - Numeric columns must be numeric and finite (if require_finite=True)
        - is_present (post table) must be boolean or 0/1 integer
    """
    schema = RAW_SCHEMA_COLUMNS if table == "raw" else POST_SCHEMA_COLUMNS
    
    # Check required columns exist (always check, even if empty)
    missing_cols = set(schema.keys()) - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"Missing required columns for '{table}' table: {sorted(missing_cols)}. "
            f"Required: {list(schema.keys())}"
        )
    
    # Validate schema_version exists (always check, even if empty)
    if "schema_version" not in df.columns:
        raise ValueError("schema_version column is required")
    
    # Skip value checks if empty
    if df.empty:
        return
    
    if df["schema_version"].isna().any():
        raise ValueError("schema_version cannot contain null values")
    
    unique_versions = df["schema_version"].unique()
    if len(unique_versions) != 1:
        raise ValueError(
            f"Multiple schema versions found: {unique_versions}. "
            f"All rows must have the same schema_version."
        )
    
    actual_version = df["schema_version"].iloc[0]
    if actual_version != SCHEMA_VERSION:
        raise ValueError(
            f"Schema version mismatch: expected '{SCHEMA_VERSION}', got '{actual_version}'"
        )
    
    # Validate phrase_type values
    if "phrase_type" in df.columns:
        invalid_types = set(df["phrase_type"].unique()) - ALLOWED_PHRASE_TYPES
        if invalid_types:
            raise ValueError(
                f"Invalid phrase_type values: {sorted(invalid_types)}. "
                f"Must be one of: {sorted(ALLOWED_PHRASE_TYPES)}"
            )
    
    # Validate numeric columns are numeric and finite
    numeric_cols = ["conf_raw", "x1", "y1", "x2", "y2"]
    if table == "post":
        numeric_cols.extend(["noise_floor", "tau_eff", "conf_eff"])
    
    for col in numeric_cols:
        if col not in df.columns:
            continue
        
        # Check column is numeric type
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise ValueError(
                f"Column '{col}' must be numeric type, got {df[col].dtype}"
            )
        
        # Check for finite values if required
        if require_finite:
            values = df[col].to_numpy()
            if not np.isfinite(values).all():
                n_nan = np.isnan(values).sum()
                n_inf = np.isinf(values).sum()
                raise ValueError(
                    f"Column '{col}' contains non-finite values: "
                    f"{n_nan} NaN, {n_inf} inf. Set require_finite=False to allow."
                )
    
    # Validate integer columns
    if "entity_id" in df.columns:
        if not pd.api.types.is_integer_dtype(df["entity_id"]):
            raise ValueError(f"entity_id must be integer type, got {df['entity_id'].dtype}")
    
    # Validate boolean column (post table only)
    if table == "post" and "is_present" in df.columns:
        col_dtype = df["is_present"].dtype
        if pd.api.types.is_bool_dtype(col_dtype) or str(col_dtype) == "boolean":
            # Already boolean - valid
            pass
        elif pd.api.types.is_integer_dtype(col_dtype):
            # Integer - must be 0/1 only
            if not df["is_present"].isin([0, 1]).all():
                raise ValueError(
                    "is_present as integer must contain only 0 or 1 values"
                )
        else:
            raise ValueError(
                f"is_present must be boolean or 0/1 integer, got {col_dtype}"
            )


def coerce_detection_df(df: pd.DataFrame, table: DetectionTableName) -> pd.DataFrame:
    """
    Coerce a DataFrame to the detection schema with safe type conversions.
    
    Args:
        df: Input DataFrame
        table: Table type ("raw" or "post")
    
    Returns:
        Coerced DataFrame with:
        - schema_version added if missing
        - Columns reordered to canonical schema order
        - Extra columns removed (strict mode)
        - Dtypes coerced safely
    
    Raises:
        ValueError: If coercion fails for required columns
    
    Notes:
        - String columns: converted to pandas "string" dtype (NaN stays NaN)
        - Numeric columns: pd.to_numeric with errors="raise"
        - Boolean columns: strict coercion (bool or 0/1 int only)
        - Extra columns not in schema are dropped with warning
    """
    df = df.copy()
    
    # Add schema_version if missing
    if "schema_version" not in df.columns:
        df["schema_version"] = SCHEMA_VERSION
    
    schema = RAW_SCHEMA_COLUMNS if table == "raw" else POST_SCHEMA_COLUMNS
    
    # Check for extra columns and drop them
    extra_cols = set(df.columns) - set(schema.keys())
    if extra_cols:
        import warnings
        warnings.warn(
            f"Dropping extra columns not in '{table}' schema: {sorted(extra_cols)}",
            UserWarning
        )
        df = df[[col for col in df.columns if col in schema]]
    
    # Coerce dtypes with strict error handling
    for col, dtype_str in schema.items():
        if col not in df.columns:
            continue
        
        try:
            if dtype_str == "string":
                # Use pandas string dtype (keeps NaN as NaN, not "nan")
                df[col] = df[col].astype("string")
            
            elif dtype_str == "int64":
                # Convert to numeric, then nullable Int64
                df[col] = pd.to_numeric(df[col], errors="raise")
                df[col] = df[col].astype("Int64")
            
            elif dtype_str == "float64":
                # Convert to numeric
                if len(df) > 0:
                    df[col] = pd.to_numeric(df[col], errors="raise")
                else:
                    # For empty DataFrame, explicitly set to float64
                    df[col] = df[col].astype("float64")
            
            elif dtype_str == "bool":
                # Robust boolean coercion (accept bool, 0/1 int, 0.0/1.0 float, object with bool values)
                if pd.api.types.is_bool_dtype(df[col]) or str(df[col].dtype) == "boolean":
                    # Already boolean
                    df[col] = df[col].astype("boolean")
                elif pd.api.types.is_integer_dtype(df[col]):
                    # Integer: accept only 0/1
                    if len(df) > 0 and not df[col].isin([0, 1]).all():
                        raise ValueError(
                            f"Column '{col}' has non-boolean integers. "
                            f"Only 0 and 1 are allowed."
                        )
                    df[col] = df[col].astype("boolean")
                elif pd.api.types.is_float_dtype(df[col]):
                    # Float: accept only 0.0/1.0 (common Parquet round-trip)
                    if len(df) > 0 and not df[col].isin([0.0, 1.0]).all():
                        raise ValueError(
                            f"Column '{col}' has non-boolean floats. "
                            f"Only 0.0 and 1.0 are allowed."
                        )
                    df[col] = df[col].astype(int).astype("boolean")
                elif df[col].dtype == object:
                    # Object dtype: try to convert (common for mixed types)
                    # Accept True/False, 0/1, 0.0/1.0
                    unique_vals = set(df[col].dropna().unique())
                    valid_vals = {True, False, 0, 1, 0.0, 1.0}
                    if not unique_vals.issubset(valid_vals):
                        raise ValueError(
                            f"Column '{col}' has invalid boolean values: {unique_vals - valid_vals}"
                        )
                    # Convert to int then boolean
                    df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64").astype("boolean")
                else:
                    raise ValueError(
                        f"Column '{col}' cannot be coerced to boolean from {df[col].dtype}"
                    )
        
        except Exception as e:
            raise ValueError(
                f"Failed to coerce column '{col}' to {dtype_str}: {e}"
            ) from e
    
    # Reorder columns to canonical schema order
    df = df[list(schema.keys())]
    
    return df


# ============================================================================
# Phase 5 Day 3: Deterministic shard naming and discovery
# ============================================================================

def make_shard_filename(
    table: DetectionTableName,
    split: str,
    model_id: str,
    shard_idx: int,
    schema_version: str = SCHEMA_VERSION,
    worker_id: Optional[int] = None
) -> str:
    """
    Generate deterministic shard filename.
    
    Args:
        table: Table type ("raw" or "post")
        split: Dataset split (e.g., "train", "val", "test")
        model_id: Model identifier
        shard_idx: Shard index (0-based)
        schema_version: Schema version string
        worker_id: Optional worker ID for parallel processing (prevents race conditions)
    
    Returns:
        Filename string (not full path)
    
    Example:
        >>> make_shard_filename("raw", "train", "owlvit-base", 0)
        'raw__train__owlvit-base__v1__shard000000.parquet'
        >>> make_shard_filename("raw", "train", "owlvit-base", 0, worker_id=1)
        'raw__train__owlvit-base__v1__shard000000_w1.parquet'
    """
    base = f"{table}__{split}__{model_id}__{schema_version}__shard{shard_idx:06d}"
    if worker_id is not None:
        base += f"_w{worker_id}"
    return base + ".parquet"


def list_shards(
    out_dir: Path,
    table: DetectionTableName,
    split: str,
    model_id: str
) -> List[Path]:
    """
    List all shard files for a given table/split/model combination.
    
    Args:
        out_dir: Directory containing shards
        table: Table type ("raw" or "post")
        split: Dataset split
        model_id: Model identifier
    
    Returns:
        Sorted list of shard paths (by shard index, then lexicographic)
    """
    if not out_dir.exists():
        return []
    
    # Match pattern for this table/split/model
    pattern = f"{table}__{split}__{model_id}__*__shard*.parquet"
    shards = list(out_dir.glob(pattern))
    
    # Sort by shard index (using parse_shard_index), then lexicographic
    def sort_key(p: Path) -> tuple:
        idx = parse_shard_index(p)
        return (idx if idx is not None else float('inf'), p.name)
    
    return sorted(shards, key=sort_key)


# ============================================================================
# Phase 5 Day 3: File locking for concurrency safety
# ============================================================================

def _acquire_lock(lock_path: Path, max_tries: int = 50, sleep_sec: float = 0.05) -> int:
    """
    Acquire an exclusive file lock using atomic file creation.
    
    Args:
        lock_path: Path to lock file
        max_tries: Maximum number of attempts
        sleep_sec: Sleep duration between retries
    
    Returns:
        File descriptor for the lock file
    
    Raises:
        RuntimeError: If lock cannot be acquired after max_tries
    """
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    
    for attempt in range(max_tries):
        try:
            # Atomic create-exclusive
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_RDWR)
            return fd
        except FileExistsError:
            time.sleep(sleep_sec)
    
    raise RuntimeError(
        f"Could not acquire lock {lock_path.name} after {max_tries} attempts. "
        f"Another process may be holding the lock."
    )


def _release_lock(fd: int, lock_path: Path) -> None:
    """
    Release a file lock.
    
    Args:
        fd: File descriptor from _acquire_lock
        lock_path: Path to lock file
    """
    try:
        os.close(fd)
    except OSError:
        pass
    
    try:
        lock_path.unlink()
    except FileNotFoundError:
        pass


# ============================================================================
# Phase 5 Day 3: Parquet shard writer (resume-safe, atomic)
# ============================================================================

class ParquetShardWriter:
    """
    Resume-safe Parquet shard writer with atomic writes.
    
    Maintains an internal buffer and writes shards when buffer reaches
    shard_size_rows. Supports resuming from existing shards.
    """
    
    def __init__(
        self,
        out_dir: Path,
        table: DetectionTableName,
        split: str,
        model_id: str,
        shard_size_rows: int,
        force: bool = False,
        worker_id: Optional[int] = None,
        allow_mixed_schema: bool = False
    ):
        """
        Initialize shard writer.
        
        Args:
            out_dir: Output directory for shards
            table: Table type ("raw" or "post")
            split: Dataset split
            model_id: Model identifier
            shard_size_rows: Number of rows per shard
            force: If True, overwrite existing shards; if False, resume from last shard
            worker_id: Optional worker ID for parallel processing (prevents filename collisions)
            allow_mixed_schema: If False (default), fail on schema version mismatch; if True, only warn
        """
        self.out_dir = Path(out_dir)
        self.table = table
        self.split = split
        self.model_id = model_id
        self.shard_size_rows = shard_size_rows
        self.force = force
        self.worker_id = worker_id
        self.allow_mixed_schema = allow_mixed_schema
        
        self.out_dir.mkdir(parents=True, exist_ok=True)
        
        # Lock file for concurrency safety
        self.lock_path = self.out_dir / f"{table}__{split}__{model_id}__{SCHEMA_VERSION}.lock"
        
        # Check for schema version mismatches in existing shards
        self._check_existing_schema_versions()
        
        # Acquire lock for shard index allocation
        lock_fd = _acquire_lock(self.lock_path)
        try:
            # Determine starting shard index under lock
            if force:
                self.current_shard_idx = 0
                # Delete existing shards for this table/split/model
                existing = list_shards(self.out_dir, table, split, model_id)
                for shard_path in existing:
                    shard_path.unlink()
            else:
                # Resume from max existing shard index + 1 (using robust parser)
                existing = list_shards(self.out_dir, table, split, model_id)
                if existing:
                    indices = []
                    for path in existing:
                        idx = parse_shard_index(path)
                        if idx is not None:
                            indices.append(idx)
                    self.current_shard_idx = max(indices) + 1 if indices else 0
                else:
                    self.current_shard_idx = 0
        finally:
            _release_lock(lock_fd, self.lock_path)
        
        self.buffer: List[pd.DataFrame] = []
        self.buffer_rows = 0
    
    def _check_existing_schema_versions(self) -> None:
        """
        Check existing shards for schema version mismatches.
        
        By default (allow_mixed_schema=False), raises ValueError if any shard
        has a different schema version. If allow_mixed_schema=True, only warns.
        
        Raises:
            ValueError: If schema version mismatch found and allow_mixed_schema=False
        """
        existing = list_shards(self.out_dir, self.table, self.split, self.model_id)
        if not existing:
            return
        
        # Check all shards (or sample if too many) for schema version
        mismatched_versions = set()
        check_shards = existing if len(existing) <= 10 else existing[::max(1, len(existing) // 10)]
        
        for shard_path in check_shards:
            try:
                df_version = pd.read_parquet(shard_path, columns=["schema_version"], engine="pyarrow")
                if not df_version.empty:
                    shard_version = df_version["schema_version"].iloc[0]
                    if shard_version != SCHEMA_VERSION:
                        mismatched_versions.add(shard_version)
            except Exception:
                # If we can't read the shard, skip it
                continue
        
        if mismatched_versions:
            msg = (
                f"Existing shards have schema version(s) {mismatched_versions} "
                f"but code expects '{SCHEMA_VERSION}'. "
                f"Consider migrating or using force=True to overwrite."
            )
            
            if self.allow_mixed_schema:
                warnings.warn(msg, UserWarning, stacklevel=3)
            else:
                raise ValueError(
                    msg + " Set allow_mixed_schema=True to proceed anyway (not recommended)."
                )
    
    def append(self, df_rows: pd.DataFrame) -> None:
        """
        Append rows to buffer, flushing to disk when shard size is reached.
        
        Automatically coerces and validates data before buffering.
        
        Args:
            df_rows: DataFrame with rows to append
        
        Raises:
            ValueError: If validation fails
        
        Note:
            Empty DataFrames are silently ignored (no-op). This means empty
            shard markers are never written. If you need to mark "processed
            but found nothing", consider writing a metadata file separately.
        """
        if df_rows.empty:
            # No-op: we never write empty shards
            return
        
        # Coerce and validate before buffering
        df_rows = coerce_detection_df(df_rows, table=self.table)
        validate_detection_df(df_rows, table=self.table, require_finite=True)
        
        self.buffer.append(df_rows)
        self.buffer_rows += len(df_rows)
        
        # Flush if buffer exceeds shard size
        while self.buffer_rows >= self.shard_size_rows:
            self._write_shard()
    
    def flush(self) -> None:
        """Flush any remaining buffered rows to disk."""
        if self.buffer_rows > 0:
            self._write_shard()
    
    def close(self) -> None:
        """Close writer, flushing any remaining data."""
        self.flush()
    
    def _write_shard(self) -> None:
        """Write one shard from buffer with atomic write operation (with concurrency lock)."""
        if not self.buffer:
            return
        
        # Concatenate buffer
        df = pd.concat(self.buffer, ignore_index=True)
        
        # Determine how many rows to write
        rows_to_write = min(self.shard_size_rows, len(df))
        shard_df = df.iloc[:rows_to_write].copy()
        
        # Keep remaining rows in buffer
        if len(df) > rows_to_write:
            self.buffer = [df.iloc[rows_to_write:]]
            self.buffer_rows = len(df) - rows_to_write
        else:
            self.buffer = []
            self.buffer_rows = 0
        
        # Sort for determinism (conf_raw descending = highest confidence first)
        # Primary sort: image_id, then conf_raw (desc), then stable secondary keys
        sort_cols = ["image_id"]
        sort_ascending = [True]
        
        # Add confidence as primary sort within image (highest first)
        if "conf_raw" in shard_df.columns:
            sort_cols.append("conf_raw")
            sort_ascending.append(False)  # Descending (highest first)
        
        # For post table, also sort by effective confidence
        if self.table == "post" and "conf_eff" in shard_df.columns:
            sort_cols.append("conf_eff")
            sort_ascending.append(False)  # Descending
        
        # Add stable secondary keys for determinism
        sort_cols.extend(["entity_id", "phrase_type", "x1", "y1", "x2", "y2"])
        sort_ascending.extend([True, True, True, True, True, True])
        
        shard_df = shard_df.sort_values(
            by=sort_cols,
            ascending=sort_ascending
        ).reset_index(drop=True)
        
        # Acquire lock for shard writing (prevents concurrent overwrites)
        lock_fd = _acquire_lock(self.lock_path)
        try:
            # Generate filename
            filename = make_shard_filename(
                self.table,
                self.split,
                self.model_id,
                self.current_shard_idx,
                worker_id=self.worker_id
            )
            final_path = self.out_dir / filename
            temp_path = final_path.with_suffix(".parquet.tmp")
            
            # Write to temp file
            try:
                shard_df.to_parquet(temp_path, engine="pyarrow", compression="snappy", index=False)
            except ImportError:
                raise ImportError(
                    "pyarrow is required for Parquet writing. "
                    "Install with: pip install pyarrow"
                )
            
            # Atomic rename
            temp_path.replace(final_path)
            
            self.current_shard_idx += 1
        finally:
            _release_lock(lock_fd, self.lock_path)


# ============================================================================
# Phase 5 Day 3: Parquet shard reader
# ============================================================================

def iter_parquet_shards(
    out_dir: Path,
    table: DetectionTableName,
    split: str,
    model_id: str,
    columns: Optional[List[str]] = None,
    validate: bool = True
) -> Iterator[pd.DataFrame]:
    """
    Iterate over Parquet shards, yielding DataFrames.
    
    Args:
        out_dir: Directory containing shards
        table: Table type ("raw" or "post")
        split: Dataset split
        model_id: Model identifier
        columns: Optional list of columns to read (None = all columns)
        validate: If True, validate each shard against schema (default: True)
    
    Yields:
        DataFrame for each shard
    
    Raises:
        ImportError: If pyarrow is not installed
        ValueError: If validation fails (when validate=True)
    """
    shards = list_shards(out_dir, table, split, model_id)
    
    for shard_path in shards:
        try:
            df = pd.read_parquet(shard_path, columns=columns, engine="pyarrow")
        except ImportError as e:
            raise ImportError(
                "pyarrow is required for Parquet reading. "
                "Install with: pip install pyarrow"
            ) from e
        
        # Validate shard if requested
        if validate:
            try:
                validate_detection_df(df, table=table, require_finite=True)
            except ValueError as e:
                raise ValueError(
                    f"Shard validation failed for {shard_path.name}: {e}"
                ) from e
        
        yield df


def read_parquet_table(
    out_dir: Path,
    table: DetectionTableName,
    split: str,
    model_id: str,
    columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Read all shards for a table into a single DataFrame.
    
    WARNING: This loads all data into memory. For large datasets,
    use iter_parquet_shards() instead.
    
    Args:
        out_dir: Directory containing shards
        table: Table type ("raw" or "post")
        split: Dataset split
        model_id: Model identifier
        columns: Optional list of columns to read
    
    Returns:
        Concatenated DataFrame
    """
    dfs = list(iter_parquet_shards(out_dir, table, split, model_id, columns))
    if not dfs:
        # Return empty DataFrame with correct schema and dtypes
        schema = RAW_SCHEMA_COLUMNS if table == "raw" else POST_SCHEMA_COLUMNS
        empty_df = pd.DataFrame(columns=list(schema.keys()))
        # Coerce to ensure correct dtypes (not object)
        return coerce_detection_df(empty_df, table=table)
    return pd.concat(dfs, ignore_index=True)
