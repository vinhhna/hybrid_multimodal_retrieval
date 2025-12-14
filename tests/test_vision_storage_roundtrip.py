"""
Round-trip test for detection storage (Phase 5 Day 3).

Tests that raw and postprocessed detections can be written to Parquet shards
and read back with full integrity, including schema_version validation.
"""

import tempfile
import warnings
from pathlib import Path

import pandas as pd
import pytest

from src.vision.storage import (
    SCHEMA_VERSION,
    RAW_SCHEMA_COLUMNS,
    POST_SCHEMA_COLUMNS,
    ParquetShardWriter,
    iter_parquet_shards,
    read_parquet_table,
    validate_detection_df,
    coerce_detection_df,
    make_shard_filename,
    list_shards,
)


def test_raw_detections_roundtrip():
    """Test raw detections write and read with small shards."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir)
        
        # Create test data (5 rows, shard_size=2 -> 3 shards)
        df_original = pd.DataFrame({
            "schema_version": [SCHEMA_VERSION] * 5,
            "image_id": ["img_001", "img_002", "img_003", "img_001", "img_002"],
            "entity_id": [1, 2, 3, 4, 5],
            "phrase_type": ["object", "attribute", "composed", "object", "null"],
            "model_id": ["owlvit-base"] * 5,
            "conf_raw": [0.9, 0.8, 0.7, 0.6, 0.5],
            "x1": [10.0, 20.0, 30.0, 40.0, 50.0],
            "y1": [15.0, 25.0, 35.0, 45.0, 55.0],
            "x2": [50.0, 60.0, 70.0, 80.0, 90.0],
            "y2": [55.0, 65.0, 75.0, 85.0, 95.0],
        })
        
        # Validate before writing
        validate_detection_df(df_original, table="raw")
        
        # Write shards
        writer = ParquetShardWriter(
            out_dir=out_dir,
            table="raw",
            split="test",
            model_id="owlvit-base",
            shard_size_rows=2,
            force=False
        )
        writer.append(df_original)
        writer.close()
        
        # Check shards exist
        shards = list_shards(out_dir, "raw", "test", "owlvit-base")
        assert len(shards) >= 2, f"Expected at least 2 shards, got {len(shards)}"
        
        # Check filenames are deterministic
        for i, shard_path in enumerate(shards):
            expected_name = make_shard_filename("raw", "test", "owlvit-base", i)
            assert shard_path.name == expected_name, f"Shard {i} has wrong name: {shard_path.name}"
        
        # Read back using iterator
        dfs_iter = list(iter_parquet_shards(out_dir, "raw", "test", "owlvit-base"))
        df_read_iter = pd.concat(dfs_iter, ignore_index=True)
        
        # Validate read data
        validate_detection_df(df_read_iter, table="raw")
        
        # Check schema_version exists and is correct
        assert "schema_version" in df_read_iter.columns, "schema_version column missing"
        assert (df_read_iter["schema_version"] == SCHEMA_VERSION).all(), "schema_version mismatch"
        
        # Check row count (may differ due to sorting/deduplication)
        assert len(df_read_iter) == len(df_original), f"Row count mismatch: {len(df_read_iter)} vs {len(df_original)}"
        
        # Check key columns exist
        for col in ["image_id", "entity_id", "phrase_type", "conf_raw", "x1", "y1", "x2", "y2"]:
            assert col in df_read_iter.columns, f"Column {col} missing"
        
        # Read back using convenience function
        df_read_full = read_parquet_table(out_dir, "raw", "test", "owlvit-base")
        assert len(df_read_full) == len(df_original), "Full table read row count mismatch"


def test_postprocessed_detections_roundtrip():
    """Test postprocessed detections write and read."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir)
        
        # Create test data with postprocessed columns
        df_original = pd.DataFrame({
            # Raw columns
            "schema_version": [SCHEMA_VERSION] * 3,
            "image_id": ["img_001", "img_002", "img_003"],
            "entity_id": [1, 2, 3],
            "phrase_type": ["object", "attribute", "composed"],
            "model_id": ["owlvit-base"] * 3,
            "conf_raw": [0.9, 0.8, 0.7],
            "x1": [10.0, 20.0, 30.0],
            "y1": [15.0, 25.0, 35.0],
            "x2": [50.0, 60.0, 70.0],
            "y2": [55.0, 65.0, 75.0],
            # Postprocessed columns
            "noise_floor": [0.1, 0.15, 0.2],
            "tau_eff": [0.3, 0.35, 0.4],
            "conf_eff": [0.8, 0.65, 0.5],
            "is_present": [True, True, False],
        })
        
        # Validate before writing
        validate_detection_df(df_original, table="post")
        
        # Write shards
        writer = ParquetShardWriter(
            out_dir=out_dir,
            table="post",
            split="test",
            model_id="owlvit-base",
            shard_size_rows=2,
            force=False
        )
        writer.append(df_original)
        writer.close()
        
        # Read back
        df_read = read_parquet_table(out_dir, "post", "test", "owlvit-base")
        
        # Validate
        validate_detection_df(df_read, table="post")
        
        # Check schema_version
        assert "schema_version" in df_read.columns
        assert (df_read["schema_version"] == SCHEMA_VERSION).all()
        
        # Check postprocessed columns exist
        for col in ["noise_floor", "tau_eff", "conf_eff", "is_present"]:
            assert col in df_read.columns, f"Postprocessed column {col} missing"
        
        # Check row count
        assert len(df_read) == len(df_original)


def test_coerce_adds_schema_version():
    """Test that coerce_detection_df adds schema_version if missing."""
    df_no_version = pd.DataFrame({
        "image_id": ["img_001"],
        "entity_id": [1],
        "phrase_type": ["object"],
        "model_id": ["owlvit-base"],
        "conf_raw": [0.9],
        "x1": [10.0],
        "y1": [15.0],
        "x2": [50.0],
        "y2": [55.0],
    })
    
    # Coerce should add schema_version
    df_coerced = coerce_detection_df(df_no_version, table="raw")
    assert "schema_version" in df_coerced.columns
    assert (df_coerced["schema_version"] == SCHEMA_VERSION).all()
    
    # Should also reorder columns to canonical order
    expected_cols = list(RAW_SCHEMA_COLUMNS.keys())
    assert list(df_coerced.columns) == expected_cols


def test_phrase_type_validation():
    """Test that invalid phrase_type values are rejected."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir)
        
        # Create data with invalid phrase_type
        df_invalid = pd.DataFrame({
            "schema_version": [SCHEMA_VERSION] * 2,
            "image_id": ["img_001", "img_002"],
            "entity_id": [1, 2],
            "phrase_type": ["object", "INVALID_TYPE"],  # Invalid!
            "model_id": ["owlvit-base"] * 2,
            "conf_raw": [0.9, 0.8],
            "x1": [10.0, 20.0],
            "y1": [15.0, 25.0],
            "x2": [50.0, 60.0],
            "y2": [55.0, 65.0],
        })
        
        # Validation should fail
        with pytest.raises(ValueError, match="Invalid phrase_type values"):
            validate_detection_df(df_invalid, table="raw")


def test_resume_safe_writing():
    """Test that writer resumes from existing shards."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir)
        
        # Write first batch
        df_batch1 = pd.DataFrame({
            "schema_version": [SCHEMA_VERSION] * 2,
            "image_id": ["img_001", "img_002"],
            "entity_id": [1, 2],
            "phrase_type": ["object", "attribute"],
            "model_id": ["owlvit-base"] * 2,
            "conf_raw": [0.9, 0.8],
            "x1": [10.0, 20.0],
            "y1": [15.0, 25.0],
            "x2": [50.0, 60.0],
            "y2": [55.0, 65.0],
        })
        
        writer1 = ParquetShardWriter(
            out_dir=out_dir,
            table="raw",
            split="test",
            model_id="owlvit-base",
            shard_size_rows=2,
            force=False
        )
        writer1.append(df_batch1)
        writer1.close()
        
        # Check one shard exists
        shards1 = list_shards(out_dir, "raw", "test", "owlvit-base")
        assert len(shards1) == 1
        
        # Write second batch (should create shard 1, not overwrite shard 0)
        df_batch2 = pd.DataFrame({
            "schema_version": [SCHEMA_VERSION] * 2,
            "image_id": ["img_003", "img_004"],
            "entity_id": [3, 4],
            "phrase_type": ["composed", "null"],
            "model_id": ["owlvit-base"] * 2,
            "conf_raw": [0.7, 0.6],
            "x1": [30.0, 40.0],
            "y1": [35.0, 45.0],
            "x2": [70.0, 80.0],
            "y2": [75.0, 85.0],
        })
        
        writer2 = ParquetShardWriter(
            out_dir=out_dir,
            table="raw",
            split="test",
            model_id="owlvit-base",
            shard_size_rows=2,
            force=False  # Resume-safe
        )
        writer2.append(df_batch2)
        writer2.close()
        
        # Check two shards exist
        shards2 = list_shards(out_dir, "raw", "test", "owlvit-base")
        assert len(shards2) == 2
        
        # Read all data
        df_all = read_parquet_table(out_dir, "raw", "test", "owlvit-base")
        assert len(df_all) == 4  # 2 + 2


def test_force_overwrite():
    """Test that force=True overwrites existing shards."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir)
        
        # Write first batch
        df_batch1 = pd.DataFrame({
            "schema_version": [SCHEMA_VERSION] * 2,
            "image_id": ["img_001", "img_002"],
            "entity_id": [1, 2],
            "phrase_type": ["object", "attribute"],
            "model_id": ["owlvit-base"] * 2,
            "conf_raw": [0.9, 0.8],
            "x1": [10.0, 20.0],
            "y1": [15.0, 25.0],
            "x2": [50.0, 60.0],
            "y2": [55.0, 65.0],
        })
        
        writer1 = ParquetShardWriter(
            out_dir=out_dir,
            table="raw",
            split="test",
            model_id="owlvit-base",
            shard_size_rows=2,
            force=False
        )
        writer1.append(df_batch1)
        writer1.close()
        
        # Write second batch with force=True (should overwrite)
        df_batch2 = pd.DataFrame({
            "schema_version": [SCHEMA_VERSION] * 3,
            "image_id": ["img_003", "img_004", "img_005"],
            "entity_id": [3, 4, 5],
            "phrase_type": ["composed", "null", "object"],
            "model_id": ["owlvit-base"] * 3,
            "conf_raw": [0.7, 0.6, 0.5],
            "x1": [30.0, 40.0, 50.0],
            "y1": [35.0, 45.0, 55.0],
            "x2": [70.0, 80.0, 90.0],
            "y2": [75.0, 85.0, 95.0],
        })
        
        writer2 = ParquetShardWriter(
            out_dir=out_dir,
            table="raw",
            split="test",
            model_id="owlvit-base",
            shard_size_rows=2,
            force=True  # Overwrite
        )
        writer2.append(df_batch2)
        writer2.close()
        
        # Read all data (should only have batch2 data)
        df_all = read_parquet_table(out_dir, "raw", "test", "owlvit-base")
        assert len(df_all) == 3  # Only batch2
        assert set(df_all["entity_id"].values) == {3, 4, 5}


def test_coerce_drops_extra_columns():
    """Test that coerce drops columns not in schema."""
    import warnings
    
    df_with_extra = pd.DataFrame({
        "schema_version": [SCHEMA_VERSION],
        "image_id": ["img_001"],
        "entity_id": [1],
        "phrase_type": ["object"],
        "model_id": ["owlvit-base"],
        "conf_raw": [0.9],
        "x1": [10.0],
        "y1": [15.0],
        "x2": [50.0],
        "y2": [55.0],
        "extra_col1": ["foo"],
        "extra_col2": [123],
    })
    
    # Coerce should drop extra columns with warning
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        df_coerced = coerce_detection_df(df_with_extra, table="raw")
        
        # Check warning was raised
        assert len(w) == 1
        assert "extra_col1" in str(w[0].message)
        assert "extra_col2" in str(w[0].message)
    
    # Extra columns should be gone
    assert "extra_col1" not in df_coerced.columns
    assert "extra_col2" not in df_coerced.columns
    
    # All schema columns should remain
    for col in RAW_SCHEMA_COLUMNS.keys():
        assert col in df_coerced.columns


def test_validation_rejects_nan():
    """Test that validation rejects NaN in numeric columns by default."""
    df_with_nan = pd.DataFrame({
        "schema_version": [SCHEMA_VERSION, SCHEMA_VERSION],
        "image_id": ["img_001", "img_002"],
        "entity_id": [1, 2],
        "phrase_type": ["object", "object"],
        "model_id": ["owlvit-base", "owlvit-base"],
        "conf_raw": [0.9, float('nan')],  # NaN in second row
        "x1": [10.0, 20.0],
        "y1": [15.0, 25.0],
        "x2": [50.0, 60.0],
        "y2": [55.0, 65.0],
    })
    
    # Validation should fail with require_finite=True (default)
    with pytest.raises(ValueError, match="non-finite"):
        validate_detection_df(df_with_nan, table="raw", require_finite=True)
    
    # Should pass with require_finite=False
    validate_detection_df(df_with_nan, table="raw", require_finite=False)


def test_validation_enforces_schema_version():
    """Test that validation enforces correct schema_version."""
    df_wrong_version = pd.DataFrame({
        "schema_version": ["v0", "v0"],  # Wrong version
        "image_id": ["img_001", "img_002"],
        "entity_id": [1, 2],
        "phrase_type": ["object", "object"],
        "model_id": ["owlvit-base", "owlvit-base"],
        "conf_raw": [0.9, 0.8],
        "x1": [10.0, 20.0],
        "y1": [15.0, 25.0],
        "x2": [50.0, 60.0],
        "y2": [55.0, 65.0],
    })
    
    # Should fail with version mismatch
    with pytest.raises(ValueError, match="Schema version mismatch"):
        validate_detection_df(df_wrong_version, table="raw")


def test_validation_checks_empty_dataframes():
    """Test that validation checks columns even for empty DataFrames."""
    # Empty DataFrame with missing columns
    df_empty_wrong = pd.DataFrame({
        "image_id": [],
        "entity_id": [],
    })
    
    # Should fail due to missing columns
    with pytest.raises(ValueError, match="Missing required columns"):
        validate_detection_df(df_empty_wrong, table="raw")
    
    # Empty DataFrame with all columns should pass
    df_empty_correct = pd.DataFrame(columns=list(RAW_SCHEMA_COLUMNS.keys()))
    df_empty_correct = coerce_detection_df(df_empty_correct, table="raw")
    validate_detection_df(df_empty_correct, table="raw")  # Should not raise


def test_boolean_coercion_accepts_floats():
    """Test that boolean coercion accepts 0.0/1.0 floats."""
    df_with_float_bools = pd.DataFrame({
        "schema_version": [SCHEMA_VERSION] * 3,
        "image_id": ["img_001", "img_002", "img_003"],
        "entity_id": [1, 2, 3],
        "phrase_type": ["object", "attribute", "composed"],
        "model_id": ["owlvit-base"] * 3,
        "conf_raw": [0.9, 0.8, 0.7],
        "x1": [10.0, 20.0, 30.0],
        "y1": [15.0, 25.0, 35.0],
        "x2": [50.0, 60.0, 70.0],
        "y2": [55.0, 65.0, 75.0],
        "noise_floor": [0.1, 0.15, 0.2],
        "tau_eff": [0.3, 0.35, 0.4],
        "conf_eff": [0.8, 0.65, 0.5],
        "is_present": [1.0, 0.0, 1.0],  # Float 0.0/1.0
    })
    
    # Should successfully coerce float bools to boolean
    df_coerced = coerce_detection_df(df_with_float_bools, table="post")
    assert pd.api.types.is_bool_dtype(df_coerced["is_present"]) or str(df_coerced["is_present"].dtype) == "boolean"
    
    # Should validate
    validate_detection_df(df_coerced, table="post")


def test_worker_id_in_filename():
    """Test that worker_id is included in shard filename."""
    from src.vision.storage import make_shard_filename
    
    # Without worker_id
    filename1 = make_shard_filename("raw", "train", "owlvit-base", 0)
    assert filename1 == "raw__train__owlvit-base__v1__shard000000.parquet"
    
    # With worker_id
    filename2 = make_shard_filename("raw", "train", "owlvit-base", 0, worker_id=1)
    assert filename2 == "raw__train__owlvit-base__v1__shard000000_w1.parquet"
    
    filename3 = make_shard_filename("raw", "train", "owlvit-base", 5, worker_id=3)
    assert filename3 == "raw__train__owlvit-base__v1__shard000005_w3.parquet"


def test_sorting_conf_raw_descending():
    """Test that shards are sorted with conf_raw descending (highest first)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir)
        
        # Create data with varying confidence (enough for one shard)
        df = pd.DataFrame({
            "schema_version": [SCHEMA_VERSION] * 5,
            "image_id": ["img_001"] * 5,
            "entity_id": [1, 2, 3, 4, 5],
            "phrase_type": ["object"] * 5,
            "model_id": ["owlvit-base"] * 5,
            "conf_raw": [0.5, 0.9, 0.3, 0.8, 0.6],  # Unsorted
            "x1": [10.0] * 5,
            "y1": [15.0] * 5,
            "x2": [50.0] * 5,
            "y2": [55.0] * 5,
        })
        
        # Write shards (use shard_size >= data size for single shard)
        writer = ParquetShardWriter(
            out_dir=out_dir,
            table="raw",
            split="test",
            model_id="owlvit-base",
            shard_size_rows=10,  # Large enough for all data in one shard
            force=True
        )
        writer.append(df)
        writer.close()
        
        # Read back
        df_read = read_parquet_table(out_dir, "raw", "test", "owlvit-base")
        
        # Check that conf_raw is sorted descending within same image
        conf_values = df_read["conf_raw"].tolist()
        assert conf_values == [0.9, 0.8, 0.6, 0.5, 0.3], f"Expected descending order, got {conf_values}"


def test_empty_read_has_correct_dtypes():
    """Test that reading from empty directory returns DataFrame with correct dtypes."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir)
        
        # Read from empty directory
        df_empty = read_parquet_table(out_dir, "raw", "test", "owlvit-base")
        
        # Should have all columns
        assert set(df_empty.columns) == set(RAW_SCHEMA_COLUMNS.keys())
        
        # Should have correct dtypes (not object)
        assert str(df_empty["schema_version"].dtype) == "string"
        assert pd.api.types.is_integer_dtype(df_empty["entity_id"])
        assert pd.api.types.is_float_dtype(df_empty["conf_raw"])


def test_legacy_functions_still_work():
    """Test that legacy functions are deprecated but still functional."""
    from src.vision.storage import raw_shard_path, write_raw_shard, iter_raw_shards
    import warnings
    
    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir)
        
        # raw_shard_path should return legacy-named path with deprecation warning
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            path = raw_shard_path(out_dir, 0, "owlvit-base")
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert path.name == "raw_detections_owlvit-base_shard_00000.parquet"
        
        # write_raw_shard should work with deprecation warning
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            write_raw_shard(path, [])
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert path.exists()
        
        # iter_raw_shards should find the legacy-named shard
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            shards = iter_raw_shards(out_dir, "owlvit-base")
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert len(shards) == 1
            assert shards[0] == path


def test_schema_version_enforcement():
    """Test that schema version mismatch raises ValueError by default."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir)
        
        # Write a shard with correct schema
        writer1 = ParquetShardWriter(
            out_dir, "raw", "test", "owlvit-base", 
            shard_size_rows=10, force=True
        )
        df1 = pd.DataFrame({
            "schema_version": ["v1"],
            "image_id": ["img_001"],
            "entity_id": [1],
            "phrase_type": ["object"],
            "model_id": ["owlvit-base"],
            "conf_raw": [0.9],
            "x1": [10.0], "y1": [15.0], "x2": [50.0], "y2": [55.0],
        })
        writer1.append(df1)
        writer1.close()
        
        # Manually create a shard with wrong schema version
        df2 = df1.copy()
        df2["schema_version"] = "v0"  # Wrong version
        wrong_path = out_dir / "raw__test__owlvit-base__v1__shard000001.parquet"
        df2.to_parquet(wrong_path, engine="pyarrow", index=False)
        
        # Try to create writer with default allow_mixed_schema=False - should fail
        with pytest.raises(ValueError, match="schema version"):
            ParquetShardWriter(
                out_dir, "raw", "test", "owlvit-base",
                shard_size_rows=10, force=False
            )
        
        # With allow_mixed_schema=True, should only warn
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            writer3 = ParquetShardWriter(
                out_dir, "raw", "test", "owlvit-base",
                shard_size_rows=10, force=False, allow_mixed_schema=True
            )
            assert len(w) > 0
            assert any("schema version" in str(warning.message) for warning in w)
            writer3.close()


def test_boolean_stored_as_float():
    """Test that is_present stored as float 0.0/1.0 passes validation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir)
        
        # Create post table DataFrame with is_present as float
        df = pd.DataFrame({
            "schema_version": [SCHEMA_VERSION] * 2,
            "image_id": ["img_001", "img_001"],
            "entity_id": [1, 2],
            "phrase_type": ["object", "object"],
            "model_id": ["owlvit-base", "owlvit-base"],
            "conf_raw": [0.9, 0.5],
            "x1": [10.0, 20.0], "y1": [15.0, 25.0],
            "x2": [50.0, 60.0], "y2": [55.0, 65.0],
            "noise_floor": [0.1, 0.1],
            "tau_eff": [0.3, 0.3],
            "conf_eff": [0.8, 0.4],
            "is_present": [1.0, 0.0],  # Floats instead of bools
        })
        
        # Write with writer
        writer = ParquetShardWriter(
            out_dir, "post", "test", "owlvit-base",
            shard_size_rows=10, force=True
        )
        writer.append(df)
        writer.close()
        
        # Read back with validation - should pass
        df_read = read_parquet_table(out_dir, "post", "test", "owlvit-base")
        assert len(df_read) == 2
        # Should be coerced to boolean dtype
        assert pd.api.types.is_bool_dtype(df_read["is_present"]) or str(df_read["is_present"].dtype) == "boolean"


def test_parse_shard_index_with_worker_suffix():
    """Test that parse_shard_index handles worker suffixes correctly."""
    from src.vision.storage import parse_shard_index
    
    # New naming
    assert parse_shard_index(Path("raw__train__owlvit__v1__shard000042.parquet")) == 42
    assert parse_shard_index(Path("raw__train__owlvit__v1__shard000042_w1.parquet")) == 42
    assert parse_shard_index(Path("post__val__dino__v1__shard000123_w5.parquet")) == 123
    
    # Legacy naming
    assert parse_shard_index(Path("raw_detections_owlvit_shard_00042.parquet")) == 42
    
    # Invalid
    assert parse_shard_index(Path("random_file.parquet")) is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
