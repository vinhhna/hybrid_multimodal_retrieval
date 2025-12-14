"""
Unit tests for entity index and neighbor precomputation (Phase 5 Day 4).

Tests:
  - compute_mutual_knn_mask with toy data (no FAISS required)
  - compute_semantic_neighbors with FAISS (skipped if faiss not installed)
  - save/load roundtrip for neighbor artifacts
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pytest

from src.graph.entity_index import (
    normalize_rows,
    compute_mutual_knn_mask,
    load_neighbor_artifacts,
)


# ============================================================================
# Test normalize_rows
# ============================================================================

def test_normalize_rows_shape():
    """Test that normalize_rows preserves shape."""
    x = np.random.randn(10, 5).astype(np.float32)
    x_norm = normalize_rows(x)
    assert x_norm.shape == x.shape


def test_normalize_rows_unit_norm():
    """Test that normalized rows have unit L2 norm."""
    x = np.random.randn(10, 5).astype(np.float32)
    x_norm = normalize_rows(x)
    norms = np.linalg.norm(x_norm, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-6)


# ============================================================================
# Test compute_mutual_knn_mask (no FAISS required)
# ============================================================================

def test_compute_mutual_knn_mask_toy():
    """Test mutual k-NN mask computation with hand-constructed toy data."""
    # Construct a simple neighbor graph:
    #   Entity 0: neighbors = [1, 2]
    #   Entity 1: neighbors = [0, 2]
    #   Entity 2: neighbors = [1, 3]
    #   Entity 3: neighbors = [2, 0]
    #
    # Expected mutual flags:
    #   mutual[0, 0] = 1 (0 -> 1: is 0 in neighbors[1]=[0,2]? YES)
    #   mutual[0, 1] = 0 (0 -> 2: is 0 in neighbors[2]=[1,3]? NO)
    #   mutual[1, 0] = 1 (1 -> 0: is 1 in neighbors[0]=[1,2]? YES)
    #   mutual[1, 1] = 1 (1 -> 2: is 1 in neighbors[2]=[1,3]? YES)
    #   mutual[2, 0] = 1 (2 -> 1: is 2 in neighbors[1]=[0,2]? YES)
    #   mutual[2, 1] = 1 (2 -> 3: is 2 in neighbors[3]=[2,0]? YES)
    #   mutual[3, 0] = 1 (3 -> 2: is 3 in neighbors[2]=[1,3]? YES)
    #   mutual[3, 1] = 0 (3 -> 0: is 3 in neighbors[0]=[1,2]? NO)
    
    neighbors = np.array([
        [1, 2],
        [0, 2],
        [1, 3],
        [2, 0],
    ], dtype=np.int32)
    
    mutual = compute_mutual_knn_mask(neighbors)
    
    assert mutual.shape == neighbors.shape
    assert mutual.dtype == np.bool_, f"Expected bool dtype, got {mutual.dtype}"
    
    expected = np.array([
        [1, 0],
        [1, 1],
        [1, 1],
        [1, 0],
    ], dtype=np.bool_)
    
    np.testing.assert_array_equal(mutual, expected)


def test_compute_mutual_knn_mask_all_mutual():
    """Test mutual mask when all neighbors are bidirectional."""
    # Construct symmetric neighbors (all mutual):
    #   Entity 0: neighbors = [1]
    #   Entity 1: neighbors = [0]
    neighbors = np.array([
        [1],
        [0],
    ], dtype=np.int32)
    
    mutual = compute_mutual_knn_mask(neighbors)
    
    assert mutual.dtype == np.bool_
    
    expected = np.array([
        [1],
        [1],
    ], dtype=np.bool_)
    
    np.testing.assert_array_equal(mutual, expected)


def test_compute_mutual_knn_mask_no_mutual():
    """Test mutual mask when no neighbors are bidirectional."""
    # Construct asymmetric neighbors (no mutual):
    #   Entity 0: neighbors = [1, 2]
    #   Entity 1: neighbors = [2, 3]
    #   Entity 2: neighbors = [3, 0]
    #   Entity 3: neighbors = [0, 1]
    neighbors = np.array([
        [1, 2],
        [2, 3],
        [3, 0],
        [0, 1],
    ], dtype=np.int32)
    
    mutual = compute_mutual_knn_mask(neighbors)
    
    assert mutual.dtype == np.bool_
    
    # Check for actual mutual edges:
    # 0 -> 1: is 0 in neighbors[1]=[2,3]? NO -> 0
    # 0 -> 2: is 0 in neighbors[2]=[3,0]? YES -> 1
    # 1 -> 2: is 1 in neighbors[2]=[3,0]? NO -> 0
    # 1 -> 3: is 1 in neighbors[3]=[0,1]? YES -> 1
    # 2 -> 3: is 2 in neighbors[3]=[0,1]? NO -> 0
    # 2 -> 0: is 2 in neighbors[0]=[1,2]? YES -> 1
    # 3 -> 0: is 3 in neighbors[0]=[1,2]? NO -> 0
    # 3 -> 1: is 3 in neighbors[1]=[2,3]? YES -> 1
    
    expected = np.array([
        [0, 1],  # 0 -> 1 (no), 0 -> 2 (yes)
        [0, 1],  # 1 -> 2 (no), 1 -> 3 (yes)
        [0, 1],  # 2 -> 3 (no), 2 -> 0 (yes)
        [0, 1],  # 3 -> 0 (no), 3 -> 1 (yes)
    ], dtype=np.bool_)
    
    np.testing.assert_array_equal(mutual, expected)


# ============================================================================
# Test compute_semantic_neighbors (requires FAISS)
# ============================================================================

def test_compute_neighbors_shape_and_self_exclusion():
    """Test that compute_semantic_neighbors returns correct shape and excludes self."""
    pytest.importorskip("faiss")
    
    from src.graph.entity_index import build_faiss_ip_index, compute_semantic_neighbors
    
    # Create small random embeddings
    n = 50
    d = 512
    k = 5
    
    x = np.random.randn(n, d).astype(np.float32)
    x_norm = normalize_rows(x)
    
    # Build index
    index = build_faiss_ip_index(x_norm)
    
    # Compute neighbors
    neighbors = compute_semantic_neighbors(index, x_norm, k=k, batch_size=10)
    
    # Check shape
    assert neighbors.shape == (n, k)
    assert neighbors.dtype == np.int32
    
    # Check self-exclusion: no row should contain its own index
    for i in range(n):
        assert i not in neighbors[i], f"Entity {i} contains self in neighbors"
    
    # Check all neighbor IDs are in valid range
    assert neighbors.min() >= 0
    assert neighbors.max() < n


def test_compute_neighbors_deterministic():
    """Test that neighbor computation is deterministic."""
    pytest.importorskip("faiss")
    
    from src.graph.entity_index import build_faiss_ip_index, compute_semantic_neighbors
    
    n = 30
    d = 512
    k = 5
    
    x = np.random.randn(n, d).astype(np.float32)
    x_norm = normalize_rows(x)
    
    index = build_faiss_ip_index(x_norm)
    
    neighbors1 = compute_semantic_neighbors(index, x_norm, k=k, batch_size=10)
    neighbors2 = compute_semantic_neighbors(index, x_norm, k=k, batch_size=10)
    
    np.testing.assert_array_equal(neighbors1, neighbors2)


# ============================================================================
# Test save/load roundtrip
# ============================================================================

def test_save_load_neighbor_arrays_roundtrip(tmp_path):
    """Test saving and loading neighbor arrays."""
    # Create toy arrays
    neighbors = np.array([
        [1, 2, 3],
        [0, 2, 3],
        [0, 1, 3],
        [0, 1, 2],
    ], dtype=np.int32)
    
    mutual = np.array([
        [1, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
        [1, 1, 1],
    ], dtype=np.uint8)
    
    neighbors_path = tmp_path / "neighbors.npy"
    mutual_path = tmp_path / "mutual.npy"
    
    # Save
    np.save(neighbors_path, neighbors)
    np.save(mutual_path, mutual)
    
    # Load
    neighbors_loaded, mutual_loaded = load_neighbor_artifacts(neighbors_path, mutual_path)
    
    # Check
    np.testing.assert_array_equal(neighbors_loaded, neighbors)
    np.testing.assert_array_equal(mutual_loaded, mutual)


def test_load_neighbor_artifacts_missing_file(tmp_path):
    """Test that load_neighbor_artifacts raises FileNotFoundError for missing files."""
    neighbors_path = tmp_path / "missing_neighbors.npy"
    mutual_path = tmp_path / "missing_mutual.npy"
    
    with pytest.raises(FileNotFoundError):
        load_neighbor_artifacts(neighbors_path, mutual_path)


# ============================================================================
# Test FAISS index save/load (requires FAISS)
# ============================================================================

def test_faiss_index_save_load_roundtrip(tmp_path):
    """Test saving and loading FAISS index."""
    pytest.importorskip("faiss")
    
    from src.graph.entity_index import build_faiss_ip_index, save_faiss_index, load_faiss_index
    
    n = 20
    d = 512
    k = 5
    
    x = np.random.randn(n, d).astype(np.float32)
    x_norm = normalize_rows(x)
    
    # Build index
    index = build_faiss_ip_index(x_norm)
    
    # Save
    index_path = tmp_path / "test.index"
    save_faiss_index(index, index_path)
    
    assert index_path.exists()
    
    # Load
    index_loaded = load_faiss_index(index_path)
    
    # Verify by searching
    query = x_norm[:5]
    D1, I1 = index.search(query, k)
    D2, I2 = index_loaded.search(query, k)
    
    np.testing.assert_array_equal(I1, I2)
    np.testing.assert_allclose(D1, D2, atol=1e-6)


def test_load_faiss_index_missing_file(tmp_path):
    """Test that load_faiss_index raises FileNotFoundError for missing index."""
    pytest.importorskip("faiss")
    
    from src.graph.entity_index import load_faiss_index
    
    index_path = tmp_path / "missing.index"
    
    with pytest.raises(FileNotFoundError):
        load_faiss_index(index_path)


# ============================================================================
# Test artifact validation
# ============================================================================

def test_load_neighbor_artifacts_validation_invalid_neighbors_shape(tmp_path):
    """Test that load_neighbor_artifacts validates neighbors shape."""
    # Save invalid 1D neighbors
    neighbors = np.array([1, 2, 3], dtype=np.int32)
    mutual = np.array([[1, 0]], dtype=np.bool_)
    
    neighbors_path = tmp_path / "neighbors.npy"
    mutual_path = tmp_path / "mutual.npy"
    
    np.save(neighbors_path, neighbors)
    np.save(mutual_path, mutual)
    
    with pytest.raises(ValueError, match="must be 2D"):
        load_neighbor_artifacts(neighbors_path, mutual_path)


def test_load_neighbor_artifacts_validation_out_of_range(tmp_path):
    """Test that load_neighbor_artifacts validates neighbor IDs are in range."""
    # Create neighbors with out-of-range ID
    neighbors = np.array([
        [1, 2],
        [0, 2],
        [1, 5],  # ID 5 is out of range (only 3 entities)
    ], dtype=np.int32)
    
    mutual = np.array([
        [1, 0],
        [1, 0],
        [1, 0],
    ], dtype=np.bool_)
    
    neighbors_path = tmp_path / "neighbors.npy"
    mutual_path = tmp_path / "mutual.npy"
    
    np.save(neighbors_path, neighbors)
    np.save(mutual_path, mutual)
    
    with pytest.raises(ValueError, match="out-of-range"):
        load_neighbor_artifacts(neighbors_path, mutual_path)


def test_load_neighbor_artifacts_validation_negative_ids(tmp_path):
    """Test that load_neighbor_artifacts validates neighbor IDs are non-negative."""
    # Create neighbors with negative ID
    neighbors = np.array([
        [1, 2],
        [0, -1],  # Negative ID
        [1, 0],
    ], dtype=np.int32)
    
    mutual = np.array([
        [1, 0],
        [1, 0],
        [1, 0],
    ], dtype=np.bool_)
    
    neighbors_path = tmp_path / "neighbors.npy"
    mutual_path = tmp_path / "mutual.npy"
    
    np.save(neighbors_path, neighbors)
    np.save(mutual_path, mutual)
    
    with pytest.raises(ValueError, match="negative"):
        load_neighbor_artifacts(neighbors_path, mutual_path)


def test_load_neighbor_artifacts_uint8_to_bool_conversion(tmp_path):
    """Test that load_neighbor_artifacts converts uint8 mutual mask to bool."""
    neighbors = np.array([
        [1, 2],
        [0, 2],
        [1, 0],
    ], dtype=np.int32)
    
    # Save as uint8 (legacy format)
    mutual_uint8 = np.array([
        [1, 0],
        [1, 1],
        [0, 1],
    ], dtype=np.uint8)
    
    neighbors_path = tmp_path / "neighbors.npy"
    mutual_path = tmp_path / "mutual.npy"
    
    np.save(neighbors_path, neighbors)
    np.save(mutual_path, mutual_uint8)
    
    # Load and verify conversion to bool
    neighbors_loaded, mutual_loaded = load_neighbor_artifacts(neighbors_path, mutual_path)
    
    assert mutual_loaded.dtype == np.bool_
    np.testing.assert_array_equal(mutual_loaded, mutual_uint8.astype(np.bool_))


def test_compute_mutual_knn_mask_validation(tmp_path):
    """Test that compute_mutual_knn_mask validates neighbor IDs."""
    # Out of range neighbors
    neighbors_bad = np.array([
        [1, 2],
        [0, 5],  # 5 is out of range
    ], dtype=np.int32)
    
    with pytest.raises(ValueError, match="out-of-range"):
        compute_mutual_knn_mask(neighbors_bad)
    
    # Negative neighbors
    neighbors_neg = np.array([
        [1, 2],
        [0, -1],
    ], dtype=np.int32)
    
    with pytest.raises(ValueError, match="negative"):
        compute_mutual_knn_mask(neighbors_neg)


def test_load_neighbor_artifacts_self_neighbor_validation(tmp_path):
    """Test that load_neighbor_artifacts detects self-neighbors."""
    # Create neighbors with self-neighbor (entity 1 has itself as neighbor)
    neighbors = np.array([
        [1, 2],
        [1, 0],  # self-neighbor: entity 1 includes itself
        [0, 1],
    ], dtype=np.int32)
    
    mutual = np.ones_like(neighbors, dtype=np.bool_)
    
    neighbors_path = tmp_path / "neighbors.npy"
    mutual_path = tmp_path / "mutual.npy"
    
    np.save(neighbors_path, neighbors)
    np.save(mutual_path, mutual)
    
    # Should raise error about self-neighbors
    with pytest.raises(ValueError, match="self-neighbors"):
        load_neighbor_artifacts(neighbors_path, mutual_path)


def test_load_entity_embeddings_expected_dim(tmp_path):
    """Test that load_entity_embeddings validates expected dimension."""
    from src.graph.entity_index import load_entity_embeddings
    
    # Create embeddings with wrong dimension
    embeddings_wrong_dim = np.random.randn(10, 128).astype(np.float32)
    wrong_dim_path = tmp_path / "embeddings_wrong_dim.npy"
    np.save(wrong_dim_path, embeddings_wrong_dim)
    
    # Should raise error with expected_dim=512
    with pytest.raises(ValueError, match="Expected embedding dimension 512, got 128"):
        load_entity_embeddings(wrong_dim_path, expected_dim=512)
    
    # Should succeed without expected_dim check
    result = load_entity_embeddings(wrong_dim_path)
    assert result.shape == (10, 128)
    
    # Should succeed with correct expected_dim
    result = load_entity_embeddings(wrong_dim_path, expected_dim=128)
    assert result.shape == (10, 128)


@pytest.mark.skipif(not pytest.importorskip("torch", reason="torch not installed"), reason="torch required")
def test_load_entity_embeddings_torch_compatibility(tmp_path):
    """Test that load_entity_embeddings handles torch.load with/without weights_only."""
    import torch
    from src.graph.entity_index import load_entity_embeddings
    
    # Test direct tensor
    embeddings_tensor = torch.randn(10, 512)
    tensor_path = tmp_path / "embeddings_tensor.pt"
    torch.save(embeddings_tensor, tensor_path)
    
    result = load_entity_embeddings(tensor_path, expected_dim=512)
    assert result.shape == (10, 512)
    assert result.dtype == np.float32
    
    # Test dict checkpoint with entity_embeddings key
    checkpoint = {"entity_embeddings": torch.randn(10, 512), "other_key": "value"}
    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save(checkpoint, checkpoint_path)
    
    result = load_entity_embeddings(checkpoint_path, expected_dim=512)
    assert result.shape == (10, 512)
    
    # Test dict checkpoint with embeddings key
    checkpoint2 = {"embeddings": torch.randn(10, 512)}
    checkpoint2_path = tmp_path / "checkpoint2.pt"
    torch.save(checkpoint2, checkpoint2_path)
    
    result = load_entity_embeddings(checkpoint2_path, expected_dim=512)
    assert result.shape == (10, 512)


@pytest.mark.skipif(not pytest.importorskip("faiss", reason="faiss not installed"), reason="faiss required")
def test_validate_faiss_index():
    """Test FAISS index validation helper."""
    from src.graph.entity_index import build_faiss_ip_index, validate_faiss_index, normalize_rows
    
    # Create valid index
    embeddings = np.random.randn(100, 512).astype(np.float32)
    embeddings_norm = normalize_rows(embeddings)
    index = build_faiss_ip_index(embeddings_norm)
    
    # Should pass with correct parameters
    validate_faiss_index(index, expected_n=100, expected_dim=512)
    
    # Should fail with wrong N
    with pytest.raises(ValueError, match="has 100 vectors, expected 50"):
        validate_faiss_index(index, expected_n=50, expected_dim=512)
    
    # Should fail with wrong dimension
    with pytest.raises(ValueError, match="has dimension 512, expected 256"):
        validate_faiss_index(index, expected_n=100, expected_dim=256)
