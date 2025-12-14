"""
Entity FAISS index for Phase 5 vision-grounded neighbor search.

Builds and loads FAISS indexes over entity embeddings for fast k-NN search
during candidate generation (safe neighbors expansion).

Phase 5 Day 4: Full implementation with FAISS index building, neighbor precomputation,
and mutual k-NN mask computation for vectorized candidate generation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Tuple
import numpy as np


# ============================================================================
# Normalization and contiguity helpers
# ============================================================================

def normalize_rows(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    L2-normalize rows of a matrix.
    
    Args:
        x: Input array of shape [N, D].
        eps: Small constant for numerical stability.
    
    Returns:
        Row-normalized array of same shape as x (new array, not in-place).
    """
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return x / (norm + eps)


def as_float32_contiguous(x: np.ndarray) -> np.ndarray:
    """
    Ensure array is float32 and contiguous (required by FAISS).
    
    Args:
        x: Input array.
    
    Returns:
        Contiguous float32 array.
    """
    return np.ascontiguousarray(x.astype(np.float32, copy=False))


# ============================================================================
# FAISS index building and loading (lazy imports)
# ============================================================================

def build_faiss_ip_index(x_norm: np.ndarray, validate_norms: bool = False) -> "Any":
    """
    Build a FAISS IndexFlatIP (inner product) index over normalized embeddings.
    
    This function lazy-imports faiss to avoid import-time failures on systems
    without faiss installed.
    
    Args:
        x_norm: L2-normalized embeddings of shape [N, D].
        validate_norms: If True, validate that vectors have unit L2 norm.
    
    Returns:
        FAISS IndexFlatIP with embeddings added.
    
    Raises:
        ImportError: If faiss is not installed.
        ValueError: If x_norm is not 2D or norms are invalid.
    """
    try:
        import faiss
    except ImportError as e:
        raise ImportError(
            "faiss is not installed. Install with: pip install faiss-cpu"
        ) from e
    
    if x_norm.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {x_norm.shape}")
    
    # Ensure float32 and contiguous
    x_norm = as_float32_contiguous(x_norm)
    
    # Optional norm validation
    if validate_norms:
        norms = np.linalg.norm(x_norm, axis=1)
        if not np.allclose(norms, 1.0, atol=1e-3):
            raise ValueError(
                f"Vectors must be L2-normalized for IndexFlatIP. "
                f"Found norm range: [{norms.min():.4f}, {norms.max():.4f}]"
            )
    
    d = x_norm.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(x_norm)
    
    return index


def save_faiss_index(index: "Any", path: Path) -> None:
    """
    Save a FAISS index to disk.
    
    Args:
        index: FAISS index object.
        path: Output path for index file.
    
    Raises:
        ImportError: If faiss is not installed.
    """
    try:
        import faiss
    except ImportError as e:
        raise ImportError(
            "faiss is not installed. Install with: pip install faiss-cpu"
        ) from e
    
    path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(path))


def load_faiss_index(path: Path) -> "Any":
    """
    Load a FAISS index from disk.
    
    Args:
        path: Path to FAISS index file.
    
    Returns:
        FAISS index object.
    
    Raises:
        ImportError: If faiss is not installed.
        FileNotFoundError: If index file does not exist.
    """
    try:
        import faiss
    except ImportError as e:
        raise ImportError(
            "faiss is not installed. Install with: pip install faiss-cpu"
        ) from e
    
    if not path.exists():
        raise FileNotFoundError(f"FAISS index not found: {path}")
    
    return faiss.read_index(str(path))


def validate_faiss_index(index: "Any", expected_n: int, expected_dim: int) -> None:
    """
    Validate FAISS index metadata matches expected values.
    
    Args:
        index: FAISS index object.
        expected_n: Expected number of vectors (index.ntotal).
        expected_dim: Expected dimension (index.d).
    
    Raises:
        ValueError: If index metadata does not match expected values.
    """
    ntotal = getattr(index, "ntotal", None)
    d = getattr(index, "d", None)
    
    if ntotal is None or d is None:
        raise ValueError("FAISS index missing ntotal/d metadata")
    
    if int(ntotal) != int(expected_n):
        raise ValueError(f"FAISS index ntotal {ntotal} != expected {expected_n}")
    
    if int(d) != int(expected_dim):
        raise ValueError(f"FAISS index dim {d} != expected {expected_dim}")


# ============================================================================
# Neighbor computation
# ============================================================================

def compute_semantic_neighbors(
    index: "Any",
    x_norm: np.ndarray,
    k: int,
    batch_size: int = 4096,
) -> np.ndarray:
    """
    Compute top-k semantic neighbors for all entities, excluding self.
    
    Robustly handles duplicate vectors by dynamically increasing search depth
    if needed to guarantee k non-self neighbors per entity.
    
    Args:
        index: FAISS index built on normalized embeddings.
        x_norm: L2-normalized embeddings of shape [N, D].
        k: Number of neighbors to retrieve (excluding self).
        batch_size: Batch size for FAISS search.
    
    Returns:
        Neighbor IDs array of shape [N, k], dtype int32.
        NEVER contains -1 padding or self neighbors.
    
    Raises:
        ValueError: If k >= N.
        RuntimeError: If unable to find k non-self neighbors for any entity.
    """
    n = x_norm.shape[0]
    
    if k >= n:
        raise ValueError(f"k={k} must be < N={n}")
    
    # Ensure float32 and contiguous
    x_norm = as_float32_contiguous(x_norm)
    
    neighbors = np.zeros((n, k), dtype=np.int32)
    
    for start_idx in range(0, n, batch_size):
        end_idx = min(start_idx + batch_size, n)
        batch = as_float32_contiguous(x_norm[start_idx:end_idx])
        batch_indices = np.arange(start_idx, end_idx)
        
        # Search for k+1 neighbors (expecting self to be first)
        k_search = k + 1
        D, I = index.search(batch, k_search)
        
        # Check if self is first (fast path)
        if np.all(I[:, 0] == batch_indices):
            # Self is first, take columns 1:k+1
            neighbors[start_idx:end_idx] = I[:, 1:k+1].astype(np.int32)
        else:
            # Fallback: filter out self per row with per-row retry logic
            for i, row_idx in enumerate(batch_indices):
                row_neighbors = I[i]
                # Remove self
                mask = row_neighbors != row_idx
                filtered = row_neighbors[mask]
                
                # If we don't have enough non-self neighbors, retry with more depth
                # Use per-row k_search to avoid mutation affecting other rows
                k_search_row = k_search
                retry_count = 0
                max_retries = 5
                
                while len(filtered) < k and k_search_row < n and retry_count < max_retries:
                    k_search_row = min(n, k_search_row + max(8, k))
                    D_retry, I_retry = index.search(batch[i:i+1], k_search_row)
                    row_neighbors = I_retry[0]
                    mask = row_neighbors != row_idx
                    filtered = row_neighbors[mask]
                    retry_count += 1
                
                # MUST have k non-self neighbors (no -1 padding allowed)
                if len(filtered) < k:
                    raise RuntimeError(
                        f"Unable to find {k} non-self neighbors for entity {row_idx}. "
                        f"Found only {len(filtered)} neighbors after {retry_count} retries. "
                        f"This may indicate duplicate/identical vectors in embeddings. "
                        f"Solutions: 1) Reduce K_neighbors in config, 2) Remove duplicate entities, "
                        f"3) Add noise to break ties."
                    )
                
                neighbors[row_idx] = filtered[:k].astype(np.int32)
    
    return neighbors


def compute_mutual_knn_mask(neighbors: np.ndarray, block_size: int = 1024) -> np.ndarray:
    """
    Compute mutual k-NN mask for precomputed neighbors using block-vectorization.
    
    mutual[i, jpos] = True if i is in neighbors[neighbors[i, jpos]].
    
    This implementation uses block processing with numpy broadcasting to avoid
    slow nested Python loops. For each block of entities, it checks membership
    via vectorized operations.
    
    Args:
        neighbors: Neighbor IDs array of shape [N, K], dtype int32.
        block_size: Number of entities to process per block (memory vs speed).
    
    Returns:
        Mutual mask array of shape [N, K], dtype bool.
    
    Raises:
        ValueError: If neighbors contains invalid IDs.
    """
    neighbors = np.asarray(neighbors)
    n, k = neighbors.shape
    
    # Validate neighbor IDs are in valid range
    if neighbors.min() < 0:
        raise ValueError(f"Neighbors contains negative IDs: min={neighbors.min()}")
    if neighbors.max() >= n:
        raise ValueError(
            f"Neighbors contains out-of-range IDs: max={neighbors.max()} >= N={n}"
        )
    
    mutual = np.zeros((n, k), dtype=np.bool_)
    idx = np.arange(n, dtype=neighbors.dtype)
    
    for start in range(0, n, block_size):
        end = min(n, start + block_size)
        # Current entity IDs in this block
        I = idx[start:end]  # shape: [B]
        # Their neighbors
        nbr = neighbors[start:end]  # shape: [B, K]
        # Neighbors of neighbors (gather)
        nbr2 = neighbors[nbr]  # shape: [B, K, K]
        # Check if entity i is in neighbors[j] for each j in neighbors[i]
        # Broadcasting: I[:, None, None] shape [B, 1, 1], nbr2 shape [B, K, K]
        mutual[start:end] = (nbr2 == I[:, None, None]).any(axis=2)
    
    return mutual


# ============================================================================
# Artifact loading helpers
# ============================================================================

@dataclass(frozen=True)
class EntityIndexArtifacts:
    """
    Paths to precomputed entity index artifacts.
    
    Attributes:
        neighbors_path: Path to neighbors array (.npy).
        mutual_path: Path to mutual k-NN mask array (.npy).
        faiss_index_path: Path to FAISS index file (.index).
    """
    neighbors_path: Path
    mutual_path: Path
    faiss_index_path: Path


def load_neighbor_artifacts(
    neighbors_path: Path,
    mutual_path: Path,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load precomputed neighbor arrays from disk with validation.
    
    Args:
        neighbors_path: Path to neighbors .npy file.
        mutual_path: Path to mutual mask .npy file.
    
    Returns:
        Tuple of (neighbors, mutual):
          - neighbors: [N, K] int32 array of neighbor IDs
          - mutual: [N, K] bool array of mutual k-NN flags
    
    Raises:
        FileNotFoundError: If either file does not exist.
        ValueError: If arrays have invalid shape, dtype, or values.
    """
    if not neighbors_path.exists():
        raise FileNotFoundError(f"Neighbors file not found: {neighbors_path}")
    if not mutual_path.exists():
        raise FileNotFoundError(f"Mutual mask file not found: {mutual_path}")
    
    neighbors = np.load(neighbors_path, allow_pickle=False)
    mutual = np.load(mutual_path, allow_pickle=False)
    
    # Validate neighbors
    if neighbors.ndim != 2:
        raise ValueError(
            f"Neighbors must be 2D array, got shape {neighbors.shape}"
        )
    if not np.issubdtype(neighbors.dtype, np.integer):
        raise ValueError(
            f"Neighbors must have integer dtype, got {neighbors.dtype}"
        )
    if neighbors.min() < 0:
        raise ValueError(
            f"Neighbors contains negative IDs: min={neighbors.min()}"
        )
    
    n, k = neighbors.shape
    if neighbors.max() >= n:
        raise ValueError(
            f"Neighbors contains out-of-range IDs: max={neighbors.max()} >= N={n}"
        )
    
    # Validate mutual
    if mutual.ndim != 2:
        raise ValueError(
            f"Mutual mask must be 2D array, got shape {mutual.shape}"
        )
    if mutual.shape != neighbors.shape:
        raise ValueError(
            f"Mutual mask shape {mutual.shape} must match neighbors shape {neighbors.shape}"
        )
    
    # Cast mutual to bool (handle uint8 legacy format)
    if mutual.dtype == np.uint8:
        if not np.all((mutual == 0) | (mutual == 1)):
            raise ValueError(
                "Mutual mask with uint8 dtype must contain only 0 or 1 values"
            )
        mutual = mutual.astype(np.bool_)
    elif mutual.dtype != np.bool_:
        raise ValueError(
            f"Mutual mask must be bool or uint8 dtype, got {mutual.dtype}"
        )
    
    # Validate no self-neighbors (critical for Day 5 safe_neighbors correctness)
    ids = np.arange(n, dtype=np.int32)[:, None]
    if (neighbors == ids).any():
        raise ValueError(
            "Neighbors array contains self-neighbors. This will corrupt Day 5 safe_neighbors. "
            "Recompute artifacts with: python scripts/precompute_entity_neighbors.py --config CONFIG --overwrite"
        )
    
    # Ensure neighbors is int32
    neighbors = neighbors.astype(np.int32, copy=False)
    
    return neighbors, mutual


def load_entity_embeddings(path: Path, expected_dim: int | None = None) -> np.ndarray:
    """
    Load entity embeddings from disk (supports .npy and .pt).
    
    For .pt files, supports:
      - Direct torch.Tensor
      - Dict checkpoints with keys: "entity_embeddings", "embeddings", "x", "tensor"
      - Dict with single tensor value (fallback)
    
    This function lazy-imports torch to avoid import-time dependency.
    Compatible with older torch versions that don't have weights_only kwarg.
    
    Args:
        path: Path to embeddings file (.npy or .pt).
        expected_dim: If provided, validate embeddings have this dimension.
    
    Returns:
        Embeddings array of shape [N, D], dtype float32.
    
    Raises:
        FileNotFoundError: If embeddings file does not exist.
        ValueError: If file format is unsupported, checkpoint format unrecognized,
                    or dimension mismatch with expected_dim.
    """
    if not path.exists():
        raise FileNotFoundError(f"Entity embeddings not found: {path}")
    
    if path.suffix == ".npy":
        arr = np.load(path, allow_pickle=False)
        if arr.ndim != 2:
            raise ValueError(
                f"Expected 2D embeddings array, got shape {arr.shape}"
            )
        if expected_dim is not None and arr.shape[1] != expected_dim:
            raise ValueError(
                f"Expected embedding dimension {expected_dim}, got {arr.shape[1]}"
            )
        return arr.astype(np.float32)
    
    elif path.suffix == ".pt":
        try:
            import torch
        except ImportError as e:
            raise ImportError(
                "torch is required to load .pt files. Install with: pip install torch"
            ) from e
        
        # Handle older torch versions without weights_only kwarg
        try:
            obj = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            # Older torch doesn't have weights_only parameter
            obj = torch.load(path, map_location="cpu")
        
        # Handle different checkpoint formats
        tensor = None
        
        if isinstance(obj, torch.Tensor):
            tensor = obj
        elif isinstance(obj, dict):
            # Try common keys in order (prefer entity_embeddings first for this repo)
            for key in ("entity_embeddings", "embeddings", "x", "tensor"):
                if key in obj:
                    candidate = obj[key]
                    if isinstance(candidate, (torch.Tensor, torch.nn.Parameter)):
                        tensor = candidate
                        break
            
            # Fallback: if dict has single tensor value, use it
            if tensor is None:
                tensor_values = [
                    v for v in obj.values()
                    if isinstance(v, (torch.Tensor, torch.nn.Parameter))
                ]
                if len(tensor_values) == 1:
                    tensor = tensor_values[0]
                else:
                    raise ValueError(
                        f"Unrecognized .pt checkpoint format. "
                        f"Expected direct Tensor or dict with keys: "
                        f"'entity_embeddings', 'embeddings', 'x', or 'tensor'. "
                        f"Found keys: {list(obj.keys())}"
                    )
        else:
            raise ValueError(
                f"Unsupported .pt object type: {type(obj)}. "
                f"Expected torch.Tensor or dict."
            )
        
        # Convert tensor to numpy
        arr = tensor.detach().cpu().numpy()
        
        if arr.ndim != 2:
            raise ValueError(
                f"Expected 2D embeddings tensor, got shape {arr.shape}"
            )
        
        if expected_dim is not None and arr.shape[1] != expected_dim:
            raise ValueError(
                f"Expected embedding dimension {expected_dim}, got {arr.shape[1]}"
            )
        
        return arr.astype(np.float32)
    
    else:
        raise ValueError(
            f"Unsupported embeddings format: {path.suffix}. Use .npy or .pt"
        )
