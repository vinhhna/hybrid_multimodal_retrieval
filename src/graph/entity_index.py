"""
Entity FAISS index for Phase 5 vision-grounded neighbor search.

Builds and loads FAISS indexes over entity embeddings for fast k-NN search
during candidate generation (safe neighbors expansion).

Phase 5 Day 1: Placeholder API (NotImplementedError).
Phase 5 Day 4: Full implementation with FAISS index building and loading.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Tuple, Optional
import numpy as np


def build_entity_faiss_index(
    entity_embeddings: np.ndarray,
    index_path: Path,
    index_type: str = "Flat",
    normalize: bool = True,
) -> None:
    """
    Build a FAISS index over entity embeddings for k-NN search.
    
    Phase 5 Day 1: Placeholder (raises NotImplementedError).
    Phase 5 Day 4: Full implementation with FAISS index building.
    
    Implementation plan (Day 4):
      1. Optionally normalize embeddings (for cosine similarity)
      2. Create FAISS index (IndexFlatIP for cosine, IndexFlatL2 for L2)
      3. Add embeddings to index
      4. Write index to disk with faiss.write_index()
    
    Args:
        entity_embeddings: Entity embedding matrix (shape: [num_entities, embedding_dim])
        index_path: Output path for FAISS index file (.index or .faiss)
        index_type: FAISS index type ("Flat", "IVF", "HNSW", etc.)
        normalize: If True, L2-normalize embeddings for cosine similarity
    
    Raises:
        NotImplementedError: Always on Day 1 (deferred to Day 4).
    
    Example:
        >>> # Day 4+ usage
        >>> import numpy as np
        >>> embeddings = np.random.rand(10000, 512).astype("float32")
        >>> build_entity_faiss_index(embeddings, Path("data/indices/entity.index"))
    """
    raise NotImplementedError(
        "build_entity_faiss_index() implementation deferred to Phase 5 Day 4. "
        "Day 1 provides only the function signature for scaffolding. "
        "Requires faiss-cpu or faiss-gpu installation (not imported at module level)."
    )


def load_entity_faiss_index(index_path: Path) -> Any:
    """
    Load a FAISS index from disk.
    
    Phase 5 Day 1: Placeholder (raises NotImplementedError).
    Phase 5 Day 4: Full implementation with FAISS index loading.
    
    Implementation plan (Day 4):
      1. Read index from disk with faiss.read_index()
      2. Return FAISS index object for search
    
    Args:
        index_path: Path to FAISS index file (.index or .faiss)
    
    Returns:
        FAISS index object (e.g., faiss.IndexFlatIP or faiss.IndexIVFFlat).
    
    Raises:
        NotImplementedError: Always on Day 1 (deferred to Day 4).
        FileNotFoundError: If index file does not exist (Day 4+).
    
    Example:
        >>> # Day 4+ usage
        >>> index = load_entity_faiss_index(Path("data/indices/entity.index"))
        >>> D, I = index.search(query_embeddings, k=10)  # k-NN search
    """
    raise NotImplementedError(
        "load_entity_faiss_index() implementation deferred to Phase 5 Day 4. "
        "Day 1 provides only the function signature for scaffolding. "
        "Requires faiss-cpu or faiss-gpu installation (not imported at module level)."
    )


def load_neighbors_arrays(
    neighbors_path: Path,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load precomputed k-NN neighbors from disk.
    
    Phase 5 Day 1: Placeholder (raises NotImplementedError).
    Phase 5 Day 4: Full implementation with numpy loading.
    
    Precomputed neighbors are stored as two arrays:
      - neighbors_indices: [num_entities, k] array of neighbor entity IDs
      - neighbors_distances: [num_entities, k] array of distances (or similarities)
    
    This avoids repeated FAISS searches during candidate generation if neighbors
    are precomputed once and cached.
    
    Args:
        neighbors_path: Path to neighbors file (e.g., .npz or .npy)
    
    Returns:
        Tuple of (neighbors_indices, neighbors_distances):
          - neighbors_indices: [num_entities, k] int array
          - neighbors_distances: [num_entities, k] float array
    
    Raises:
        NotImplementedError: Always on Day 1 (deferred to Day 4).
        FileNotFoundError: If neighbors file does not exist (Day 4+).
    
    Example:
        >>> # Day 4+ usage
        >>> indices, dists = load_neighbors_arrays(Path("data/entities/neighbors.npz"))
        >>> print(indices.shape)  # e.g., (10000, 16) for 10k entities, k=16
        >>> # Get neighbors for entity 42:
        >>> neighbor_ids = indices[42]
        >>> neighbor_sims = dists[42]
    """
    raise NotImplementedError(
        "load_neighbors_arrays() implementation deferred to Phase 5 Day 4. "
        "Day 1 provides only the function signature for scaffolding. "
        "Will load from .npz or .npy file using numpy on Day 4."
    )
