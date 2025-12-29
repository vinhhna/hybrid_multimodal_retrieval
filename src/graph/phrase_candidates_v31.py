"""
Phrase candidate generation for Phase 5 vision-grounded entity graph (v3.1).

Generates candidate entity phrases for each image using multiple sources:
  - Caption entities (from Flickr30K captions)
  - Visual prior entities (via FAISS topK image→entity retrieval)
  - Safe neighbors (semantically similar entities with visual evidence)
  - Global fallback (high-frequency entities as last resort)

Phase 5 Day 5: Full vectorized implementation with priority ordering and C_max cap.

Algorithm:
  1. E_cap(I): caption-linked entity IDs from context
  2. E_vis(I): FAISS topK query (image→entity retrieval; "Stage-0 visual prior")
  3. Nbr_safe(I): semantic neighbors expanded from seeds, with masks:
     - Gather neighbors[E_seed] → 2D [S, K] int32 array
     - Apply masks (mutual-kNN, similarity, pixel anchoring) vectorized
  4. E_global: precomputed list (top-N frequent caption entities)
  5. Merge with stable priority: caption > visual prior > safe neighbors > global
  6. Budget cap C_max

Performance:
  - No Python loops over candidate entities (only over images/batches)
  - Vectorized NumPy array ops for gather/masks/flatten/dedup
  - Target: <10ms/image excluding disk IO
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Tuple
import numpy as np
import json


@dataclass
class CandidateGenConfigV31:
    """Configuration for Phase 5 Day 5 candidate generation."""
    
    # Budget and source caps
    C_max: int
    visual_prior_topK: int
    visual_prior_tau_vis: Optional[float]  # Similarity floor for visual prior (filter FAISS by score)
    global_topN: int
    priority_order: List[str]
    
    # Per-source caps (applied before merge; None = no cap)
    max_caption: Optional[int] = None
    max_visual_prior: Optional[int] = None
    max_safe_neighbors: Optional[int] = None
    max_global: Optional[int] = None
    
    # Safe neighbors config
    safe_neighbors_enabled: bool = True
    seed_sources: List[str] = None
    use_mutual_knn: bool = True
    require_in_vis_prior: bool = False
    tau_sim: Optional[float] = None
    
    # Artifact paths
    entity_faiss_index_path: Path = None
    entity_neighbors_path: Path = None
    entity_neighbors_mutual_path: Path = None
    entity_global_list_path: Optional[Path] = None


# ============================================================================
# Helper functions
# ============================================================================

def _cap_source(arr: np.ndarray, k: Optional[int]) -> np.ndarray:
    """
    Truncate array to first k elements if k is specified.
    
    Args:
        arr: 1D array of entity IDs.
        k: Maximum number of elements to keep (None = no cap).
    
    Returns:
        Truncated array (or original if k is None or k <= 0).
    """
    if k is None or k <= 0:
        return arr
    return arr[:k]


# ============================================================================
# Stable unique preserving order (NumPy-only, no Python loops)
# ============================================================================

def stable_unique_preserve_order(ids: np.ndarray) -> np.ndarray:
    """
    Remove duplicates from array while preserving first occurrence order.
    
    Uses NumPy-only operations (no per-entity Python loops).
    
    Args:
        ids: 1D array of entity IDs (any integer dtype).
    
    Returns:
        1D int32 array with duplicates removed, preserving first occurrence order.
    
    Example:
        >>> ids = np.array([3, 1, 4, 1, 5, 9, 3])
        >>> stable_unique_preserve_order(ids)
        array([3, 1, 4, 5, 9], dtype=int32)
    """
    if ids.size == 0:
        return ids.astype(np.int32, copy=False)
    
    ids = ids.astype(np.int32, copy=False)
    _, first_idx = np.unique(ids, return_index=True)
    return ids[np.sort(first_idx)]


# ============================================================================
# Pixel anchoring membership via sorted merge (no Python loops)
# ============================================================================

def pixel_anchoring_mask(
    neighbors: np.ndarray,
    vis_prior_ids: np.ndarray,
) -> np.ndarray:
    """
    Compute boolean mask for neighbors that are also in visual prior (pixel anchoring).
    
    Uses sorted merge via searchsorted (no Python membership loops).
    Index-safe: handles empty vis_prior and uses clip to avoid out-of-bounds access.
    
    Args:
        neighbors: [S, K] int32 array of neighbor IDs.
        vis_prior_ids: 1D int32 array of visual prior entity IDs.
    
    Returns:
        [S, K] bool array: True where neighbors[i, j] is in vis_prior_ids.
    
    Example:
        >>> neighbors = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
        >>> vis_prior = np.array([2, 5, 10], dtype=np.int32)
        >>> mask = pixel_anchoring_mask(neighbors, vis_prior)
        >>> mask
        array([[False,  True, False],
               [False,  True, False]])
    """
    # Sort and ensure int32
    vis = np.unique(vis_prior_ids.astype(np.int32, copy=False))
    
    # Early exit if no visual prior entities
    if vis.size == 0:
        return np.zeros(neighbors.shape, dtype=bool)
    
    vis.sort()
    
    flat = neighbors.reshape(-1).astype(np.int32, copy=False)
    
    # searchsorted + safe bounds check with clip
    pos = np.searchsorted(vis, flat)
    in_bounds = pos < vis.size
    # Safe lookup: clip pos to valid range to avoid IndexError
    vis_at = np.take(vis, np.clip(pos, 0, vis.size - 1))
    in_vis = in_bounds & (vis_at == flat)
    
    return in_vis.reshape(neighbors.shape)


# ============================================================================
# Global entity computation (deterministic, tie-break by ID ascending)
# ============================================================================

def compute_global_top_entities_from_context(
    entity_context: dict,
    topN: int,
) -> np.ndarray:
    """
    Compute top-N frequent caption entities from entity_context.
    
    Deterministic: tie-break by entity_id ascending.
    
    Args:
        entity_context: Dict mapping entity_id (str) -> {"entity": str, "image_ids": List[str], ...}
        topN: Number of top entities to return.
    
    Returns:
        1D int32 array of entity IDs, length min(topN, n_entities).
    
    Example:
        >>> context = {"0": {"entity": "dog", "image_ids": ["1", "2", "3"]},
        ...            "1": {"entity": "cat", "image_ids": ["1"]}}
        >>> compute_global_top_entities_from_context(context, topN=2)
        array([0, 1], dtype=int32)  # dog first (more images), then cat
    """
    counts = []
    for eid_str, info in entity_context.items():
        eid = int(eid_str)
        count = len(info.get("image_ids", []))
        counts.append((eid, count))
    
    # Sort by count descending, then by entity_id ascending (deterministic tie-break)
    counts.sort(key=lambda x: (-x[1], x[0]))
    
    top_ids = [eid for eid, _ in counts[:topN]]
    return np.array(top_ids, dtype=np.int32)


# ============================================================================
# Config loader
# ============================================================================

def load_candidate_gen_config_from_yaml(config_path: Path) -> CandidateGenConfigV31:
    """
    Load Day 5 candidate generation config from entity_graph.yaml.
    
    Resolves relative paths against repo root (config_path.parents[1]).
    
    Args:
        config_path: Path to YAML config file (typically configs/entity_graph.yaml).
    
    Returns:
        CandidateGenConfigV31 with all required parameters.
    
    Raises:
        FileNotFoundError: If config file does not exist.
        KeyError: If required Phase 5 keys are missing.
    """
    import yaml
    
    # Convert to Path if string
    if isinstance(config_path, str):
        config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    
    phase5 = cfg.get("phase5", {})
    candidates = phase5.get("candidates", {})
    safe_neighbors = phase5.get("safe_neighbors", {})
    artifacts = phase5.get("artifacts", {})
    
    # Validate Phase 5 config exists
    if not phase5:
        raise KeyError(
            "Missing 'phase5' section in config. "
            "Day 5 candidate generation requires phase5.candidates, phase5.safe_neighbors, and phase5.artifacts."
        )
    
    # Resolve relative paths against repo root (configs/ parent)
    base_dir = config_path.resolve().parents[1]
    
    def _resolve_path(path_str: str) -> Path:
        """Resolve path relative to repo root if not absolute."""
        p = Path(path_str)
        return p if p.is_absolute() else (base_dir / p)
    
    # Extract priority order and validate
    priority_order = candidates.get("priority_order", ["caption", "visual_prior", "safe_neighbors", "global"])
    valid_sources = {"caption", "visual_prior", "safe_neighbors", "global"}
    invalid = set(priority_order) - valid_sources
    if invalid:
        raise ValueError(f"Invalid priority_order sources: {invalid}. Valid: {valid_sources}")
    
    # Read tau_vis (may be named tau_vis or visual_prior_tau_vis in YAML)
    tau_vis = candidates.get("visual_prior_tau_vis") or candidates.get("tau_vis")
    
    # Detect config drift: prevent duplicate paths in entity_graph.* and phase5.artifacts.*
    entity_graph = cfg.get("entity_graph", {})
    for key in ["entity_faiss_index_path", "entity_neighbors_path", "entity_neighbors_mutual_path"]:
        phase5_val = artifacts.get(key)
        eg_val = entity_graph.get(key)
        if phase5_val and eg_val and phase5_val != eg_val:
            raise ValueError(
                f"Config drift detected for '{key}': "
                f"phase5.artifacts.{key}='{phase5_val}' != entity_graph.{key}='{eg_val}'. "
                f"Use phase5.artifacts.* as the canonical source and remove duplicate from entity_graph."
            )
    
    # Extract values with defaults matching the plan
    return CandidateGenConfigV31(
        C_max=candidates.get("C_max", 128),
        visual_prior_topK=candidates.get("visual_prior_topK", 50),
        visual_prior_tau_vis=tau_vis,
        global_topN=candidates.get("global_topN", 50),
        priority_order=priority_order,
        max_caption=candidates.get("max_caption"),
        max_visual_prior=candidates.get("max_visual_prior"),
        max_safe_neighbors=candidates.get("max_safe_neighbors"),
        max_global=candidates.get("max_global"),
        safe_neighbors_enabled=safe_neighbors.get("enabled", True),
        seed_sources=safe_neighbors.get("seed_sources", ["caption"]),
        use_mutual_knn=safe_neighbors.get("use_mutual_knn", True),
        require_in_vis_prior=safe_neighbors.get("require_in_vis_prior", False),
        tau_sim=safe_neighbors.get("tau_sim", None),
        entity_faiss_index_path=_resolve_path(artifacts.get("entity_faiss_index_path") or eg_val or "data/entities/entity_faiss.index"),
        entity_neighbors_path=_resolve_path(artifacts.get("entity_neighbors_path", "data/entities/entity_neighbors.npy")),
        entity_neighbors_mutual_path=_resolve_path(artifacts.get("entity_neighbors_mutual_path", "data/entities/entity_neighbors_mutual.npy")),
        entity_global_list_path=_resolve_path(artifacts["entity_global_list_path"]) if artifacts.get("entity_global_list_path") else None,
    )


# ============================================================================
# Artifact loaders (reuse existing helpers)
# ============================================================================

def load_neighbors_artifacts(
    neighbors_path: Path,
    mutual_path: Path,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load precomputed neighbor arrays from disk.
    
    Reuses existing helper from src.graph.entity_index if available,
    otherwise implements minimal loader.
    
    Args:
        neighbors_path: Path to neighbors .npy file.
        mutual_path: Path to mutual mask .npy file.
    
    Returns:
        Tuple of (neighbors, mutual):
          - neighbors: [N, K] int32 array of neighbor IDs
          - mutual: [N, K] bool array of mutual k-NN flags
    
    Raises:
        FileNotFoundError: If either file does not exist.
    """
    try:
        from src.graph.entity_index import load_neighbor_artifacts as _load_helper
        return _load_helper(neighbors_path, mutual_path)
    except ImportError:
        # Fallback: minimal loader if entity_index not available
        if not neighbors_path.exists():
            raise FileNotFoundError(f"Neighbors file not found: {neighbors_path}")
        if not mutual_path.exists():
            raise FileNotFoundError(f"Mutual mask file not found: {mutual_path}")
        
        neighbors = np.load(neighbors_path, allow_pickle=False)
        neighbors = neighbors.astype(np.int32, copy=False)
        
        mutual = np.load(mutual_path, allow_pickle=False)
        
        # Cast mutual to bool (handle uint8 legacy format)
        if mutual.dtype == np.uint8:
            mutual = mutual.astype(np.bool_, copy=False)
        elif mutual.dtype != np.bool_:
            mutual = mutual.astype(np.bool_, copy=False)
        
        return neighbors, mutual


def load_faiss_index(index_path: Path) -> "Any":
    """
    Load a FAISS index from disk.
    
    Reuses existing helper from src.graph.entity_index if available.
    
    Args:
        index_path: Path to FAISS index file.
    
    Returns:
        FAISS index object.
    
    Raises:
        ImportError: If faiss is not installed.
        FileNotFoundError: If index file does not exist.
    """
    try:
        from src.graph.entity_index import load_faiss_index as _load_helper
        return _load_helper(index_path)
    except ImportError:
        # Fallback: minimal loader
        try:
            import faiss
        except ImportError as e:
            raise ImportError(
                "faiss is not installed. Install with: pip install faiss-cpu"
            ) from e
        
        if not index_path.exists():
            raise FileNotFoundError(f"FAISS index not found: {index_path}")
        
        return faiss.read_index(str(index_path))


def load_entity_context(context_path: Path) -> dict:
    """
    Load entity context from JSON file.
    
    Args:
        context_path: Path to entity_context.json.
    
    Returns:
        Dict mapping entity_id (str) -> {"entity": str, "image_ids": List[str], ...}
    
    Raises:
        FileNotFoundError: If context file does not exist.
    """
    if not context_path.exists():
        raise FileNotFoundError(f"Entity context not found: {context_path}")
    
    with open(context_path, "r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================================
# PhraseCandidateGeneratorV31 (main class)
# ============================================================================

class PhraseCandidateGeneratorV31:
    """
    Phase 5 Day 5 candidate generator with vectorized safe neighbors.
    
    Generates candidate entity phrases for each image using:
      - Caption entities (E_cap)
      - Visual prior entities (E_vis via FAISS)
      - Safe neighbors (Nbr_safe with vectorized masks)
      - Global fallback (E_global)
    
    Performance:
      - No Python loops over candidate entities
      - Vectorized NumPy array ops
      - Target: <10ms/image excluding disk IO
    """
    
    def __init__(
        self,
        cfg: CandidateGenConfigV31,
        *,
        neighbors: np.ndarray,
        mutual: np.ndarray,
        faiss_index: Optional["Any"],
        global_list: np.ndarray,
        n_entities: int,
    ):
        """
        Initialize candidate generator with preloaded artifacts.
        
        Args:
            cfg: Configuration for candidate generation.
            neighbors: [N, K] int32 array of neighbor IDs.
            mutual: [N, K] bool array of mutual k-NN flags.
            faiss_index: FAISS index for image→entity retrieval (None if unavailable).
            global_list: 1D int32 array of global entity IDs.
            n_entities: Total number of entities in vocabulary.
        
        Raises:
            ValueError: If tau_sim is configured (not yet implemented in v3.1).
        """
        self.cfg = cfg
        self.neighbors = neighbors
        self.mutual = mutual
        self.faiss_index = faiss_index
        self.global_list = global_list
        self.n_entities = n_entities
        
        # Validate artifacts
        if neighbors.ndim != 2 or neighbors.dtype != np.int32:
            raise ValueError(f"neighbors must be 2D int32 array, got shape {neighbors.shape}, dtype {neighbors.dtype}")
        if mutual.ndim != 2 or mutual.dtype != np.bool_:
            raise ValueError(f"mutual must be 2D bool array, got shape {mutual.shape}, dtype {mutual.dtype}")
        if neighbors.shape != mutual.shape:
            raise ValueError(f"neighbors and mutual must have same shape, got {neighbors.shape} vs {mutual.shape}")
        if global_list.dtype != np.int32:
            raise ValueError(f"global_list must be int32 array, got {global_list.dtype}")
        
        # Fail-fast: tau_sim not implemented in v3.1
        if cfg.tau_sim is not None:
            raise ValueError(
                f"tau_sim={cfg.tau_sim} requires precomputed neighbor similarity artifact "
                f"(entity_neighbors_sims.npy), which is not yet available. "
                f"Set 'phase5.safe_neighbors.tau_sim: null' in config until Day 6+ implements similarity artifacts."
            )
    
    def _valid_ids(self, ids: np.ndarray) -> np.ndarray:
        """
        Filter entity IDs to valid range [0, n_entities).
        
        Removes negatives (e.g., FAISS -1 for no result) and out-of-range IDs.
        
        Args:
            ids: 1D array of entity IDs (any integer dtype).
        
        Returns:
            1D int32 array of valid entity IDs.
        """
        if ids is None or ids.size == 0:
            return np.empty((0,), dtype=np.int32)
        ids = ids.astype(np.int32, copy=False).reshape(-1)
        return ids[(ids >= 0) & (ids < self.n_entities)]
    
    def generate_for_image(
        self,
        *,
        cap_ids: np.ndarray,
        image_vec: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Generate candidate entity IDs for a single image.
        
        Args:
            cap_ids: 1D int32 array of caption entity IDs for this image.
            image_vec: Optional 1D float32 image embedding for FAISS query.
                       If None, treats E_vis(I) = [] and continues.
        
        Returns:
            1D int32 array of candidate entity IDs, length ≤ C_max,
            in stable priority order (caption > visual prior > safe neighbors > global).
        
        Example:
            >>> gen = PhraseCandidateGeneratorV31(...)
            >>> cap_ids = np.array([10, 25, 42], dtype=np.int32)
            >>> image_vec = np.random.randn(512).astype(np.float32)
            >>> candidates = gen.generate_for_image(cap_ids=cap_ids, image_vec=image_vec)
            >>> print(len(candidates))  # e.g., 87 (≤ C_max)
        """
        # 1. E_cap(I): validate caption IDs
        e_cap = self._valid_ids(cap_ids)
        
        # 2. E_vis(I): visual prior via FAISS
        e_vis = self._get_visual_prior(image_vec)
        
        # 3. Nbr_safe(I): safe neighbors with vectorized masks
        nbr_safe = self._get_safe_neighbors(e_cap, e_vis)
        
        # 4. E_global: precomputed global list
        e_global = self.global_list
        
        # 5. Merge with stable priority and budget cap
        return self._merge_candidates(e_cap, e_vis, nbr_safe, e_global)
    
    def generate_for_batch(
        self,
        *,
        cap_ids_list: List[np.ndarray],
        image_vecs: Optional[np.ndarray] = None,
        batch_size: int = 1024,
    ) -> List[np.ndarray]:
        """
        Generate candidate entity IDs for a batch of images.
        
        Uses batched FAISS search for efficiency (controlled by batch_size).
        
        Args:
            cap_ids_list: List of 1D int32 arrays (one per image).
            image_vecs: Optional [N, D] float32 array of image embeddings.
                        If None, treats all E_vis(I) = [] and continues.
            batch_size: Batch size for FAISS queries (default 1024).
        
        Returns:
            List of 1D int32 arrays (parallel structure to cap_ids_list).
        
        Example:
            >>> gen = PhraseCandidateGeneratorV31(...)
            >>> cap_ids_list = [np.array([10, 25], dtype=np.int32), np.array([30], dtype=np.int32)]
            >>> image_vecs = np.random.randn(2, 512).astype(np.float32)
            >>> candidates = gen.generate_for_batch(cap_ids_list=cap_ids_list, image_vecs=image_vecs)
            >>> print(len(candidates))  # 2 (one per image)
        """
        n_images = len(cap_ids_list)
        
        # Precompute visual prior IDs for all images using batched FAISS search
        vis_ids_list = [np.empty((0,), dtype=np.int32) for _ in range(n_images)]
        
        if image_vecs is not None and self.faiss_index is not None:
            # Fail if FAISS required but faiss module unavailable
            try:
                import faiss  # noqa: F401
            except ImportError as e:
                raise ImportError(
                    "image_vecs provided but faiss is not installed. "
                    "Install with: pip install faiss-cpu"
                ) from e
            
            # Ensure contiguous float32 for FAISS
            vecs = np.ascontiguousarray(image_vecs.astype(np.float32, copy=False))
            
            # Batched FAISS search
            for start in range(0, n_images, batch_size):
                end = min(start + batch_size, n_images)
                chunk = vecs[start:end]
                
                try:
                    D, I = self.faiss_index.search(chunk, self.cfg.visual_prior_topK)
                    # D, I shape [chunk_size, K]; I may contain -1 for no result
                    
                    for j in range(I.shape[0]):
                        ids = I[j]
                        dists = D[j]
                        
                        # Apply tau_vis filter if configured
                        if self.cfg.visual_prior_tau_vis is not None:
                            mask = dists >= self.cfg.visual_prior_tau_vis
                            ids = ids[mask]
                        
                        vis_ids_list[start + j] = self._valid_ids(ids)
                except Exception as e:
                    # Graceful fallback: treat as no visual prior for this batch
                    import warnings
                    warnings.warn(f"FAISS batch search failed: {e}. Treating E_vis = [] for batch [{start}:{end}].")
        
        # Generate candidates for each image using precomputed visual prior
        results = []
        for i, cap_ids in enumerate(cap_ids_list):
            # Validate caption IDs
            e_cap = self._valid_ids(cap_ids)
            
            # Use precomputed visual prior
            e_vis = vis_ids_list[i]
            
            # Compute safe neighbors
            nbr_safe = self._get_safe_neighbors(e_cap, e_vis)
            
            # Merge with stable priority and budget cap
            e_global = self.global_list
            candidates = self._merge_candidates(e_cap, e_vis, nbr_safe, e_global)
            results.append(candidates)
        
        return results
    
    # ========================================================================
    # Internal helpers
    # ========================================================================
    
    def _get_visual_prior(self, image_vec: Optional[np.ndarray]) -> np.ndarray:
        """
        Query FAISS for visual prior entities.
        
        Applies tau_vis similarity floor if configured (filters by FAISS distance).
        Ensures FAISS input is float32 contiguous for dtype safety.
        
        Args:
            image_vec: 1D image embedding (any float dtype), or None.
        
        Returns:
            1D int32 array of entity IDs from visual prior (empty if image_vec is None or FAISS unavailable).
        """
        if image_vec is None or self.faiss_index is None:
            return np.array([], dtype=np.int32)
        
        # Ensure float32 contiguous for FAISS (handles float64 inputs safely)
        vec = np.ascontiguousarray(image_vec.astype(np.float32, copy=False))
        if vec.ndim == 1:
            vec = vec.reshape(1, -1)
        
        # FAISS query (keep distances for tau_vis filtering)
        try:
            D, I = self.faiss_index.search(vec, self.cfg.visual_prior_topK)
            ids = I[0]  # [K]
            dists = D[0]  # [K] (inner product scores for IndexFlatIP)
            
            # Apply tau_vis filter if configured
            if self.cfg.visual_prior_tau_vis is not None:
                mask = dists >= self.cfg.visual_prior_tau_vis
                ids = ids[mask]
            
            # Validate IDs: filter negatives and out-of-range
            return self._valid_ids(ids)
        except Exception as e:
            # Graceful fallback if FAISS query fails
            import warnings
            warnings.warn(f"FAISS query failed: {e}. Treating E_vis(I) = [].")
            return np.array([], dtype=np.int32)
    
    def _get_safe_neighbors(
        self,
        cap_ids: np.ndarray,
        vis_ids: np.ndarray,
    ) -> np.ndarray:
        """
        Compute safe neighbors with vectorized masks.
        
        Args:
            cap_ids: 1D int32 array of caption entity IDs (already validated).
            vis_ids: 1D int32 array of visual prior entity IDs (already validated).
        
        Returns:
            1D int32 array of safe neighbor entity IDs (may contain duplicates; dedup in merge).
        """
        if not self.cfg.safe_neighbors_enabled:
            return np.array([], dtype=np.int32)
        
        # Determine seeds based on seed_sources
        seeds = []
        if "caption" in self.cfg.seed_sources:
            seeds.append(cap_ids)
        if "visual_prior" in self.cfg.seed_sources:
            seeds.append(vis_ids)
        
        if not seeds:
            return np.array([], dtype=np.int32)
        
        seed_ids = np.concatenate(seeds, axis=0).astype(np.int32, copy=False)
        # Use stable unique to preserve deterministic ordering
        seed_ids = stable_unique_preserve_order(seed_ids)
        
        if seed_ids.size == 0:
            return np.array([], dtype=np.int32)
        
        # Validate seed IDs before indexing
        seed_ids = self._valid_ids(seed_ids)
        
        if seed_ids.size == 0:
            return np.array([], dtype=np.int32)
        
        # Gather neighbors[seed_ids] → [S, K] int32
        neigh = self.neighbors[seed_ids]  # [S, K]
        mask = np.ones(neigh.shape, dtype=bool)
        
        # Apply mutual k-NN mask
        if self.cfg.use_mutual_knn:
            mask &= self.mutual[seed_ids]
        
        # Apply pixel anchoring (require neighbor in visual prior)
        if self.cfg.require_in_vis_prior:
            if vis_ids.size > 0:
                mask &= pixel_anchoring_mask(neigh, vis_ids)
            else:
                # No visual prior → no safe neighbors pass pixel anchoring
                return np.array([], dtype=np.int32)
        
        # Extract safe neighbors (flatten and return)
        safe = neigh[mask]
        return safe.astype(np.int32, copy=False)
    
    def _merge_candidates(
        self,
        e_cap: np.ndarray,
        e_vis: np.ndarray,
        nbr_safe: np.ndarray,
        e_global: np.ndarray,
    ) -> np.ndarray:
        """
        Merge candidate sources with stable priority and budget cap.
        
        Per-source caps applied before merge.
        Priority order determined by cfg.priority_order.
        Deduplication preserves first occurrence.
        
        Args:
            e_cap: 1D int32 caption entity IDs.
            e_vis: 1D int32 visual prior entity IDs.
            nbr_safe: 1D int32 safe neighbor entity IDs.
            e_global: 1D int32 global entity IDs.
        
        Returns:
            1D int32 array of candidate entity IDs, length ≤ C_max.
        """
        # Apply per-source caps
        e_cap = _cap_source(e_cap, self.cfg.max_caption)
        e_vis = _cap_source(e_vis, self.cfg.max_visual_prior)
        nbr_safe = _cap_source(nbr_safe, self.cfg.max_safe_neighbors)
        e_global = _cap_source(e_global, self.cfg.max_global)
        
        # Build priority-ordered list
        priority_map = {
            "caption": e_cap,
            "visual_prior": e_vis,
            "safe_neighbors": nbr_safe,
            "global": e_global,
        }
        
        # Concatenate in priority order
        sources = [priority_map[src] for src in self.cfg.priority_order if src in priority_map]
        merged = np.concatenate(sources, axis=0) if sources else np.array([], dtype=np.int32)
        
        # Stable unique (preserve first occurrence order)
        merged = stable_unique_preserve_order(merged)
        
        # Apply global budget cap
        merged = merged[:self.cfg.C_max]
        
        return merged


# ============================================================================
# Convenience factory function
# ============================================================================

def create_candidate_generator_from_config(
    config_path: Path,
    entity_context: Optional[dict] = None,
    context_path: Optional[Path] = None,
) -> PhraseCandidateGeneratorV31:
    """
    Create a PhraseCandidateGeneratorV31 from config file.
    
    Loads all required artifacts (neighbors, mutual, FAISS index, global list).
    
    Args:
        config_path: Path to entity_graph.yaml config file.
        entity_context: Optional pre-loaded entity context dict.
                        If None, will load from context_path or config.
        context_path: Optional path to entity_context.json.
                      If None, will use path from config.
    
    Returns:
        Initialized PhraseCandidateGeneratorV31 ready for inference.
    
    Example:
        >>> gen = create_candidate_generator_from_config(
        ...     config_path=Path("configs/entity_graph.yaml")
        ... )
        >>> candidates = gen.generate_for_image(
        ...     cap_ids=np.array([10, 25], dtype=np.int32),
        ...     image_vec=np.random.randn(512).astype(np.float32)
        ... )
    """
    # Convert to Path if string
    if isinstance(config_path, str):
        config_path = Path(config_path)
    
    cfg = load_candidate_gen_config_from_yaml(config_path)
    
    # Load neighbors and mutual
    neighbors, mutual = load_neighbors_artifacts(
        cfg.entity_neighbors_path,
        cfg.entity_neighbors_mutual_path,
    )
    
    # Load FAISS index (graceful fallback if unavailable)
    try:
        faiss_index = load_faiss_index(cfg.entity_faiss_index_path)
        
        # Validate FAISS index metadata
        n_entities = neighbors.shape[0]
        if faiss_index.d != 512:
            raise ValueError(
                f"FAISS index dimension mismatch: expected d=512 (CLIP), got d={faiss_index.d}. "
                f"Index path: {cfg.entity_faiss_index_path}"
            )
        if faiss_index.ntotal != n_entities:
            raise ValueError(
                f"FAISS index size mismatch: expected ntotal={n_entities} (neighbors.shape[0]), "
                f"got ntotal={faiss_index.ntotal}. Index path: {cfg.entity_faiss_index_path}"
            )
    except (ImportError, FileNotFoundError, RuntimeError) as e:
        import warnings
        warnings.warn(f"Could not load FAISS index: {e}. Visual prior will be disabled.")
        faiss_index = None
    
    # Load or compute global list
    if cfg.entity_global_list_path and cfg.entity_global_list_path.exists():
        global_list = np.load(cfg.entity_global_list_path, allow_pickle=False).astype(np.int32, copy=False)
    else:
        # Compute from entity context
        if entity_context is None:
            if context_path is None:
                # Get context_path from config and resolve against repo root
                import yaml
                with open(config_path, "r", encoding="utf-8") as f:
                    full_cfg = yaml.safe_load(f)
                
                # Resolve context_path using same base_dir logic
                base_dir = config_path.resolve().parents[1]
                context_path_str = full_cfg.get("entity_graph", {}).get("context_path", "data/entities/entity_context.json")
                context_path = Path(context_path_str)
                if not context_path.is_absolute():
                    context_path = base_dir / context_path
            
            entity_context = load_entity_context(context_path)
        
        global_list = compute_global_top_entities_from_context(entity_context, cfg.global_topN)
        
        # Persist E_global if path is configured (atomic write)
        if cfg.entity_global_list_path:
            try:
                cfg.entity_global_list_path.parent.mkdir(parents=True, exist_ok=True)
                # Atomic write: use file handle to avoid .npy.tmp.npy bug
                tmp_path = cfg.entity_global_list_path.with_suffix(cfg.entity_global_list_path.suffix + ".tmp")
                with open(tmp_path, "wb") as f:
                    np.save(f, global_list.astype(np.int32, copy=False), allow_pickle=False)
                tmp_path.replace(cfg.entity_global_list_path)
            except Exception as e:
                import warnings
                warnings.warn(f"Could not persist E_global to {cfg.entity_global_list_path}: {e}")
    
    n_entities = neighbors.shape[0]
    
    return PhraseCandidateGeneratorV31(
        cfg=cfg,
        neighbors=neighbors,
        mutual=mutual,
        faiss_index=faiss_index,
        global_list=global_list,
        n_entities=n_entities,
    )
