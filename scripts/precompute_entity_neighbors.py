"""
Precompute entity semantic neighbors and mutual k-NN mask (Phase 5 Day 4).

This script:
  1. Loads entity embeddings from disk
  2. Builds FAISS IndexFlatIP over normalized embeddings
  3. Computes top-K semantic neighbors for all entities (excluding self)
  4. Computes mutual k-NN mask
  5. Persists artifacts to disk for fast Day 5 candidate generation

Resume-safe: skips computation if outputs exist unless --overwrite is set.
Atomic writes: writes to temp files then replaces to avoid partial outputs.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys
import time
from typing import Dict, Any

import numpy as np

# Add repo root to path
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from src.graph.config import load_entity_graph_config, get_entity_graph_config
from src.graph.entity_index import (
    load_entity_embeddings,
    normalize_rows,
    build_faiss_ip_index,
    save_faiss_index,
    compute_semantic_neighbors,
    compute_mutual_knn_mask,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Precompute entity semantic neighbors and mutual k-NN mask (Phase 5 Day 4)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/entity_graph.yaml",
        help="Path to entity graph config YAML (default: configs/entity_graph.yaml)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing artifacts (default: skip if exists)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--write-stats",
        action="store_true",
        help="Write diagnostics JSON to neighbors_path.with_suffix('.stats.json')",
    )
    return parser.parse_args()


def check_artifacts_exist(
    faiss_path: Path,
    neighbors_path: Path,
    mutual_path: Path,
) -> bool:
    """Check if all output artifacts already exist."""
    return faiss_path.exists() and neighbors_path.exists() and mutual_path.exists()


def get_embeddings_metadata(embeddings_path: Path) -> Dict[str, Any]:
    """
    Get embeddings file metadata for staleness detection.
    
    Args:
        embeddings_path: Path to embeddings file.
    
    Returns:
        Dict with path, size, mtime.
    """
    stat = embeddings_path.stat()
    return {
        "path": str(embeddings_path.resolve()),
        "size_bytes": stat.st_size,
        "mtime": stat.st_mtime,
    }


def create_manifest(
    embeddings_path: Path,
    n: int,
    d: int,
    k_neighbors: int,
    normalize: bool,
) -> Dict[str, Any]:
    """
    Create artifact manifest for staleness detection.
    
    Args:
        embeddings_path: Path to embeddings file.
        n: Number of entities.
        d: Embedding dimension.
        k_neighbors: K neighbors parameter.
        normalize: Whether embeddings were normalized.
    
    Returns:
        Manifest dict with embeddings metadata and config.
    """
    return {
        "embeddings": get_embeddings_metadata(embeddings_path),
        "n_entities": n,
        "embedding_dim": d,
        "k_neighbors": k_neighbors,
        "normalize_embeddings": normalize,
    }


def validate_manifest(
    manifest: Dict[str, Any],
    embeddings_path: Path,
    expected_n: int,
    expected_d: int,
    expected_k: int,
    expected_normalize: bool,
) -> bool:
    """
    Validate manifest matches current embeddings and config.
    
    Args:
        manifest: Loaded manifest dict.
        embeddings_path: Current embeddings path.
        expected_n: Expected number of entities.
        expected_d: Expected embedding dimension.
        expected_k: Expected K_neighbors.
        expected_normalize: Expected normalize flag.
    
    Returns:
        True if manifest matches, False otherwise.
    """
    # Check config parameters
    if manifest.get("n_entities") != expected_n:
        logger.warning(
            f"Manifest n_entities mismatch: {manifest.get('n_entities')} != {expected_n}"
        )
        return False
    if manifest.get("embedding_dim") != expected_d:
        logger.warning(
            f"Manifest embedding_dim mismatch: {manifest.get('embedding_dim')} != {expected_d}"
        )
        return False
    if manifest.get("k_neighbors") != expected_k:
        logger.warning(
            f"Manifest k_neighbors mismatch: {manifest.get('k_neighbors')} != {expected_k}"
        )
        return False
    if manifest.get("normalize_embeddings") != expected_normalize:
        logger.warning(
            f"Manifest normalize_embeddings mismatch: {manifest.get('normalize_embeddings')} != {expected_normalize}"
        )
        return False
    
    # Check embeddings file metadata
    current_meta = get_embeddings_metadata(embeddings_path)
    manifest_meta = manifest.get("embeddings", {})
    
    if manifest_meta.get("path") != current_meta["path"]:
        logger.warning(
            f"Embeddings path changed: {manifest_meta.get('path')} != {current_meta['path']}"
        )
        return False
    if manifest_meta.get("size_bytes") != current_meta["size_bytes"]:
        logger.warning(
            f"Embeddings size changed: {manifest_meta.get('size_bytes')} != {current_meta['size_bytes']} bytes"
        )
        return False
    if manifest_meta.get("mtime") != current_meta["mtime"]:
        logger.warning(
            f"Embeddings mtime changed: {manifest_meta.get('mtime')} != {current_meta['mtime']}"
        )
        return False
    
    return True


def validate_artifact_staleness(
    neighbors_path: Path,
    mutual_path: Path,
    faiss_path: Path,
    embeddings_path: Path,
    expected_n: int,
    expected_k: int,
    expected_dim: int,
    expected_normalize: bool,
) -> bool:
    """
    Validate that existing artifacts match current config and embeddings.
    
    Checks:
      - Manifest: embeddings metadata (path, size, mtime) and config (N, D, K, normalize)
      - Neighbors/mutual: shape, dtype, no self-neighbors, valid range
      - FAISS index: ntotal, dimension
    
    Args:
        neighbors_path: Path to neighbors .npy file.
        mutual_path: Path to mutual mask .npy file.
        faiss_path: Path to FAISS index file.
        embeddings_path: Path to embeddings file.
        expected_n: Expected number of entities.
        expected_k: Expected K_neighbors.
        expected_dim: Expected embedding dimension.
        expected_normalize: Expected normalize_embeddings flag.
    
    Returns:
        True if artifacts are valid and match config, False otherwise.
    """
    try:
        from src.graph.entity_index import (
            load_neighbor_artifacts,
            load_faiss_index,
            validate_faiss_index,
        )
        
        # Check manifest first (most important: detects stale embeddings)
        manifest_path = neighbors_path.with_suffix(".manifest.json")
        if not manifest_path.exists():
            logger.warning(f"Manifest not found: {manifest_path}. Artifacts may be stale.")
            return False
        
        with manifest_path.open("r", encoding="utf-8") as f:
            manifest = json.load(f)
        
        if not validate_manifest(
            manifest, embeddings_path, expected_n, expected_dim, expected_k, expected_normalize
        ):
            logger.warning("Manifest validation failed. Embeddings or config changed.")
            return False
        
        logger.info("Manifest validated: embeddings and config match.")
        
        # Validate neighbors/mutual
        neighbors, mutual = load_neighbor_artifacts(neighbors_path, mutual_path)
        
        # Check shape match
        if neighbors.shape != (expected_n, expected_k):
            logger.warning(
                f"Neighbors shape mismatch: expected ({expected_n}, {expected_k}), "
                f"got {neighbors.shape}"
            )
            return False
        
        logger.info(
            f"Neighbors/mutual validated: shape=({expected_n}, {expected_k}), "
            f"mutual_rate={mutual.mean():.4f}"
        )
        
        # Validate FAISS index
        try:
            index = load_faiss_index(faiss_path)
            validate_faiss_index(index, expected_n, expected_dim)
            logger.info(
                f"FAISS index validated: ntotal={index.ntotal}, dim={index.d}"
            )
        except ImportError:
            logger.error(
                "FAISS is not installed. Cannot validate existing FAISS index. "
                "Either install FAISS (pip install faiss-cpu) or rerun with --overwrite."
            )
            return False
        
        return True
        
    except Exception as e:
        logger.warning(f"Artifact validation failed: {e}")
        return False


def validate_embeddings(embeddings: np.ndarray, expected_dim: int = 512) -> None:
    """
    Validate entity embeddings shape, dtype, and values.
    
    Args:
        embeddings: Entity embeddings array.
        expected_dim: Expected embedding dimension (default: 512 for CLIP).
    
    Raises:
        ValueError: If shape, dtype, or values are invalid.
    """
    if embeddings.ndim != 2:
        raise ValueError(f"Expected 2D embeddings, got shape {embeddings.shape}")
    
    n, d = embeddings.shape
    if d != expected_dim:
        raise ValueError(
            f"Expected embedding dim {expected_dim}, got {d}. "
            "Phase 5 requires CLIP-space embeddings (dim=512)."
        )
    
    # Check for NaN/inf values (corrupted embeddings)
    if not np.isfinite(embeddings).all():
        nan_count = np.isnan(embeddings).sum()
        inf_count = np.isinf(embeddings).sum()
        raise ValueError(
            f"Embeddings contain non-finite values: {nan_count} NaNs, {inf_count} infs. "
            "Embeddings file may be corrupted. Check upstream embedding generation."
        )
    
    logger.info(f"Embeddings shape: {embeddings.shape}, dtype: {embeddings.dtype}")


def validate_computed_neighbors(
    neighbors: np.ndarray,
    mutual: np.ndarray,
    n: int,
    k: int,
) -> None:
    """
    Validate computed neighbors and mutual mask before writing to disk.
    
    Enforces critical invariants:
      - Correct shape and dtype
      - No self-neighbors
      - Valid ID range
      - No -1 padding
    
    Args:
        neighbors: Computed neighbors array.
        mutual: Computed mutual mask.
        n: Number of entities.
        k: Number of neighbors per entity.
    
    Raises:
        ValueError: If any validation check fails.
    """
    # Shape and dtype
    if neighbors.shape != (n, k):
        raise ValueError(
            f"Neighbors shape must be ({n}, {k}), got {neighbors.shape}"
        )
    if neighbors.dtype != np.int32:
        raise ValueError(
            f"Neighbors dtype must be int32, got {neighbors.dtype}"
        )
    if mutual.shape != (n, k):
        raise ValueError(
            f"Mutual mask shape must be ({n}, {k}), got {mutual.shape}"
        )
    if mutual.dtype != np.bool_:
        raise ValueError(
            f"Mutual mask dtype must be bool, got {mutual.dtype}"
        )
    
    # No -1 padding allowed
    if (neighbors < 0).any():
        raise ValueError(
            f"Neighbors contains negative IDs (padding). Min value: {neighbors.min()}. "
            "This indicates failed neighbor computation."
        )
    
    # Valid range
    if neighbors.max() >= n:
        raise ValueError(
            f"Neighbors contains out-of-range IDs: max={neighbors.max()} >= N={n}"
        )
    
    # No self-neighbors (critical)
    ids = np.arange(n, dtype=np.int32)[:, None]
    if (neighbors == ids).any():
        raise ValueError(
            "Neighbors contains self-neighbors. This will corrupt Day 5 safe_neighbors."
        )
    
    logger.info("Post-compute validation passed: neighbors and mutual are valid.")


def atomic_write_faiss(index: Any, path: Path) -> None:
    """
    Atomically write FAISS index to disk (write temp then replace).
    
    Args:
        index: FAISS index object.
        path: Output path for FAISS index.
    """
    import tempfile
    import shutil
    
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Write to system temp directory first to avoid path encoding issues with FAISS
    with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.index') as tmp:
        tmp_path_str = tmp.name
    
    try:
        # FAISS write to temp location
        save_faiss_index(index, Path(tmp_path_str))
        # Move to final location
        shutil.move(tmp_path_str, str(path))
    finally:
        # Clean up temp file if it still exists
        if Path(tmp_path_str).exists():
            Path(tmp_path_str).unlink()


def atomic_write_npy(arr: np.ndarray, path: Path) -> None:
    """
    Atomically write numpy array to disk (write temp then replace).
    
    Uses file handle to control exact filename and avoid numpy auto-appending .npy.
    
    Args:
        arr: Numpy array to save.
        path: Output path (must have .npy suffix).
    
    Raises:
        ValueError: If path does not end with .npy.
    """
    if path.suffix != ".npy":
        raise ValueError(f"Path must have .npy suffix, got: {path}")
    
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    
    # Write with file handle to control exact filename
    with tmp_path.open("wb") as f:
        np.save(f, arr, allow_pickle=False)
    
    tmp_path.replace(path)


def atomic_write_json(data: Dict[str, Any], path: Path) -> None:
    """
    Atomically write JSON to disk (write temp then replace).
    
    Args:
        data: Dictionary to serialize.
        path: Output path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.parent / (path.name + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    tmp_path.replace(path)


def main() -> None:
    """Main entry point."""
    args = parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    logger.info("=" * 80)
    logger.info("Precompute Entity Semantic Neighbors (Phase 5 Day 4)")
    logger.info("=" * 80)
    
    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        sys.exit(1)
    
    logger.info(f"Loading config from: {config_path}")
    cfg = load_entity_graph_config(config_path)
    entity_cfg = get_entity_graph_config(cfg)
    
    # Extract config parameters with type casting
    # Resolve relative paths against repo root to avoid surprises
    embeddings_path = Path(entity_cfg.get("entity_embeddings_path", "data/entities/entity_embeddings.pt"))
    if not embeddings_path.is_absolute():
        embeddings_path = _REPO_ROOT / embeddings_path
    
    faiss_path = Path(entity_cfg.get("entity_faiss_index_path", "data/entities/entity_faiss.index"))
    if not faiss_path.is_absolute():
        faiss_path = _REPO_ROOT / faiss_path
    
    neighbors_path = Path(entity_cfg.get("entity_neighbors_path", "data/entities/entity_neighbors.npy"))
    if not neighbors_path.is_absolute():
        neighbors_path = _REPO_ROOT / neighbors_path
    
    mutual_path = Path(entity_cfg.get("entity_neighbors_mutual_path", "data/entities/entity_neighbors_mutual.npy"))
    if not mutual_path.is_absolute():
        mutual_path = _REPO_ROOT / mutual_path
    
    # Type cast and validate config values
    try:
        k_neighbors = int(entity_cfg.get("K_neighbors", 16))
        batch_size = int(entity_cfg.get("faiss_batch_size", 4096))
    except (TypeError, ValueError) as e:
        logger.error(f"Invalid config: K_neighbors and batch_size must be integers: {e}")
        sys.exit(1)
    
    normalize = entity_cfg.get("normalize_embeddings", True)
    if not isinstance(normalize, bool):
        logger.error(
            f"Invalid config: normalize_embeddings must be bool, got {type(normalize).__name__}"
        )
        sys.exit(1)
    
    logger.info(f"Resolved paths:")
    logger.info(f"  Entity embeddings: {embeddings_path.resolve()}")
    logger.info(f"  FAISS index: {faiss_path.resolve()}")
    logger.info(f"  Neighbors: {neighbors_path.resolve()}")
    logger.info(f"  Mutual mask: {mutual_path.resolve()}")
    logger.info(f"Config parameters:")
    logger.info(f"  K_neighbors: {k_neighbors}")
    logger.info(f"  Batch size: {batch_size}")
    logger.info(f"  Normalize embeddings: {normalize}")
    
    # Load embeddings first to get N for staleness validation
    logger.info("Loading entity embeddings...")
    start_time = time.time()
    embeddings = load_entity_embeddings(embeddings_path, expected_dim=512)
    logger.info(f"Loaded embeddings in {time.time() - start_time:.2f}s")
    
    # Validate
    validate_embeddings(embeddings, expected_dim=512)
    n, d = embeddings.shape
    
    if k_neighbors >= n:
        raise ValueError(
            f"K_neighbors={k_neighbors} must be < N={n}. "
            "Reduce K_neighbors in config."
        )
    
    # Staleness validation gate: check if artifacts exist and match config
    if check_artifacts_exist(faiss_path, neighbors_path, mutual_path) and not args.overwrite:
        logger.info("Artifacts already exist. Validating staleness...")
        if validate_artifact_staleness(
            neighbors_path, mutual_path, faiss_path, embeddings_path,
            n, k_neighbors, d, normalize
        ):
            logger.info("Artifacts verified (manifest/neighbors/mutual/faiss). Skipping computation.")
            logger.info("Use --overwrite to force recomputation.")
            sys.exit(0)
        else:
            logger.error(
                "Artifacts exist but do not match current config/embeddings. "
                "Rerun with --overwrite to recompute."
            )
            sys.exit(2)
    
    # Proceeding with computation
    if args.overwrite:
        logger.info("--overwrite flag set. Recomputing all artifacts.")
    
    # Normalize embeddings (required for cosine similarity)
    if not normalize:
        logger.error(
            "normalize_embeddings=False is not supported. "
            "Phase 5 requires L2-normalized embeddings for cosine similarity. "
            "Set normalize_embeddings=true in config."
        )
        sys.exit(1)
    
    logger.info("L2-normalizing embeddings for cosine similarity...")
    embeddings_norm = normalize_rows(embeddings)
    
    # Build FAISS index in memory (do NOT write yet)
    logger.info("Building FAISS IndexFlatIP with norm validation...")
    start_time = time.time()
    index = build_faiss_ip_index(embeddings_norm, validate_norms=True)
    logger.info(f"Built FAISS index in {time.time() - start_time:.2f}s")
    logger.info(f"Index size: {index.ntotal} vectors")
    
    # Compute neighbors (in memory, do NOT write yet)
    logger.info(f"Computing top-{k_neighbors} semantic neighbors (batch_size={batch_size})...")
    start_time = time.time()
    neighbors = compute_semantic_neighbors(index, embeddings_norm, k=k_neighbors, batch_size=batch_size)
    logger.info(f"Computed neighbors in {time.time() - start_time:.2f}s")
    logger.info(f"Neighbors shape: {neighbors.shape}, dtype: {neighbors.dtype}")
    
    # Compute mutual k-NN mask (in memory, do NOT write yet)
    logger.info("Computing mutual k-NN mask...")
    start_time = time.time()
    mutual = compute_mutual_knn_mask(neighbors)
    logger.info(f"Computed mutual mask in {time.time() - start_time:.2f}s")
    logger.info(f"Mutual mask shape: {mutual.shape}, dtype: {mutual.dtype}")
    
    # Validate BOTH neighbors and mutual before writing (prevents partial artifacts)
    logger.info("Validating computed artifacts...")
    validate_computed_neighbors(neighbors, mutual, n, k_neighbors)
    
    # Write all three artifacts atomically (only after validation passes)
    logger.info(f"Saving FAISS index to {faiss_path}...")
    start_time = time.time()
    atomic_write_faiss(index, faiss_path)
    logger.info(f"Saved FAISS index in {time.time() - start_time:.2f}s")
    
    logger.info(f"Saving neighbors to {neighbors_path}...")
    start_time = time.time()
    atomic_write_npy(neighbors, neighbors_path)
    logger.info(f"Saved neighbors in {time.time() - start_time:.2f}s")
    
    logger.info(f"Saving mutual mask to {mutual_path}...")
    start_time = time.time()
    atomic_write_npy(mutual, mutual_path)
    logger.info(f"Saved mutual mask in {time.time() - start_time:.2f}s")
    
    # Write manifest for staleness detection
    logger.info("Writing manifest for staleness detection...")
    manifest = create_manifest(embeddings_path, n, d, k_neighbors, normalize)
    manifest_path = neighbors_path.with_suffix(".manifest.json")
    atomic_write_json(manifest, manifest_path)
    logger.info(f"Wrote manifest to {manifest_path}")
    
    # Write diagnostics JSON (optional)
    mutual_rate = float(mutual.mean())
    if args.write_stats:
        diagnostics = {
            "n_entities": int(n),
            "embedding_dim": int(d),
            "k_neighbors": int(k_neighbors),
            "neighbors_dtype": str(neighbors.dtype),
            "mutual_dtype": str(mutual.dtype),
            "neighbor_id_min": int(neighbors.min()),
            "neighbor_id_max": int(neighbors.max()),
            "mutual_rate": mutual_rate,
        }
        
        diagnostics_path = neighbors_path.with_suffix(".stats.json")
        logger.info(f"Writing diagnostics to {diagnostics_path}...")
        atomic_write_json(diagnostics, diagnostics_path)
    
    logger.info("=" * 80)
    logger.info("Precomputation complete!")
    logger.info(f"  N entities: {n}")
    logger.info(f"  K neighbors: {k_neighbors}")
    logger.info(f"  Mutual k-NN rate: {mutual_rate:.4f}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
