"""
Entity graph construction module for Phase 4.

This module implements the construction of an entity-only PyG HeteroData graph
from entity vocabulary, entity embeddings, and entity context. The graph contains:
  - A single node type: "entity"
  - Two edge types:
    - ("entity", "sem", "entity"): semantic edges via k-NN in CLIP embedding space
    - ("entity", "cooc", "entity"): co-occurrence edges from shared images/captions

Design invariants:
  - All entity embeddings are CLIP text embeddings (dim=512).
  - Embeddings are float32, L2-normalized, and NaN/Inf-free.
  - Graph stored as torch_geometric.data.HeteroData.
  - Off-graph metadata: entity_vocab.json, entity_context.json.

Phase 4 implementation: Day 5-7 (Week 1-2)
Status: IMPLEMENTED - Graph construction with semantic and co-occurrence edges
"""

from __future__ import annotations

import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import torch
from torch_geometric.data import HeteroData


def cap_degrees(
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    degree_cap: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Cap node degrees by keeping only the highest-weight edges per source node.
    
    For each source node with out-degree > degree_cap, keep only the top
    degree_cap edges sorted by weight descending, then by neighbor id ascending
    for determinism.
    
    Args:
        edge_index: Edge index tensor [2, E] with (src, dst) pairs.
        edge_weight: Edge weight tensor [E].
        degree_cap: Maximum out-degree per source node.
    
    Returns:
        Tuple of (capped_edge_index, capped_edge_weight) with same dtypes.
    """
    if degree_cap is None or degree_cap <= 0:
        return edge_index, edge_weight
    
    src, dst = edge_index
    num_edges = src.size(0)
    
    # Group edges by source node
    src_to_edges = defaultdict(list)
    for edge_idx in range(num_edges):
        src_to_edges[src[edge_idx].item()].append(edge_idx)
    
    # For each source, keep top degree_cap edges
    keep_mask = torch.zeros(num_edges, dtype=torch.bool, device=edge_index.device)
    
    for src_node, edge_indices in src_to_edges.items():
        if len(edge_indices) <= degree_cap:
            # Keep all edges
            for idx in edge_indices:
                keep_mask[idx] = True
        else:
            # Sort by weight desc, then by dst asc for determinism
            edge_data = [
                (edge_weight[idx].item(), dst[idx].item(), idx)
                for idx in edge_indices
            ]
            edge_data.sort(key=lambda x: (-x[0], x[1]))  # weight desc, dst asc
            
            # Keep top degree_cap
            for _, _, idx in edge_data[:degree_cap]:
                keep_mask[idx] = True
    
    # Apply mask
    capped_edge_index = edge_index[:, keep_mask]
    capped_edge_weight = edge_weight[keep_mask]
    
    return capped_edge_index, capped_edge_weight


def build_semantic_edges(
    entity_embeddings: torch.Tensor,
    cfg: Dict[str, Any],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build semantic edges via k-NN over normalized entity embeddings.
    
    Uses chunked similarity computation to avoid OOM on large entity sets.
    For each entity, selects top k_sem neighbors by cosine similarity.
    Applies degree capping to avoid hub explosion.
    
    Args:
        entity_embeddings: CLIP text embeddings for entities, shape [N, d].
                          Must be L2-normalized, float32, no NaN/Inf.
        cfg: Configuration dictionary. Uses cfg["entity_graph"]["k_sem"] and
             cfg["entity_graph"]["degree_cap"].
    
    Returns:
        Tuple of (edge_index, edge_weight):
          - edge_index: [2, E] long tensor of edge indices
          - edge_weight: [E] float tensor of edge weights (cosine similarities)
    """
    N, D = entity_embeddings.shape
    entity_cfg = cfg.get("entity_graph", {})
    k_sem = int(entity_cfg.get("k_sem", 16))
    degree_cap = int(entity_cfg.get("degree_cap", 64))
    chunk_size = int(entity_cfg.get("knn_chunk_size", 2048))
    
    print(f"\n  Building semantic edges:")
    print(f"    k_sem: {k_sem}")
    print(f"    degree_cap: {degree_cap}")
    print(f"    chunk_size: {chunk_size}")
    print(f"    entity_embeddings shape: {list(entity_embeddings.shape)}")
    
    # Validate embeddings are normalized
    sample_size = min(100, N)
    sample_idx = torch.randperm(N)[:sample_size]
    sample_norms = torch.linalg.norm(entity_embeddings[sample_idx], dim=1)
    mean_norm = sample_norms.mean().item()
    
    if abs(mean_norm - 1.0) > 0.01:
        print(f"    Warning: Sample embeddings have mean norm {mean_norm:.4f} (expected ~1.0)")
    
    # Validate dimension
    if D != 512:
        raise RuntimeError(f"Expected embedding dimension 512, got {D}")
    
    # Chunked k-NN computation
    all_src = []
    all_dst = []
    all_weight = []
    
    start_time = time.time()
    
    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        chunk = entity_embeddings[start:end]  # [C, D]
        
        # Compute dot products (cosine similarity for normalized vectors)
        scores = chunk @ entity_embeddings.T  # [C, N]
        
        # Mask self-similarity
        row_idx = torch.arange(start, end, device=scores.device)
        scores[torch.arange(end - start, device=scores.device), row_idx] = -1e9
        
        # Get top-k
        k_actual = min(k_sem, N - 1)  # Don't exceed N-1 neighbors
        topk_vals, topk_idx = torch.topk(scores, k=k_actual, dim=-1)
        
        # Build edge list
        src_ids = torch.arange(start, end, device=scores.device).unsqueeze(1).expand_as(topk_idx)
        all_src.append(src_ids.reshape(-1))
        all_dst.append(topk_idx.reshape(-1))
        all_weight.append(topk_vals.reshape(-1))
        
        if (start // chunk_size + 1) % 5 == 0 or end >= N:
            elapsed = time.time() - start_time
            print(f"    Processed {end}/{N} entities ({elapsed:.1f}s)")
    
    # Concatenate all chunks
    src = torch.cat(all_src, dim=0)
    dst = torch.cat(all_dst, dim=0)
    weight = torch.cat(all_weight, dim=0)
    
    print(f"    Initial edges: {src.size(0)}")
    
    # Symmetrize by adding reverse edges (avoid duplicates)
    # Create a set of (src, dst) pairs to check for existing edges
    edge_set = set()
    for i in range(src.size(0)):
        edge_set.add((src[i].item(), dst[i].item()))
    
    # Add reverse edges that don't already exist
    rev_src_list = []
    rev_dst_list = []
    rev_weight_list = []
    
    for i in range(src.size(0)):
        s, d, w = src[i].item(), dst[i].item(), weight[i].item()
        if (d, s) not in edge_set:
            rev_src_list.append(d)
            rev_dst_list.append(s)
            rev_weight_list.append(w)
            edge_set.add((d, s))
    
    if rev_src_list:
        rev_src = torch.tensor(rev_src_list, dtype=torch.long, device=src.device)
        rev_dst = torch.tensor(rev_dst_list, dtype=torch.long, device=dst.device)
        rev_weight = torch.tensor(rev_weight_list, dtype=torch.float32, device=weight.device)
        
        src = torch.cat([src, rev_src], dim=0)
        dst = torch.cat([dst, rev_dst], dim=0)
        weight = torch.cat([weight, rev_weight], dim=0)
    
    print(f"    After symmetrization: {src.size(0)} edges")
    
    # Apply degree capping
    edge_index = torch.stack([src, dst], dim=0)
    edge_index, weight = cap_degrees(edge_index, weight, degree_cap)
    
    print(f"    After degree capping: {edge_index.size(1)} edges")
    
    return edge_index, weight


def build_cooccurrence_edges(
    entity_context: Dict[int, Dict[str, Any]],
    cfg: Dict[str, Any],
    num_entities: int = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build co-occurrence edges from entity_context.
    
    For each image and caption, counts co-occurrence between entity pairs.
    Edge weight = total co-occurrence count across all images/captions.
    Applies degree capping to avoid hubs.
    
    Args:
        entity_context: Mapping from entity_id to dict with:
                        {"entity": str, "image_ids": List[str], "caption_ids": List[str]}
                        Keys must be integers (0-based entity IDs) or strings parseable as integers.
                        Entity IDs must be in range [0, num_entities) where num_entities is the
                        total number of entities in the vocabulary.
        cfg: Configuration dictionary. Uses cfg["entity_graph"]["degree_cap"].
        num_entities: Total number of entities. If None, inferred from entity_context keys.
                     Used for validation that entity IDs are within valid range.
    
    Returns:
        Tuple of (edge_index, edge_weight):
          - edge_index: [2, E] long tensor of edge indices
          - edge_weight: [E] float tensor of edge weights (co-occurrence counts)
    
    Raises:
        ValueError: If entity_context keys cannot be coerced to valid integer entity IDs.
    """
    entity_cfg = cfg.get("entity_graph", {})
    degree_cap = int(entity_cfg.get("degree_cap", 64))
    
    # Normalize entity_context keys to int if necessary
    if not all(isinstance(k, int) for k in entity_context.keys()):
        print(f"\n  [robustness] Normalizing entity_context keys to int...")
        normalized: Dict[int, Dict[str, Any]] = {}
        for k, v in entity_context.items():
            if isinstance(k, int):
                ent_id = k
            elif isinstance(k, str):
                if k.isdigit() or (k.startswith('-') and k[1:].isdigit()):
                    ent_id = int(k)
                else:
                    raise ValueError(
                        f"Invalid entity_context key '{k}' (type {type(k).__name__}): "
                        f"expected int or digit-parseable string. "
                        f"Please normalize entity_context keys upstream."
                    )
            else:
                raise ValueError(
                    f"Invalid entity_context key type {type(k).__name__} for key {k!r}: "
                    f"expected int or digit-parseable string."
                )
            
            # Validate range if num_entities is provided
            if num_entities is not None and not (0 <= ent_id < num_entities):
                raise ValueError(
                    f"Entity ID {ent_id} out of valid range [0, {num_entities}). "
                    f"Check entity_context consistency with entity_embeddings."
                )
            
            normalized[ent_id] = v
        
        entity_context = normalized
        print(f"    ✓ Normalized {len(entity_context)} entity IDs")
    
    print(f"\n  Building co-occurrence edges:")
    print(f"    degree_cap: {degree_cap}")
    print(f"    num_entities: {len(entity_context)}")
    
    # Build reverse maps: image_id -> set(entity_ids), caption_id -> set(entity_ids)
    image_to_entities = defaultdict(set)
    caption_to_entities = defaultdict(set)
    
    for ent_id, ctx in entity_context.items():
        for img_id in ctx.get("image_ids", []):
            image_to_entities[img_id].add(ent_id)
        for cap_id in ctx.get("caption_ids", []):
            caption_to_entities[cap_id].add(ent_id)
    
    print(f"    num_images with entities: {len(image_to_entities)}")
    print(f"    num_captions with entities: {len(caption_to_entities)}")
    
    # Count co-occurrences
    pair_weights = defaultdict(float)
    
    def accumulate_pairs(mapping, label):
        pair_count = 0
        for _, ents in mapping.items():
            ents_sorted = sorted(ents)
            for i in range(len(ents_sorted)):
                for j in range(i + 1, len(ents_sorted)):
                    ei, ej = ents_sorted[i], ents_sorted[j]
                    pair_weights[(ei, ej)] += 1.0
                    pair_weights[(ej, ei)] += 1.0  # symmetric
                    pair_count += 2
        print(f"      {label}: {pair_count} co-occurrence pairs")
    
    accumulate_pairs(image_to_entities, "images")
    accumulate_pairs(caption_to_entities, "captions")
    
    print(f"    Total unique directed edges: {len(pair_weights)}")
    
    if not pair_weights:
        print(f"    Warning: No co-occurrence edges found")
        # Return empty edges
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_weight = torch.zeros(0, dtype=torch.float32)
        return edge_index, edge_weight
    
    # Convert to tensors
    src_list, dst_list, w_list = [], [], []
    for (ei, ej), w in pair_weights.items():
        src_list.append(ei)
        dst_list.append(ej)
        w_list.append(w)
    
    src = torch.tensor(src_list, dtype=torch.long)
    dst = torch.tensor(dst_list, dtype=torch.long)
    weight = torch.tensor(w_list, dtype=torch.float32)
    
    print(f"    Initial edges: {src.size(0)}")
    
    # Apply degree capping
    edge_index = torch.stack([src, dst], dim=0)
    edge_index, weight = cap_degrees(edge_index, weight, degree_cap)
    
    print(f"    After degree capping: {edge_index.size(1)} edges")
    
    return edge_index, weight


def build_entity_graph(
    entity_embeddings: torch.Tensor,
    entity_context: Dict[int, Dict[str, Any]],
    cfg: Dict[str, Any],
) -> HeteroData:
    """
    Build an entity-only PyG HeteroData graph with semantic and co-occurrence edges.
    
    Constructs a graph with:
      - data["entity"].x: CLIP entity embeddings [N_entities, d_model]
      - data["entity", "sem", "entity"].edge_index: semantic edges [2, E_sem]
      - data["entity", "sem", "entity"].edge_weight: semantic edge weights [E_sem]
      - data["entity", "cooc", "entity"].edge_index: co-occurrence edges [2, E_cooc]
      - data["entity", "cooc", "entity"].edge_weight: co-occurrence edge weights [E_cooc]
    
    Args:
        entity_embeddings: CLIP text embeddings for all entities, shape [N, d].
                           Must be float32, L2-normalized, NaN/Inf-free.
        entity_context: Mapping from entity_id (int) to dict containing:
                        {"entity": str, "image_ids": List[str], "caption_ids": List[str]}
        cfg: Configuration dictionary. Use cfg["entity_graph"] for parameters:
             - k_sem: number of nearest neighbors for semantic edges
             - degree_cap: maximum degree per node
             - entity_graph_path: path to save the graph
    
    Returns:
        HeteroData graph with entity nodes and semantic/co-occurrence edges.
    
    Raises:
        ValueError: If embeddings contain NaNs or Infs, or wrong dtype/shape.
        RuntimeError: If embedding dimension is not 512.
    """
    print("\n" + "=" * 70)
    print("BUILDING ENTITY GRAPH")
    print("=" * 70)
    
    # Validate embeddings
    if entity_embeddings.dtype != torch.float32:
        raise ValueError(f"Expected float32 embeddings, got {entity_embeddings.dtype}")
    
    if not torch.isfinite(entity_embeddings).all():
        num_bad = (~torch.isfinite(entity_embeddings)).sum().item()
        raise ValueError(f"Entity embeddings contain {num_bad} non-finite values (NaNs/Infs)")
    
    N, D = entity_embeddings.shape
    if D != 512:
        raise RuntimeError(f"Expected embedding dimension 512, got {D}")
    
    print(f"\n  Entity embeddings:")
    print(f"    shape: {list(entity_embeddings.shape)}")
    print(f"    dtype: {entity_embeddings.dtype}")
    print(f"    device: {entity_embeddings.device}")
    
    # Build edges
    start_time = time.time()
    
    sem_edge_index, sem_edge_weight = build_semantic_edges(entity_embeddings, cfg)
    cooc_edge_index, cooc_edge_weight = build_cooccurrence_edges(
        entity_context, cfg, num_entities=N
    )
    
    build_time = time.time() - start_time
    
    # Construct HeteroData
    data = HeteroData()
    data["entity"].x = entity_embeddings  # [N, d]
    
    data["entity", "sem", "entity"].edge_index = sem_edge_index
    data["entity", "sem", "entity"].edge_weight = sem_edge_weight
    
    data["entity", "cooc", "entity"].edge_index = cooc_edge_index
    data["entity", "cooc", "entity"].edge_weight = cooc_edge_weight
    
    # Compute statistics
    num_entities = N
    num_sem_edges = sem_edge_index.size(1)
    num_cooc_edges = cooc_edge_index.size(1)
    
    # Degree statistics for semantic edges
    if num_sem_edges > 0:
        sem_src = sem_edge_index[0]
        sem_degrees = torch.bincount(sem_src, minlength=num_entities)
        sem_degree_mean = sem_degrees.float().mean().item()
        sem_degree_max = sem_degrees.max().item()
        sem_degree_min = sem_degrees.min().item()
    else:
        sem_degree_mean = sem_degree_max = sem_degree_min = 0
    
    # Degree statistics for co-occurrence edges
    if num_cooc_edges > 0:
        cooc_src = cooc_edge_index[0]
        cooc_degrees = torch.bincount(cooc_src, minlength=num_entities)
        cooc_degree_mean = cooc_degrees.float().mean().item()
        cooc_degree_max = cooc_degrees.max().item()
        cooc_degree_min = cooc_degrees.min().item()
    else:
        cooc_degree_mean = cooc_degree_max = cooc_degree_min = 0
    
    # Memory estimation
    node_mem_mb = (num_entities * 512 * 4) / (1024 ** 2)  # float32
    sem_edge_mem_mb = (num_sem_edges * 2 * 8 + num_sem_edges * 4) / (1024 ** 2)  # int64 + float32
    cooc_edge_mem_mb = (num_cooc_edges * 2 * 8 + num_cooc_edges * 4) / (1024 ** 2)
    total_mem_mb = node_mem_mb + sem_edge_mem_mb + cooc_edge_mem_mb
    
    # Print summary
    print("\n" + "=" * 70)
    print("GRAPH CONSTRUCTION SUMMARY")
    print("=" * 70)
    print(f"\n  Build time: {build_time:.2f}s")
    print(f"\n  Nodes:")
    print(f"    entity: {num_entities}")
    print(f"\n  Edges:")
    print(f"    semantic (sem):        {num_sem_edges}")
    print(f"      avg degree:          {sem_degree_mean:.2f}")
    print(f"      min/max degree:      {sem_degree_min} / {sem_degree_max}")
    print(f"    co-occurrence (cooc):  {num_cooc_edges}")
    print(f"      avg degree:          {cooc_degree_mean:.2f}")
    print(f"      min/max degree:      {cooc_degree_min} / {cooc_degree_max}")
    print(f"\n  Memory estimate:")
    print(f"    node features:         {node_mem_mb:.2f} MB")
    print(f"    semantic edges:        {sem_edge_mem_mb:.2f} MB")
    print(f"    co-occurrence edges:   {cooc_edge_mem_mb:.2f} MB")
    print(f"    total:                 {total_mem_mb:.2f} MB")
    print("=" * 70)
    
    return data


def save_entity_graph(data: HeteroData, path: Union[str, Path]) -> None:
    """
    Save HeteroData graph to disk using torch.save.
    
    Creates parent directory if needed. Overwrites existing file.
    
    Args:
        data: HeteroData graph to save.
        path: Output file path (typically ends in .pt).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\n  Saving graph to: {path}")
    torch.save(data, str(path))
    
    # Check file size
    file_size_mb = path.stat().st_size / (1024 ** 2)
    print(f"  File size: {file_size_mb:.2f} MB")
    print(f"  ✓ Graph saved successfully")


def load_entity_graph(path: Union[str, Path]) -> HeteroData:
    """
    Load HeteroData graph from disk using torch.load.
    
    Args:
        path: Path to the saved graph file (typically ends in .pt).
    
    Returns:
        Loaded HeteroData graph.
    
    Raises:
        FileNotFoundError: If the graph file does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Graph file not found: {path}")
    
    print(f"\n  Loading graph from: {path}")
    data = torch.load(str(path), map_location="cpu", weights_only=False)
    
    # Print summary
    num_entities = data["entity"].x.size(0)
    num_sem_edges = data["entity", "sem", "entity"].edge_index.size(1)
    num_cooc_edges = data["entity", "cooc", "entity"].edge_index.size(1)
    
    print(f"  ✓ Loaded graph:")
    print(f"    entity nodes: {num_entities}")
    print(f"    semantic edges: {num_sem_edges}")
    print(f"    co-occurrence edges: {num_cooc_edges}")
    
    return data


# ============================================================================
# Helper functions
# ============================================================================

def l2_normalize(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    L2-normalize a tensor row-wise (along the last dimension).

    This helper is used for normalizing entity embeddings before k-NN
    and graph search.

    Args:
        x: Input tensor, shape [..., d].
        eps: Small epsilon to avoid division by zero.

    Returns:
        L2-normalized tensor with same shape as input.

    Example:
        >>> embeddings = torch.randn(100, 512)
        >>> embeddings_norm = l2_normalize(embeddings)
        >>> assert torch.allclose(embeddings_norm.norm(dim=-1), torch.ones(100))
    """
    return x / (x.norm(dim=-1, keepdim=True) + eps)

