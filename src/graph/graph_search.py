"""
Graph search module for Phase 4 entity-centric retrieval.

This module implements query enrichment and graph search over the entity graph.
Key components:
  - Query enrichment: expand queries using entity vocabulary and CLIP search
  - Graph search: bounded beam search over entity graph with decay and edge weighting
  - Entity-to-image score aggregation: map entity scores to image scores

Design invariants:
  - Query enrichment is mandatory in graph mode (except for controlled ablations).
  - Graph search uses LightRAG-style scoring with decay and edge-type weights.
  - All embeddings (query, entity) are in the same CLIP space (dim=512, L2-normalized).

Phase 4 implementation plan: Day 0 - Skeleton only
Full implementation: Later days (Week 2)
"""

from __future__ import annotations

import collections
import heapq
import json
import logging
import math
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Set

import numpy as np
import torch
import torch.nn.functional as F


# ============================================================================
# Data structures for query enrichment and graph search results
# ============================================================================

@dataclass
class EnrichmentResult:
    """
    Result of query enrichment.

    Attributes:
        original_query: Original query string provided by the user.
        enriched_query: Enriched query string with entity names appended.
        q0: Original query embedding from CLIP (shape [512] or [1, 512]).
        q_enriched: Enriched query embedding from CLIP (shape [512] or [1, 512]).
        entity_ids: List of entity IDs selected for enrichment.
        entity_names: List of human-readable entity names.
        entity_scores: Tensor of scores for selected entities (shape [num_entities]).
    """
    original_query: str
    enriched_query: str
    q0: torch.Tensor
    q_enriched: torch.Tensor
    entity_ids: List[int]
    entity_names: List[str]
    entity_scores: torch.Tensor


@dataclass
class GraphSearchResult:
    """
    Result of graph search over the entity graph.

    Attributes:
        original_query: Original query string.
        enriched_query: Enriched query string with entity names appended.
        q_enriched: Enriched query embedding (1D tensor, shape [512]).
        seed_entity_ids: List of entity IDs selected as seeds.
        seed_entity_scores: Tensor of seed entity similarity scores (shape [K_seed]).
        entity_ids: List of all entity IDs after expansion.
        entity_scores: Tensor of entity scores after expansion (shape [N_entities_kept]).
        image_scores: List of (image_id, kg_score) tuples sorted by score descending.
        runtime_ms: Total runtime in milliseconds (seeding + expansion + aggregation).
    """
    original_query: str
    enriched_query: str
    q_enriched: torch.Tensor  # [512]
    seed_entity_ids: List[int]
    seed_entity_scores: torch.Tensor  # [K_seed]
    entity_ids: List[int]
    entity_scores: torch.Tensor  # [N_entities_kept]
    image_scores: List[Tuple[str, float]]  # sorted (image_id, kg_score)
    runtime_ms: float


# ============================================================================
# Query enrichment
# ============================================================================

# Module-level caches for lazy loading
_entity_embeddings_cache: Optional[torch.Tensor] = None
_entity_meta_cache: Optional[Dict[int, Dict[str, Any]]] = None
_entity_context_cache: Optional[Dict[int, Dict[str, Any]]] = None
_reverse_index_cache: Optional[Tuple[Dict[str, Set[int]], Dict[str, Set[int]]]] = None

logger = logging.getLogger(__name__)


def _l2_normalize(x: torch.Tensor) -> torch.Tensor:
    """
    L2-normalize a tensor along the last dimension.
    
    Args:
        x: Tensor of shape [d] or [N, d]
    
    Returns:
        L2-normalized tensor of same shape
    """
    if x.ndim == 1:
        x = x.unsqueeze(0)
        normalized = F.normalize(x, p=2, dim=-1)
        return normalized.squeeze(0)
    return F.normalize(x, p=2, dim=-1)


def _load_entity_embeddings(cfg: Dict[str, Any]) -> torch.Tensor:
    """
    Lazy-load entity embeddings from disk and cache at module level.
    
    Args:
        cfg: Configuration dictionary with entity_graph section
    
    Returns:
        Entity embeddings tensor of shape [N_entities, 512]
    """
    global _entity_embeddings_cache
    
    if _entity_embeddings_cache is None:
        entity_cfg = cfg.get("entity_graph", {})
        embeddings_path = entity_cfg.get("entity_embeddings_path", "data/entities/entity_embeddings.pt")
        embeddings_path = Path(embeddings_path)
        
        if not embeddings_path.exists():
            raise FileNotFoundError(f"Entity embeddings not found: {embeddings_path}")
        
        logger.info(f"Loading entity embeddings from {embeddings_path}")
        _entity_embeddings_cache = torch.load(embeddings_path, map_location='cpu')
        logger.info(f"Loaded {_entity_embeddings_cache.shape[0]} entity embeddings")
    
    return _entity_embeddings_cache


def _load_entity_meta(cfg: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    """
    Lazy-load entity metadata from disk and cache at module level.
    
    Args:
        cfg: Configuration dictionary with entity_graph section
    
    Returns:
        Dictionary mapping entity_id (int) to metadata dict
    """
    global _entity_meta_cache
    
    if _entity_meta_cache is None:
        entity_cfg = cfg.get("entity_graph", {})
        meta_path = entity_cfg.get("entity_meta_path", "data/entities/entity_meta.json")
        meta_path = Path(meta_path)
        
        if not meta_path.exists():
            raise FileNotFoundError(f"Entity metadata not found: {meta_path}")
        
        logger.info(f"Loading entity metadata from {meta_path}")
        with open(meta_path, 'r', encoding='utf-8') as f:
            raw_meta = json.load(f)
        
        # Convert string keys to int
        _entity_meta_cache = {int(k): v for k, v in raw_meta.items()}
        logger.info(f"Loaded metadata for {len(_entity_meta_cache)} entities")
    
    return _entity_meta_cache


def _load_entity_context(cfg: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    """
    Lazy-load entity context from disk and cache at module level.
    
    Args:
        cfg: Configuration dictionary with entity_graph section
    
    Returns:
        Dictionary mapping entity_id (int) to context dict with image_ids and caption_ids
    """
    global _entity_context_cache
    
    if _entity_context_cache is None:
        entity_cfg = cfg.get("entity_graph", {})
        context_path = entity_cfg.get("context_path", "data/entities/entity_context.json")
        context_path = Path(context_path)
        
        if not context_path.exists():
            raise FileNotFoundError(f"Entity context not found: {context_path}")
        
        logger.info(f"Loading entity context from {context_path}")
        with open(context_path, 'r', encoding='utf-8') as f:
            raw_context = json.load(f)
        
        # Convert string keys to int
        _entity_context_cache = {int(k): v for k, v in raw_context.items()}
        logger.info(f"Loaded context for {len(_entity_context_cache)} entities")
    
    return _entity_context_cache


def _build_reverse_index(entity_context: Dict[int, Dict[str, Any]]) -> Tuple[Dict[str, Set[int]], Dict[str, Set[int]]]:
    """
    Build reverse indices mapping image_id/caption_id to entity_ids.
    
    Cached at module level for reuse across queries.
    
    Args:
        entity_context: Mapping from entity_id to dict with image_ids and caption_ids
    
    Returns:
        Tuple of (image_to_entities, caption_to_entities) dicts
    """
    global _reverse_index_cache
    
    if _reverse_index_cache is None:
        logger.info("Building reverse index from entity_context")
        image_to_entities: Dict[str, Set[int]] = collections.defaultdict(set)
        caption_to_entities: Dict[str, Set[int]] = collections.defaultdict(set)
        
        for entity_id, context in entity_context.items():
            # Handle both string and int keys
            eid = int(entity_id) if not isinstance(entity_id, int) else entity_id
            
            for img_id in context.get("image_ids", []):
                image_to_entities[str(img_id)].add(eid)
            
            for cap_id in context.get("caption_ids", []):
                caption_to_entities[str(cap_id)].add(eid)
        
        _reverse_index_cache = (dict(image_to_entities), dict(caption_to_entities))
        logger.info(f"Built reverse index: {len(image_to_entities)} images, {len(caption_to_entities)} captions")
    
    return _reverse_index_cache


def _collect_candidate_entities(
    seeds: List[Tuple[str, float]],
    image_to_entities: Dict[str, Set[int]],
    caption_to_entities: Dict[str, Set[int]]
) -> Dict[int, int]:
    """
    Collect candidate entities from CLIP search seeds and count frequencies.
    
    Note: Currently only image_to_entities is used because seeds are image-based.
    caption_to_entities is kept for future caption-aware seeds support.
    
    Args:
        seeds: List of (image_id, score) tuples from CLIP search
        image_to_entities: Reverse index mapping image_id to entity_ids
        caption_to_entities: Reverse index mapping caption_id to entity_ids (kept for future use)
    
    Returns:
        Dictionary mapping entity_id to frequency count in seeds
    """
    freq = collections.Counter()
    
    for image_id, _score in seeds:
        image_id_str = str(image_id)
        for eid in image_to_entities.get(image_id_str, []):
            freq[eid] += 1
    
    return dict(freq)


def _score_entities(
    q0: torch.Tensor,
    entity_ids: List[int],
    entity_embeddings: torch.Tensor,
    freq: Dict[int, int],
    w_freq: float,
    w_sim: float
) -> Dict[int, float]:
    """
    Score candidate entities by combining frequency and similarity.
    
    Args:
        q0: Query embedding [512] or [1, 512], L2-normalized
        entity_ids: List of candidate entity IDs
        entity_embeddings: All entity embeddings [N_entities, 512]
        freq: Frequency count for each entity_id
        w_freq: Weight for frequency score
        w_sim: Weight for similarity score
    
    Returns:
        Dictionary mapping entity_id to combined score
    """
    if not entity_ids:
        return {}
    
    # Ensure q0 is 1D
    if q0.ndim == 2:
        q0 = q0.squeeze(0)
    
    # Get embeddings for candidate entities
    idx = torch.tensor(entity_ids, dtype=torch.long, device=entity_embeddings.device)
    emb = entity_embeddings[idx]  # [num_candidates, 512]
    
    # Compute cosine similarities (dot product since both are L2-normalized)
    sims = (emb @ q0.to(emb.device)).cpu()  # [num_candidates]
    
    # Normalize frequency scores
    freq_vals = torch.tensor([freq[eid] for eid in entity_ids], dtype=torch.float32)
    freq_max = freq_vals.max().clamp_min(1.0)
    freq_norm = freq_vals / freq_max
    
    # Normalize similarity scores to [0, 1]
    sims_min = sims.min()
    sims_max = sims.max()
    sims_range = sims_max - sims_min
    if sims_range > 1e-8:
        sims_norm = (sims - sims_min) / sims_range
    else:
        sims_norm = torch.zeros_like(sims)
    
    # Combine scores
    scores = w_freq * freq_norm + w_sim * sims_norm
    
    return {eid: float(score) for eid, score in zip(entity_ids, scores)}


def _top_k_entities(scored_entities: Dict[int, float], k: int) -> Tuple[List[int], List[float]]:
    """
    Select top-k entities by score with deterministic tie-breaking.
    
    Args:
        scored_entities: Dictionary mapping entity_id to score
        k: Number of entities to select
    
    Returns:
        Tuple of (entity_ids, scores) for top-k entities
    """
    if not scored_entities:
        return [], []
    
    # Sort by score (descending), then by entity_id (ascending) for determinism
    sorted_items = sorted(scored_entities.items(), key=lambda kv: (-kv[1], kv[0]))
    top = sorted_items[:k]
    
    if not top:
        return [], []
    
    entity_ids, entity_scores = zip(*top)
    return list(entity_ids), list(entity_scores)


def _log_enrichment(result: EnrichmentResult, cfg: Dict[str, Any]) -> None:
    """
    Log enrichment results for debugging and inspection.
    
    Args:
        result: EnrichmentResult to log
        cfg: Configuration dictionary with query_enrichment section
    """
    enrichment_cfg = cfg.get("query_enrichment", {})
    log_examples = enrichment_cfg.get("log_examples", False)
    
    if not log_examples:
        return
    
    logger.info("=" * 70)
    logger.info("Query Enrichment")
    logger.info("=" * 70)
    logger.info(f"Original: {result.original_query}")
    logger.info(f"Enriched: {result.enriched_query}")
    logger.info(f"Entities ({len(result.entity_names)}): {', '.join(result.entity_names)}")
    
    # Log top entity scores (truncate to 10)
    if len(result.entity_ids) > 0:
        logger.info("Top entity scores:")
        for i, (eid, name, score) in enumerate(zip(
            result.entity_ids[:10],
            result.entity_names[:10],
            result.entity_scores[:10].tolist()
        )):
            logger.info(f"  {i+1}. {name} (id={eid}, score={score:.4f})")
    logger.info("=" * 70)


def enrich_query(
    query: str,
    seeds: List[Tuple[str, float]],
    encoders: Any,
    entity_context: Dict[int, Dict[str, Any]],
    cfg: Dict[str, Any],
) -> EnrichmentResult:
    """
    Enrich a query by selecting top entities from pre-computed CLIP search seeds.

    This is the canonical query enrichment function for Phase 4. It expects CLIP
    search seeds (image results) to be provided by the caller (e.g., from Stage 1
    retrieval in hybrid_search.py).

    Algorithm:
      1. Encode original query with CLIP text encoder → q0 (embedding).
      2. Use provided CLIP search seeds (top K_seed_raw image results).
      3. Collect candidate entities from these seeds using entity_context.
      4. Score entities by:
         - Frequency in the K_seed_raw results.
         - Cosine similarity of entity embeddings to q0.
      5. Select top M_enrich entities.
      6. Build enriched query text using templates from cfg["query_enrichment"]:
         - For text queries: "{query}. Related: {entities}"
         - For image queries (future): "photo of {entities}"
      7. Encode enriched text with CLIP → q_enriched.
      8. Return EnrichmentResult with all details.

    Args:
        query: Original user query string.
        seeds: List of (image_id, score) tuples from CLIP search (pre-computed by caller).
        encoders: BiEncoder instance (CLIP model) for encoding query and enriched text.
        entity_context: Mapping from entity_id to dict with:
                        {"entity": str, "image_ids": List[str], "caption_ids": List[str]}
        cfg: Configuration dictionary. Use cfg["query_enrichment"] for parameters:
             - M_enrich: number of entities to select
             - w_freq: weight for frequency score
             - w_sim: weight for similarity score
             - text_template, image_template: enrichment templates
             - log_examples: whether to log results

    Returns:
        EnrichmentResult with original query, enriched query, embeddings, and selected entities.
    """
    # Load configuration
    enrichment_cfg = cfg.get("query_enrichment", {})
    M_enrich = enrichment_cfg.get("M_enrich", 8)
    w_freq = enrichment_cfg.get("w_freq", 0.5)
    w_sim = enrichment_cfg.get("w_sim", 0.5)
    text_template = enrichment_cfg.get("text_template", "{query}. Related: {entities}")
    
    # Step 1: Encode original query
    logger.debug(f"Encoding original query: {query}")
    q0_np = encoders.encode_texts([query], normalize=True, show_progress=False)
    q0 = torch.from_numpy(q0_np).squeeze(0)  # [512]
    
    # Step 2: Build reverse index (cached)
    image_to_entities, caption_to_entities = _build_reverse_index(entity_context)
    
    # Step 3: Collect candidate entities from seeds
    logger.debug(f"Collecting candidate entities from {len(seeds)} seeds")
    freq = _collect_candidate_entities(seeds, image_to_entities, caption_to_entities)
    
    if not freq:
        # No entities found in seeds, return original query
        logger.debug("No candidate entities found in seeds")
        enriched_query = query
        q_enriched = q0.clone()
        
        result = EnrichmentResult(
            original_query=query,
            enriched_query=enriched_query,
            q0=q0,
            q_enriched=q_enriched,
            entity_ids=[],
            entity_names=[],
            entity_scores=torch.tensor([], dtype=torch.float32)
        )
        _log_enrichment(result, cfg)
        return result
    
    # Step 4: Score entities by frequency + similarity
    entity_embeddings = _load_entity_embeddings(cfg)
    entity_meta = _load_entity_meta(cfg)
    
    candidate_ids = list(freq.keys())
    logger.debug(f"Scoring {len(candidate_ids)} candidate entities")
    scored_entities = _score_entities(q0, candidate_ids, entity_embeddings, freq, w_freq, w_sim)
    
    # Step 5: Select top M_enrich entities
    top_entity_ids, top_scores = _top_k_entities(scored_entities, M_enrich)
    
    if not top_entity_ids:
        # No entities selected, return original query
        logger.debug("No entities selected after scoring")
        enriched_query = query
        q_enriched = q0.clone()
        
        result = EnrichmentResult(
            original_query=query,
            enriched_query=enriched_query,
            q0=q0,
            q_enriched=q_enriched,
            entity_ids=[],
            entity_names=[],
            entity_scores=torch.tensor([], dtype=torch.float32)
        )
        _log_enrichment(result, cfg)
        return result
    
    # Get entity names from metadata
    entity_names = [entity_meta[eid].get("entity", f"entity_{eid}") for eid in top_entity_ids]
    
    # Step 6: Build enriched query text
    entities_str = ", ".join(entity_names)
    enriched_query = text_template.format(query=query.strip(), entities=entities_str)
    logger.debug(f"Enriched query: {enriched_query}")
    
    # Step 7: Encode enriched query
    q_enriched_np = encoders.encode_texts([enriched_query], normalize=True, show_progress=False)
    q_enriched = torch.from_numpy(q_enriched_np).squeeze(0)  # [512]
    
    # Step 8: Return EnrichmentResult
    entity_scores = torch.tensor(top_scores, dtype=torch.float32)
    
    result = EnrichmentResult(
        original_query=query,
        enriched_query=enriched_query,
        q0=q0,
        q_enriched=q_enriched,
        entity_ids=top_entity_ids,
        entity_names=entity_names,
        entity_scores=entity_scores
    )
    
    _log_enrichment(result, cfg)
    return result


# ============================================================================
# Graph search helpers
# ============================================================================

def _get_entity_embeddings_from_graph(graph: Any) -> torch.Tensor:
    """
    Extract entity embeddings from the graph.
    
    Args:
        graph: HeteroData with graph["entity"].x containing embeddings
    
    Returns:
        Entity embeddings tensor of shape [N_entities, 512]
    """
    entity_emb = graph["entity"].x
    assert entity_emb.dtype == torch.float32, f"Expected float32, got {entity_emb.dtype}"
    return entity_emb


def _select_seed_entities(
    q_enriched: torch.Tensor,
    entity_embeddings: torch.Tensor,
    K_seed: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Select top K_seed entities by similarity to enriched query embedding.
    
    Args:
        q_enriched: Query embedding [512] or [1, 512], L2-normalized
        entity_embeddings: All entity embeddings [N_entities, 512]
        K_seed: Number of seed entities to select
    
    Returns:
        Tuple of (seed_ids, seed_scores) where:
            seed_ids: 1D tensor of int64 entity IDs
            seed_scores: 1D tensor of float32 similarity scores
    """
    # Ensure q_enriched is 1D
    if q_enriched.ndim == 2:
        q_enriched = q_enriched.squeeze(0)
    elif q_enriched.ndim != 1:
        raise ValueError(f"Expected 1D or 2D tensor, got shape {q_enriched.shape}")
    
    # Compute similarities (dot product since both are L2-normalized)
    q_enriched = q_enriched.view(1, -1)  # [1, 512]
    sims = torch.matmul(entity_embeddings, q_enriched.t()).squeeze(-1)  # [N_entities]
    
    # Select top K
    K = min(K_seed, sims.shape[0])
    top_scores, top_idx = torch.topk(sims, K, dim=0)
    
    return top_idx, top_scores


def _build_adjacency(graph: Any) -> Dict[str, Dict[int, List[Tuple[int, float]]]]:
    """
    Build adjacency lists from graph edge_index and edge_weight.
    
    Args:
        graph: HeteroData with semantic and co-occurrence edges
    
    Returns:
        Dictionary mapping edge type to adjacency dict:
            {"sem": {u: [(v, weight), ...]}, "cooc": {u: [(v, weight), ...]}}
    """
    adj: Dict[str, Dict[int, List[Tuple[int, float]]]] = {
        "sem": defaultdict(list),
        "cooc": defaultdict(list),
    }
    
    # Add semantic edges
    if ("entity", "sem", "entity") in graph.edge_types:
        edge_index = graph["entity", "sem", "entity"].edge_index
        edge_weight = graph["entity", "sem", "entity"].edge_weight
        src, dst = edge_index
        for u, v, w in zip(src.tolist(), dst.tolist(), edge_weight.tolist()):
            adj["sem"][u].append((v, float(w)))
    
    # Add co-occurrence edges
    if ("entity", "cooc", "entity") in graph.edge_types:
        edge_index = graph["entity", "cooc", "entity"].edge_index
        edge_weight = graph["entity", "cooc", "entity"].edge_weight
        src, dst = edge_index
        for u, v, w in zip(src.tolist(), dst.tolist(), edge_weight.tolist()):
            adj["cooc"][u].append((v, float(w)))
    
    return adj


def _expand_frontier(
    seeds: Tuple[torch.Tensor, torch.Tensor],
    adj: Dict[str, Dict[int, List[Tuple[int, float]]]],
    cfg: Dict[str, Any],
) -> Dict[int, float]:
    """
    Expand frontier using priority queue beam search with LightRAG-style scoring.
    
    Args:
        seeds: Tuple of (seed_ids, seed_scores) tensors
        adj: Adjacency dict from _build_adjacency
        cfg: graph_search config dict with decay, H_max, B, N_max, T_cap_ms, type_weights
    
    Returns:
        Dictionary mapping entity_id to accumulated score
    """
    seed_ids = seeds[0].tolist()
    seed_scores = seeds[1].tolist()
    
    decay = cfg["decay"]
    H_max = cfg["H_max"]
    B = cfg["B"]
    N_max = cfg["N_max"]
    T_cap_ms = cfg["T_cap_ms"]
    type_weight_sem = cfg["type_weight_sem"]
    type_weight_cooc = cfg["type_weight_cooc"]
    
    entity_scores: Dict[int, float] = {}
    visited: Set[int] = set()
    # Heap entries: (-score, entity_id, hop)
    # Use negative score for max-heap behavior
    frontier: List[Tuple[float, int, int]] = []
    
    t_start = time.perf_counter()
    
    # Initialize with seeds (hop = 0)
    for eid, s in zip(seed_ids, seed_scores):
        score = float(s)
        entity_scores[eid] = entity_scores.get(eid, 0.0) + score
        heapq.heappush(frontier, (-score, eid, 0))
    
    num_processed = 0
    
    while frontier and num_processed < B:
        # Check time budget
        now_ms = (time.perf_counter() - t_start) * 1000.0
        if now_ms >= T_cap_ms:
            logger.debug(f"Graph expansion stopped: time budget {T_cap_ms}ms exceeded ({now_ms:.1f}ms)")
            break
        
        neg_score_u, u, h = heapq.heappop(frontier)
        score_u = -neg_score_u
        
        if u in visited:
            continue
        visited.add(u)
        num_processed += 1
        
        if h >= H_max:
            continue
        
        # Expand along semantic edges
        for v, edge_w in adj["sem"].get(u, []):
            score_delta = score_u * decay * edge_w * type_weight_sem
            if score_delta > 0.0:
                new_score = entity_scores.get(v, 0.0) + score_delta
                entity_scores[v] = new_score
                heapq.heappush(frontier, (-new_score, v, h + 1))
        
        # Expand along co-occurrence edges
        for v, edge_w in adj["cooc"].get(u, []):
            score_delta = score_u * decay * edge_w * type_weight_cooc
            if score_delta > 0.0:
                new_score = entity_scores.get(v, 0.0) + score_delta
                entity_scores[v] = new_score
                heapq.heappush(frontier, (-new_score, v, h + 1))
    
    logger.debug(f"Graph expansion: processed {num_processed} nodes, collected {len(entity_scores)} entities")
    
    # Truncate to top N_max entities if needed
    if len(entity_scores) > N_max:
        items = sorted(entity_scores.items(), key=lambda kv: (-kv[1], kv[0]))[:N_max]
        entity_scores = dict(items)
        logger.debug(f"Truncated to top {N_max} entities")
    
    return entity_scores


def _aggregate_entity_scores_to_images(
    entity_scores: Dict[int, float],
    entity_context: Dict[int, Dict[str, Any]],
    entity_meta: Dict[int, Dict[str, Any]],
) -> Dict[str, float]:
    """
    Aggregate entity scores to image scores using entity context.
    
    Args:
        entity_scores: Dictionary mapping entity_id to score
        entity_context: Mapping from entity_id to {image_ids, caption_ids}
        entity_meta: Mapping from entity_id to {name, df_caption, df_image, cf}
    
    Returns:
        Dictionary mapping image_id to aggregated KG score
    """
    image_scores: Dict[str, float] = {}
    
    for eid, score_e in entity_scores.items():
        ctx = entity_context.get(eid)
        if not ctx:
            continue
        
        # Get df_image for downweighting common entities
        meta = entity_meta.get(eid, {})
        df_image = max(int(meta.get("df_image", 1)), 1)
        
        # Weight contribution by 1/sqrt(df_image)
        weight = score_e / math.sqrt(df_image)
        
        # Add to all images containing this entity
        for img_id in ctx.get("image_ids", []):
            img_id_str = str(img_id)
            image_scores[img_id_str] = image_scores.get(img_id_str, 0.0) + weight
    
    return image_scores


# ============================================================================
# Graph search main API
# ============================================================================

def graph_search(
    query: str,
    graph: Any,  # HeteroData
    encoders: Any,
    cfg: Dict[str, Any],
) -> GraphSearchResult:
    """
    Perform bounded beam search over the entity graph with query enrichment.

    Algorithm:
      1. Call enrich_query(...) to get EnrichmentResult with q_enriched.
      2. Compute similarity between q_enriched and all entity embeddings.
      3. Select top K_seed entities as seeds.
      4. Initialize a max-heap frontier with seed entities and their scores.
      5. Expand frontier using beam search:
         - Pop highest-scoring entity u from frontier.
         - For each neighbor v of u (via semantic or co-occurrence edges):
           - Update score_v += score_u * (decay ** hop) * edge_weight * type_weight
         - Enforce bounds: H_max hops, B processed nodes, N_max entities, T_cap_ms time.
      6. Aggregate entity scores to image scores using entity_context.
      7. Return GraphSearchResult with all details.

    Args:
        query: Original user query string.
        graph: HeteroData entity graph with:
               - graph["entity"].x: entity embeddings [N, d]
               - graph["entity", "sem", "entity"]: semantic edges
               - graph["entity", "cooc", "entity"]: co-occurrence edges
        encoders: Encoder module (CLIP model) for encoding queries.
        cfg: Configuration dictionary. Must contain:
             - query_enrichment: config for enrich_query
             - graph_search: K_seed, H_max, B, N_max, T_cap_ms, decay, type_weights
             - entity_graph: paths to context and meta files

    Returns:
        GraphSearchResult with query, enriched query, seed/entity/image scores, runtime.
    """
    from .config import get_graph_search_config, get_query_enrichment_config
    
    t_start = time.perf_counter()
    
    # Load configs
    graph_search_cfg = get_graph_search_config(cfg)
    enrichment_cfg = get_query_enrichment_config(cfg)
    
    # Load entity context and meta
    entity_context = _load_entity_context(cfg)
    entity_meta = _load_entity_meta(cfg)
    
    # Step 1: Query enrichment
    # Use enrich_query to obtain enriched query embedding when enabled
    if enrichment_cfg.get("enabled", True):
        # Call enrich_query with empty seeds list for now
        # Stage 1 CLIP seeds will be provided by the retrieval layer in future integration
        logger.debug(f"Enriching query for graph search: {query}")
        seeds: List[Tuple[str, float]] = []  # Empty seeds for standalone usage
        
        enrichment_result = enrich_query(
            query=query,
            seeds=seeds,
            encoders=encoders,
            entity_context=entity_context,
            cfg=cfg,
        )
        
        # Use enriched query embedding and text
        q_enriched = enrichment_result.q_enriched
        enriched_query = enrichment_result.enriched_query
        logger.debug(f"Query enriched: {enriched_query}")
    else:
        # Enrichment disabled - use raw query
        logger.info("Query enrichment disabled, using raw query embedding")
        q_np = encoders.encode_texts([query], normalize=True, show_progress=False)
        q_enriched = torch.from_numpy(q_np).squeeze(0)
        enriched_query = query
    
    # Step 2: Get entity embeddings from graph
    entity_embeddings = _get_entity_embeddings_from_graph(graph)
    
    # Step 3: Select seed entities
    K_seed = graph_search_cfg["K_seed"]
    seed_ids, seed_scores = _select_seed_entities(q_enriched, entity_embeddings, K_seed)
    
    logger.debug(f"Selected {len(seed_ids)} seed entities")
    
    # Step 4: Build adjacency structure
    adj = _build_adjacency(graph)
    
    # Step 5: Expand frontier
    entity_scores_dict = _expand_frontier(
        (seed_ids, seed_scores),
        adj,
        graph_search_cfg
    )
    
    # Step 6: Aggregate to image scores
    image_scores_dict = _aggregate_entity_scores_to_images(
        entity_scores_dict,
        entity_context,
        entity_meta
    )
    
    # Sort image scores descending
    image_scores_list = sorted(
        image_scores_dict.items(),
        key=lambda kv: (-kv[1], kv[0])  # Sort by score desc, then by id for determinism
    )
    
    # Sort entity scores descending and convert to lists/tensors
    entity_items = sorted(
        entity_scores_dict.items(),
        key=lambda kv: (-kv[1], kv[0])
    )
    entity_ids_list = [eid for eid, _ in entity_items]
    entity_scores_list = [score for _, score in entity_items]
    entity_scores_tensor = torch.tensor(entity_scores_list, dtype=torch.float32)
    
    # Calculate runtime
    runtime_ms = (time.perf_counter() - t_start) * 1000.0
    
    logger.info(f"Graph search completed in {runtime_ms:.1f}ms: "
                f"{len(seed_ids)} seeds -> {len(entity_ids_list)} entities -> "
                f"{len(image_scores_list)} images")
    
    return GraphSearchResult(
        original_query=query,
        enriched_query=enriched_query,
        q_enriched=q_enriched,
        seed_entity_ids=seed_ids.tolist(),
        seed_entity_scores=seed_scores,
        entity_ids=entity_ids_list,
        entity_scores=entity_scores_tensor,
        image_scores=image_scores_list,
        runtime_ms=runtime_ms,
    )


# ============================================================================
# Debug and logging helpers
# ============================================================================

def _log_graph_search_debug(
    result: GraphSearchResult,
    entity_meta: Dict[int, Dict[str, Any]],
    top_k: int = 10
) -> None:
    """
    Log graph search results for debugging and inspection.
    
    Args:
        result: GraphSearchResult to log
        entity_meta: Entity metadata for looking up names
        top_k: Number of top items to log
    """
    logger.info("=" * 70)
    logger.info("Graph Search Results")
    logger.info("=" * 70)
    logger.info(f"Query: {result.original_query}")
    logger.info(f"Enriched: {result.enriched_query}")
    logger.info(f"Runtime: {result.runtime_ms:.1f}ms")
    
    # Log seed entities
    logger.info(f"\nSeed entities ({len(result.seed_entity_ids)}):")
    for i, (eid, score) in enumerate(zip(
        result.seed_entity_ids[:top_k],
        result.seed_entity_scores[:top_k].tolist()
    )):
        name = entity_meta.get(eid, {}).get("entity", f"entity_{eid}")
        logger.info(f"  {i+1}. {name} (id={eid}, score={score:.4f})")
    
    # Log expanded entities
    logger.info(f"\nExpanded entities ({len(result.entity_ids)}):")
    for i, (eid, score) in enumerate(zip(
        result.entity_ids[:top_k],
        result.entity_scores[:top_k].tolist()
    )):
        name = entity_meta.get(eid, {}).get("entity", f"entity_{eid}")
        logger.info(f"  {i+1}. {name} (id={eid}, score={score:.4f})")
    
    # Log top images
    logger.info(f"\nTop images by KG score ({len(result.image_scores)}):")
    for i, (img_id, score) in enumerate(result.image_scores[:top_k]):
        logger.info(f"  {i+1}. {img_id}: {score:.4f}")
    
    logger.info("=" * 70)
