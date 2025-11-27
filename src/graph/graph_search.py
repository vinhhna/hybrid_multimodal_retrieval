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
import json
import logging
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
        query: Original query string.
        image_scores: Mapping from image_id (str) to aggregated score (float).
        entity_scores: Mapping from entity_id (int) to score (float).
        debug: Dictionary with debugging/logging information (e.g., hop counts,
               timing, number of nodes expanded).
    """
    query: str
    image_scores: Dict[str, float]  # image_id -> score
    entity_scores: Dict[int, float]  # entity_id -> score
    debug: Dict[str, Any]


# ============================================================================
# Query enrichment
# ============================================================================

# Module-level caches for lazy loading
_entity_embeddings_cache: Optional[torch.Tensor] = None
_entity_meta_cache: Optional[Dict[int, Dict[str, Any]]] = None
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
# Graph search
# ============================================================================

def graph_search(
    query: str,
    graph: Any,  # HeteroData in full implementation
    encoders: Any,
    entity_context: Dict[int, Dict[str, Any]],
    cfg: Dict[str, Any],
) -> GraphSearchResult:
    """
    Perform bounded beam search over the entity graph.

    This is a skeleton for later implementation (Week 2 of Phase 4 plan).

    Algorithm outline (for future implementation):
      1. Call enrich_query(...) to get EnrichmentResult.
      2. Encode enriched query with CLIP text encoder → q_enriched (embedding).
      3. Compute similarity between q_enriched and all entity embeddings (graph["entity"].x).
      4. Select top K_seed entities as seeds (highest similarity).
      5. Initialize a max-heap frontier with seed entities and their scores.
      6. Expand frontier using beam search:
         - Pop highest-scoring entity u from frontier.
         - For each neighbor v of u (via semantic or co-occurrence edges):
           - Update score_v += score_u * (decay ** hop) * edge_weight * type_weight
           - Add v to frontier if not yet processed.
         - Enforce bounds: H_max hops, B processed nodes, N_max total entities, T_cap_ms time limit.
      7. Aggregate entity scores to image scores using entity_context:
         - For each image_id, sum scores of entities appearing in that image.
         - Optionally scale by 1/sqrt(df) to downweight very common entities.
      8. Return GraphSearchResult with query, image_scores, entity_scores, and debug info.

    Args:
        query: Original user query string.
        graph: HeteroData entity graph with:
               - graph["entity"].x: entity embeddings [N, d]
               - graph["entity", "sem", "entity"]: semantic edges
               - graph["entity", "cooc", "entity"]: co-occurrence edges
        encoders: Encoder module (CLIP model) for encoding enriched query.
        entity_context: Mapping from entity_id to dict with:
                        {"entity": str, "image_ids": List[str], "caption_ids": List[str]}
        cfg: Configuration dictionary. Use cfg["graph_search"] for parameters:
             - K_seed: number of seed entities
             - H_max: max hops
             - B: max processed nodes (beam width)
             - N_max: max total entities
             - T_cap_ms: time budget in milliseconds
             - decay: score decay factor per hop
             - type_weight_sem, type_weight_cooc: edge-type weights

    Returns:
        GraphSearchResult with query, image_scores, entity_scores, and debug info.

    Raises:
        NotImplementedError: This is a skeleton; full implementation in later days.

    TODO(phase4-entity): Implement in Week 2 (Day 10-12)
      - Call enrich_query(...)
      - Encode enriched query
      - Compute entity similarities and select top K_seed seeds
      - Initialize frontier (max-heap)
      - Beam search loop with scoring rule:
          score_v += score_u * (decay ** hop) * edge_weight * type_weight
      - Enforce bounds (H_max, B, N_max, T_cap_ms)
      - Aggregate entity scores to image scores
      - Return GraphSearchResult
    """
    raise NotImplementedError(
        "Phase 4 - graph_search: to be implemented in later days (Week 2, Day 10-12)."
    )


# ============================================================================
# Helper functions (for future use)
# ============================================================================

def seed_entities(
    query_embedding: Any,  # torch.Tensor in full implementation
    entity_embeddings: Any,  # torch.Tensor
    K_seed: int,
) -> List[int]:
    """
    Select top K_seed entities by similarity to query embedding.

    This is a placeholder for future implementation.

    Args:
        query_embedding: Query embedding from CLIP text encoder, shape [d].
        entity_embeddings: All entity embeddings, shape [N, d].
        K_seed: Number of seed entities to select.

    Returns:
        List of entity IDs (indices) sorted by descending similarity.

    TODO(phase4-entity): Implement in Week 2
      - Compute cosine similarity between query_embedding and entity_embeddings
      - Return top K_seed entity IDs
    """
    raise NotImplementedError("Phase 4 - seed_entities: to be implemented in Week 2.")


def expand_frontier(
    frontier: Any,  # priority queue in full implementation
    graph: Any,  # HeteroData
    entity_scores: Dict[int, float],
    cfg: Dict[str, Any],
) -> None:
    """
    Expand the beam search frontier by one step.

    This is a placeholder for future implementation.

    Args:
        frontier: Max-heap priority queue of (score, entity_id) pairs.
        graph: HeteroData entity graph with semantic and co-occurrence edges.
        entity_scores: Dictionary tracking cumulative scores for each entity.
        cfg: Configuration with decay, type_weight_sem, type_weight_cooc.

    TODO(phase4-entity): Implement in Week 2
      - Pop highest-scoring entity from frontier
      - For each neighbor via semantic/co-occurrence edges:
        - Compute new score using decay and edge weights
        - Update entity_scores
        - Push neighbor to frontier if not yet processed
    """
    raise NotImplementedError("Phase 4 - expand_frontier: to be implemented in Week 2.")


# TODO(phase4-entity): Future implementation sketch (for reference)
#
# Scoring rule for graph expansion:
#   For each edge u → v at hop h:
#     score_v += score_u * (decay ** h) * edge_weight * type_weight
#   where:
#     - decay ≈ 0.85 (from cfg["graph_search"]["decay"])
#     - edge_weight is the weight on the edge (cosine sim or co-occurrence count)
#     - type_weight is cfg["graph_search"]["type_weight_sem"] for semantic edges
#       or cfg["graph_search"]["type_weight_cooc"] for co-occurrence edges
#
# Entity → image aggregation sketch:
#   image_scores = defaultdict(float)
#   for entity_id, entity_score in entity_scores.items():
#       for image_id in entity_context[entity_id]["image_ids"]:
#           image_scores[image_id] += entity_score
#           # Optional: scale by 1/sqrt(df) to downweight common entities
#
# Fusion with CLIP and BLIP-2 scores (later days):
#   final_score = w_clip * clip_norm + w_kg * kg_norm + w_blip2 * blip2_norm
#   where each score is normalized per-query (min-max or softmax).
