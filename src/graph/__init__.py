"""
Graph package for Phase 4 (entity-centric design).

This package provides modules for:
  - Entity vocabulary and context building (entities.py)
  - Entity graph construction (build_entity_graph.py)
  - Query enrichment and graph search (graph_search.py)
  - Context synthesis and interpretability (context.py)
  - Configuration helpers (config.py)

Phase 4 implementation status:
  - entities.py: IMPLEMENTED (entity vocabulary building)
  - build_entity_graph.py: SKELETON (to be implemented in Week 1-2)
  - graph_search.py: SKELETON (to be implemented in Week 2)
  - context.py: SKELETON (to be implemented in Week 3)
  - config.py: IMPLEMENTED (config loading and helpers)

All old Phase 4 graph modules (schema.py, build.py, search.py, store.py, etc.)
have been removed in the phase4-entity branch.
"""

# Entity vocabulary building (entities.py)
from .entities import (
    EntityId,
    EntityStats,
    normalize_entity,
    extract_entities_from_caption,
    build_entity_vocabulary,
    save_entity_artifacts,
)

# Configuration helpers (config.py)
from .config import (
    load_entity_graph_config,
    get_entity_graph_config,
    get_query_enrichment_config,
    get_graph_search_config,
    get_fusion_config,
    print_config_summary,
)

# Graph construction (build_entity_graph.py) - skeleton only
from .build_entity_graph import (
    build_entity_graph,
    build_semantic_edges,
    build_cooccurrence_edges,
    save_entity_graph,
    load_entity_graph,
    l2_normalize,
)

# Query enrichment and graph search (graph_search.py) - skeleton only
from .graph_search import (
    EnrichmentResult,
    GraphSearchResult,
    enrich_query,
    graph_search,
)

# Context synthesis (context.py) - skeleton only
from .context import (
    synthesize_context,
    explain_image_ranking,
    summarize_graph_search,
)

__all__ = [
    # entities.py
    "EntityId",
    "EntityStats",
    "normalize_entity",
    "extract_entities_from_caption",
    "build_entity_vocabulary",
    "save_entity_artifacts",
    # config.py
    "load_entity_graph_config",
    "get_entity_graph_config",
    "get_query_enrichment_config",
    "get_graph_search_config",
    "get_fusion_config",
    "print_config_summary",
    # build_entity_graph.py
    "build_entity_graph",
    "build_semantic_edges",
    "build_cooccurrence_edges",
    "save_entity_graph",
    "load_entity_graph",
    "l2_normalize",
    # graph_search.py
    "EnrichmentResult",
    "GraphSearchResult",
    "enrich_query",
    "graph_search",
    # context.py
    "synthesize_context",
    "explain_image_ranking",
    "summarize_graph_search",
]
