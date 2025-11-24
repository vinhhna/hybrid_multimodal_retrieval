"""
Configuration helpers for Phase 4 entity-centric graph.

This module provides utilities for loading and accessing Phase 4 configuration
from YAML files. It centralizes config access and provides type-safe interfaces
for entity_graph, query_enrichment, graph_search, and fusion sections.

Phase 4 implementation plan: Day 0 - Prereqs & Setup
"""

from __future__ import annotations

import yaml
from pathlib import Path
from typing import Any, Dict


def load_entity_graph_config(path: str | Path) -> Dict[str, Any]:
    """
    Load the Phase 4 entity graph configuration from a YAML file.

    This helper provides a single entry point to read sections:
      - entity_graph: paths, thresholds (min_df, k_sem, degree_cap, etc.)
      - query_enrichment: K_seed_raw, M_enrich, templates, etc.
      - graph_search: K_seed, H_max, B, T_cap_ms, decay, edge weights
      - fusion: w_clip, w_kg, w_blip2

    Args:
        path: Path to the YAML configuration file (typically configs/entity_graph.yaml).

    Returns:
        Dictionary containing all config sections.

    Raises:
        FileNotFoundError: If the config file does not exist.
        yaml.YAMLError: If the YAML is malformed.

    Example usage (for later implementation):
        >>> cfg = load_entity_graph_config("configs/entity_graph.yaml")
        >>> entity_cfg = cfg.get("entity_graph", {})
        >>> graph_search_cfg = cfg.get("graph_search", {})
        >>> k_sem = entity_cfg.get("k_sem", 16)
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    
    if cfg is None:
        cfg = {}
    
    # TODO(phase4-entity): Add validation for required keys in later days
    # For now, just return the raw config and let consuming code handle defaults
    
    return cfg


def get_entity_graph_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract entity_graph configuration section with defaults.

    Args:
        cfg: Full configuration dictionary loaded from YAML.

    Returns:
        Dictionary with entity_graph config keys and defaults applied.

    Example:
        >>> cfg = load_entity_graph_config("configs/entity_graph.yaml")
        >>> entity_cfg = get_entity_graph_config(cfg)
        >>> min_df = entity_cfg["min_df"]
    """
    entity_cfg = cfg.get("entity_graph", {})
    
    # Apply defaults for critical parameters
    return {
        "min_df": entity_cfg.get("min_df", 5),
        "k_sem": entity_cfg.get("k_sem", 16),
        "degree_cap": entity_cfg.get("degree_cap", 64),
        "vocab_path": entity_cfg.get("vocab_path", "data/entities/entity_vocab.json"),
        "context_path": entity_cfg.get("context_path", "data/entities/entity_context.json"),
        "entity_embeddings_path": entity_cfg.get("entity_embeddings_path", "data/entities/entity_embeddings.pt"),
        "entity_graph_path": entity_cfg.get("entity_graph_path", "data/entities/entity_graph.pt"),
        "max_samples": entity_cfg.get("max_samples"),
        "verbose": entity_cfg.get("verbose", False),
    }


def get_query_enrichment_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract query_enrichment configuration section with defaults.

    Args:
        cfg: Full configuration dictionary loaded from YAML.

    Returns:
        Dictionary with query_enrichment config keys and defaults applied.
    """
    enrichment_cfg = cfg.get("query_enrichment", {})
    
    return {
        "enabled": enrichment_cfg.get("enabled", True),
        "K_seed_raw": enrichment_cfg.get("K_seed_raw", 100),
        "M_enrich": enrichment_cfg.get("M_enrich", 5),
        "text_template": enrichment_cfg.get("text_template", "{query}. Related: {entities}"),
        "image_template": enrichment_cfg.get("image_template", "photo of {entities}"),
    }


def get_graph_search_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract graph_search configuration section with defaults.

    Args:
        cfg: Full configuration dictionary loaded from YAML.

    Returns:
        Dictionary with graph_search config keys and defaults applied.
    """
    search_cfg = cfg.get("graph_search", {})
    
    return {
        "K_seed": search_cfg.get("K_seed", 10),
        "H_max": search_cfg.get("H_max", 2),
        "B": search_cfg.get("B", 20),
        "N_max": search_cfg.get("N_max", 200),
        "T_cap_ms": search_cfg.get("T_cap_ms", 150),
        "decay": search_cfg.get("decay", 0.85),
        "type_weight_sem": search_cfg.get("type_weight_sem", 1.0),
        "type_weight_cooc": search_cfg.get("type_weight_cooc", 0.7),
    }


def get_fusion_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract fusion configuration section with defaults.

    Args:
        cfg: Full configuration dictionary loaded from YAML.

    Returns:
        Dictionary with fusion config keys and defaults applied.
    """
    fusion_cfg = cfg.get("fusion", {})
    
    return {
        "w_clip": fusion_cfg.get("w_clip", 0.7),
        "w_kg": fusion_cfg.get("w_kg", 0.2),
        "w_blip2": fusion_cfg.get("w_blip2", 0.1),
    }


def print_config_summary(cfg: Dict[str, Any]) -> None:
    """
    Print a human-readable summary of the Phase 4 configuration.

    Useful for logging at the start of scripts to provide transparency
    about which hyperparameters are being used.

    Args:
        cfg: Full configuration dictionary loaded from YAML.
    """
    print("\n" + "=" * 70)
    print("PHASE 4 CONFIGURATION SUMMARY")
    print("=" * 70)
    
    entity_cfg = get_entity_graph_config(cfg)
    enrichment_cfg = get_query_enrichment_config(cfg)
    search_cfg = get_graph_search_config(cfg)
    fusion_cfg = get_fusion_config(cfg)
    
    print("\n[entity_graph]")
    for key, value in entity_cfg.items():
        print(f"  {key}: {value}")
    
    print("\n[query_enrichment]")
    for key, value in enrichment_cfg.items():
        print(f"  {key}: {value}")
    
    print("\n[graph_search]")
    for key, value in search_cfg.items():
        print(f"  {key}: {value}")
    
    print("\n[fusion]")
    for key, value in fusion_cfg.items():
        print(f"  {key}: {value}")
    
    print("\n" + "=" * 70)
