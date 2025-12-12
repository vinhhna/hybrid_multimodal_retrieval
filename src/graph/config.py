"""
Configuration helpers for Phase 4 entity-centric graph.

This module provides utilities for loading and accessing Phase 4 configuration
from YAML files. It centralizes config access and provides type-safe interfaces
for entity_graph, query_enrichment, graph_search, and fusion sections.

Phase 4 implementation plan: Day 0 - Prereqs & Setup
Phase 4 Day 13-14: Extended with per-mode fusion configuration
"""

from __future__ import annotations

import pprint
import yaml
from pathlib import Path
from typing import Any, Dict

# Type alias for cleaner signatures
ConfigDict = Dict[str, Any]


def load_entity_graph_config(path: str | Path) -> ConfigDict:
    """
    Load the Phase 4 entity graph configuration from a YAML file.

    This helper provides a single entry point to read sections:
      - entity_graph: paths, thresholds (min_df, k_sem, degree_cap, etc.)
      - query_enrichment: K_seed_raw, M_enrich, templates, etc.
      - graph_search: K_seed, H_max, B, T_cap_ms, decay, edge weights
      - fusion: Per-mode weights with two supported schemas:
        
        **New nested schema (recommended)**:
            fusion:
              default_mode: full
              modes:
                clip_only: {w_clip: 1.0, w_stage2: 0.0, w_kg: 0.0}
                hybrid: {w_clip: 0.7, w_stage2: 0.3, w_kg: 0.0}
                clip_kg: {w_clip: 0.7, w_stage2: 0.0, w_kg: 0.3}
                full: {w_clip: 0.6, w_stage2: 0.2, w_kg: 0.2}
        
        **Legacy flat schema (backward compatible)**:
            fusion:
              w_clip: 0.6
              w_kg: 0.2
              w_blip2: 0.2  # Auto-normalized to w_stage2

    Args:
        path: Path to the YAML configuration file (typically configs/entity_graph.yaml).
              Accepts string or Path object.

    Returns:
        Dictionary containing all config sections.

    Raises:
        FileNotFoundError: If the config file does not exist.
        yaml.YAMLError: If the YAML is malformed.

    Example:
        >>> cfg = load_entity_graph_config("configs/entity_graph.yaml")
        >>> entity_cfg = get_entity_graph_config(cfg)
        >>> fusion_cfg = get_fusion_config(cfg)
        >>> print(fusion_cfg["default_mode"])  # e.g., "full"
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    
    if cfg is None:
        cfg = {}
    
    return cfg


def get_entity_graph_config(cfg: ConfigDict) -> ConfigDict:
    """
    Extract entity_graph configuration section with defaults.

    Applies sensible defaults for all entity graph construction parameters:
      - min_df: 5 (minimum document frequency for entity filtering)
      - k_sem: 16 (number of semantic k-NN neighbors)
      - degree_cap: 64 (maximum node degree to avoid hubs)
      - build_entity_embeddings: False (whether to generate entity embeddings)
      - entity_text_template: "{}" (template for encoding entities)
      - batch_size: 64 (batch size for embedding generation)
      - verbose: False (enable detailed logging)

    Args:
        cfg: Full configuration dictionary loaded from YAML.

    Returns:
        Dictionary with entity_graph config keys and defaults applied.

    Example:
        >>> cfg = load_entity_graph_config("configs/entity_graph.yaml")
        >>> entity_cfg = get_entity_graph_config(cfg)
        >>> print(entity_cfg["min_df"])  # 5 (default) or value from YAML
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
        "entity_meta_path": entity_cfg.get("entity_meta_path", "data/entities/entity_meta.json"),
        "entity_graph_path": entity_cfg.get("entity_graph_path", "data/graph/entity_graph.pt"),
        "build_entity_embeddings": entity_cfg.get("build_entity_embeddings", False),
        "entity_text_template": entity_cfg.get("entity_text_template", "{}"),
        "batch_size": entity_cfg.get("batch_size", 64),
        "max_samples": entity_cfg.get("max_samples"),
        "verbose": entity_cfg.get("verbose", False),
    }


def get_query_enrichment_config(cfg: ConfigDict) -> ConfigDict:
    """
    Extract query_enrichment configuration section with defaults.

    Applies defaults for query enrichment parameters:
      - enabled: True (enable/disable query enrichment globally)
      - K_seed_raw: 32 (number of CLIP results to seed entity selection)
      - M_enrich: 8 (number of top entities to select for enrichment)
      - w_freq: 0.5 (weight for frequency-based entity scoring)
      - w_sim: 0.5 (weight for similarity-based entity scoring)
      - log_examples: False (log enrichment details for debugging)
      - text_template: "{query}. Related: {entities}" (enrichment format)
      - image_template: "photo of {entities}" (image query format)

    Args:
        cfg: Full configuration dictionary loaded from YAML.

    Returns:
        Dictionary with query_enrichment config keys and defaults applied.

    Example:
        >>> cfg = load_entity_graph_config("configs/entity_graph.yaml")
        >>> enrich_cfg = get_query_enrichment_config(cfg)
        >>> print(enrich_cfg["M_enrich"])  # 8 (default)
    """
    enrichment_cfg = cfg.get("query_enrichment", {})
    
    return {
        "enabled": enrichment_cfg.get("enabled", True),
        "K_seed_raw": enrichment_cfg.get("K_seed_raw", 32),
        "M_enrich": enrichment_cfg.get("M_enrich", 8),
        "w_freq": enrichment_cfg.get("w_freq", 0.5),
        "w_sim": enrichment_cfg.get("w_sim", 0.5),
        "log_examples": enrichment_cfg.get("log_examples", False),
        "text_template": enrichment_cfg.get("text_template", "{query}. Related: {entities}"),
        "image_template": enrichment_cfg.get("image_template", "photo of {entities}"),
    }


def get_graph_search_config(cfg: ConfigDict) -> ConfigDict:
    """
    Extract graph_search configuration section with defaults.

    Applies defaults for graph expansion parameters:
      - K_seed: 10 (number of seed entities for graph expansion)
      - H_max: 2 (maximum hops for graph traversal)
      - B: 20 (beam width for priority queue expansion)
      - N_max: 200 (maximum total entities to collect)
      - T_cap_ms: 150 (time budget in milliseconds)
      - decay: 0.85 (score decay factor per hop)
      - type_weight_sem: 1.0 (weight for semantic edges)
      - type_weight_cooc: 0.7 (weight for co-occurrence edges)

    Args:
        cfg: Full configuration dictionary loaded from YAML.

    Returns:
        Dictionary with graph_search config keys and defaults applied.

    Example:
        >>> cfg = load_entity_graph_config("configs/entity_graph.yaml")
        >>> search_cfg = get_graph_search_config(cfg)
        >>> print(search_cfg["H_max"])  # 2 (default for 2-hop expansion)
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


def get_fusion_config(cfg: ConfigDict) -> ConfigDict:
    """
    Extract fusion configuration section with per-mode weights (Phase 4 Day 13-14).
    
    Supports both new nested schema (modes) and legacy flat schema for backward compatibility.
    
    **New schema** (entity_graph.yaml with fusion.modes):
        fusion:
          default_mode: full
          modes:
            clip_only: {w_clip: 1.0, w_stage2: 0.0, w_kg: 0.0}
            hybrid: {w_clip: 0.7, w_stage2: 0.3, w_kg: 0.0}
            clip_kg: {w_clip: 0.7, w_stage2: 0.0, w_kg: 0.3}
            full: {w_clip: 0.6, w_stage2: 0.2, w_kg: 0.2}
    
    **Legacy schema** (flat weights, for backward compatibility):
        fusion:
          w_clip: 0.7
          w_kg: 0.2
          w_blip2: 0.1  # Automatically normalized to w_stage2
    
    **Key normalization**: Legacy `w_blip2` is automatically converted to `w_stage2`
    for consistency across the codebase. This function always returns weights using
    the w_stage2 key regardless of input schema.
    
    Args:
        cfg: Full configuration dictionary loaded from YAML.

    Returns:
        Dictionary with per-mode fusion weights and default_mode. Always returns
        the same normalized structure:
        {
            "clip_only": {"w_clip": 1.0, "w_stage2": 0.0, "w_kg": 0.0},
            "hybrid": {"w_clip": 0.7, "w_stage2": 0.3, "w_kg": 0.0},
            "clip_kg": {"w_clip": 0.7, "w_stage2": 0.0, "w_kg": 0.3},
            "full": {"w_clip": 0.6, "w_stage2": 0.2, "w_kg": 0.2},
            "default_mode": "full"  # or "hybrid" for legacy configs
        }
    
    Modes:
      - clip_only: CLIP Stage 1 only (fastest, no reranking, no KG)
      - hybrid: CLIP + BLIP-2 reranking (Phase 3 baseline, no KG)
      - clip_kg: CLIP + KG (no BLIP-2, for KG experiments)
      - full: CLIP + BLIP-2 + KG (full Phase 4 hybrid)
    
    Example:
        >>> cfg = load_entity_graph_config("configs/entity_graph.yaml")
        >>> fusion_cfg = get_fusion_config(cfg)
        >>> print(fusion_cfg["default_mode"])  # "full"
        >>> print(fusion_cfg["full"]["w_clip"])  # 0.6
        >>> print(fusion_cfg["full"]["w_stage2"])  # 0.2 (normalized from w_blip2 if needed)
    """
    fusion_cfg = cfg.get("fusion", {})
    
    # Check if new nested schema is present
    if "modes" in fusion_cfg and isinstance(fusion_cfg["modes"], dict):
        # New schema: extract modes
        modes = fusion_cfg["modes"]
        
        # Normalize key names: w_blip2 -> w_stage2 for consistency
        normalized_modes = {}
        for mode_name, weights in modes.items():
            normalized_weights = {}
            for key, value in weights.items():
                if key == "w_blip2":
                    normalized_weights["w_stage2"] = value
                else:
                    normalized_weights[key] = value
            normalized_modes[mode_name] = normalized_weights
        
        # Extract default_mode
        default_mode = fusion_cfg.get("default_mode", "full")
        
        # Build result with defaults for missing modes
        result = {
            "clip_only": normalized_modes.get("clip_only", {
                "w_clip": 1.0, "w_stage2": 0.0, "w_kg": 0.0
            }),
            "hybrid": normalized_modes.get("hybrid", {
                "w_clip": 0.7, "w_stage2": 0.3, "w_kg": 0.0
            }),
            "clip_kg": normalized_modes.get("clip_kg", {
                "w_clip": 0.7, "w_stage2": 0.0, "w_kg": 0.3
            }),
            "full": normalized_modes.get("full", {
                "w_clip": 0.6, "w_stage2": 0.2, "w_kg": 0.2
            }),
            "default_mode": default_mode
        }
        
        return result
    
    else:
        # Legacy flat schema: synthesize modes from flat weights
        w_clip = fusion_cfg.get("w_clip", 0.7)
        w_kg = fusion_cfg.get("w_kg", 0.2)
        # Accept both w_blip2 and w_stage2 for backward compatibility
        w_stage2 = fusion_cfg.get("w_stage2", fusion_cfg.get("w_blip2", 0.1))
        
        # Treat flat weights as "full" mode and synthesize other modes
        result = {
            "clip_only": {
                "w_clip": 1.0,
                "w_stage2": 0.0,
                "w_kg": 0.0
            },
            "hybrid": {
                "w_clip": 0.7,
                "w_stage2": 0.3,
                "w_kg": 0.0
            },
            "clip_kg": {
                "w_clip": 0.7,
                "w_stage2": 0.0,
                "w_kg": 0.3
            },
            "full": {
                "w_clip": w_clip,
                "w_stage2": w_stage2,
                "w_kg": w_kg
            },
            "default_mode": "hybrid"  # Default to hybrid for legacy configs
        }
        
        return result


def print_config_summary(cfg: ConfigDict) -> None:
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
    print(f"  default_mode: {fusion_cfg.get('default_mode', 'N/A')}")
    print("  modes:")
    for mode_name in ["clip_only", "hybrid", "clip_kg", "full"]:
        if mode_name in fusion_cfg:
            weights = fusion_cfg[mode_name]
            print(f"    {mode_name}:")
            print(f"      w_clip: {weights.get('w_clip', 0.0):.2f}")
            print(f"      w_stage2: {weights.get('w_stage2', 0.0):.2f}")
            print(f"      w_kg: {weights.get('w_kg', 0.0):.2f}")
    
    print("\n" + "=" * 70)
