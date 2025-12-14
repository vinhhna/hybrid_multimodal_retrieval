"""
Configuration helpers for entity-centric graph and hybrid retrieval.

This module provides utilities for loading and accessing configuration
from YAML files. It centralizes config access and provides type-safe interfaces
for entity_graph, query_enrichment, graph_search, fusion, and phase5 sections.

Phase 5 configuration includes vision-grounded entity graph with open-vocabulary
object detection, visual priors, and binding verification.
"""

from __future__ import annotations

import yaml
from pathlib import Path
from typing import Any, Dict

# Type alias for cleaner signatures
ConfigDict = Dict[str, Any]


def load_entity_graph_config(path: str | Path) -> ConfigDict:
    """
    Load entity graph configuration from a YAML file.

    This helper provides a single entry point to read sections:
      - entity_graph: paths, thresholds (min_df, k_sem, degree_cap, etc.)
      - query_enrichment: K_seed_raw, M_enrich, templates, etc.
      - graph_search: K_seed, H_max, B, T_cap_ms, decay, edge weights
      - fusion: Per-mode weights with nested schema:
            fusion:
              default_mode: full
              modes:
                clip_only: {w_clip: 1.0, w_stage2: 0.0, w_kg: 0.0}
                hybrid: {w_clip: 0.7, w_stage2: 0.3, w_kg: 0.0}
                clip_kg: {w_clip: 0.7, w_stage2: 0.0, w_kg: 0.3}
                full: {w_clip: 0.6, w_stage2: 0.2, w_kg: 0.2}
      - phase5: Vision-grounded entity graph config (optional)

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
    Extract fusion configuration section with per-mode weights.
    
    Requires nested schema with explicit modes (legacy flat schema no longer supported):
    
        fusion:
          default_mode: full
          modes:
            clip_only: {w_clip: 1.0, w_stage2: 0.0, w_kg: 0.0}
            hybrid: {w_clip: 0.7, w_stage2: 0.3, w_kg: 0.0}
            clip_kg: {w_clip: 0.7, w_stage2: 0.0, w_kg: 0.3}
            full: {w_clip: 0.6, w_stage2: 0.2, w_kg: 0.2}
    
    Args:
        cfg: Full configuration dictionary loaded from YAML.

    Returns:
        Dictionary with per-mode fusion weights and default_mode:
        {
            "clip_only": {"w_clip": 1.0, "w_stage2": 0.0, "w_kg": 0.0},
            "hybrid": {"w_clip": 0.7, "w_stage2": 0.3, "w_kg": 0.0},
            "clip_kg": {"w_clip": 0.7, "w_stage2": 0.0, "w_kg": 0.3},
            "full": {"w_clip": 0.6, "w_stage2": 0.2, "w_kg": 0.2},
            "default_mode": "full"
        }
    
    Raises:
        ValueError: If fusion section is missing or malformed (legacy flat schema).
    
    Modes:
      - clip_only: CLIP Stage 1 only (fastest, no reranking, no KG)
      - hybrid: CLIP + BLIP-2 reranking (no KG)
      - clip_kg: CLIP + KG (no BLIP-2, for KG experiments)
      - full: CLIP + BLIP-2 + KG (full hybrid retrieval)
    
    Example:
        >>> cfg = load_entity_graph_config("configs/entity_graph.yaml")
        >>> fusion_cfg = get_fusion_config(cfg)
        >>> print(fusion_cfg["default_mode"])  # "full"
        >>> print(fusion_cfg["full"]["w_clip"])  # 0.6
    """
    fusion_cfg = cfg.get("fusion", {})
    
    # Require nested schema with modes
    if "modes" not in fusion_cfg or not isinstance(fusion_cfg["modes"], dict):
        raise ValueError(
            "Invalid fusion configuration: missing 'modes' section. "
            "Expected nested schema in configs/entity_graph.yaml:\n"
            "  fusion:\n"
            "    default_mode: full\n"
            "    modes:\n"
            "      clip_only: {w_clip: 1.0, w_stage2: 0.0, w_kg: 0.0}\n"
            "      hybrid: {w_clip: 0.7, w_stage2: 0.3, w_kg: 0.0}\n"
            "      clip_kg: {w_clip: 0.7, w_stage2: 0.0, w_kg: 0.3}\n"
            "      full: {w_clip: 0.6, w_stage2: 0.2, w_kg: 0.2}\n"
            "Legacy flat fusion schema (w_clip, w_kg, w_blip2) is no longer supported."
        )
    
    modes = fusion_cfg["modes"]
    default_mode = fusion_cfg.get("default_mode", "full")
    
    # Build result with defaults for missing modes
    result = {
        "clip_only": modes.get("clip_only", {
            "w_clip": 1.0, "w_stage2": 0.0, "w_kg": 0.0
        }),
        "hybrid": modes.get("hybrid", {
            "w_clip": 0.7, "w_stage2": 0.3, "w_kg": 0.0
        }),
        "clip_kg": modes.get("clip_kg", {
            "w_clip": 0.7, "w_stage2": 0.0, "w_kg": 0.3
        }),
        "full": modes.get("full", {
            "w_clip": 0.6, "w_stage2": 0.2, "w_kg": 0.2
        }),
        "default_mode": default_mode
    }
    
    return result


def print_config_summary(cfg: ConfigDict) -> None:
    """
    Print a human-readable summary of the configuration.

    Useful for logging at the start of scripts to provide transparency
    about which hyperparameters are being used.

    Args:
        cfg: Full configuration dictionary loaded from YAML.
    """
    print("\n" + "=" * 70)
    print("CONFIGURATION SUMMARY")
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


def get_phase5_config(cfg: ConfigDict) -> ConfigDict:
    """
    Extract Phase 5 configuration section with defaults.
    
    Phase 5 (v3.1) introduces vision-grounded entity graph and hybrid retrieval
    with open-vocabulary object detection, visual priors, and binding verification.
    
    This function provides safe defaults for all Phase 5 keys. If the phase5 
    section is missing from the config, returns an empty dict (Phase 5 features 
    disabled).
    
    Args:
        cfg: Full configuration dictionary loaded from YAML.
    
    Returns:
        Dictionary with phase5 config keys and defaults applied.
        Returns empty dict if phase5 section not present (Phase 5 disabled).
    
    Example:
        >>> cfg = load_entity_graph_config("configs/entity_graph.yaml")
        >>> phase5_cfg = get_phase5_config(cfg)
        >>> if phase5_cfg:  # Check if Phase 5 features are enabled
        ...     detector_cfg = phase5_cfg.get("detector", {})
        ...     print(detector_cfg.get("model_id", "owlvit"))
    """
    phase5_cfg = cfg.get("phase5", {})
    
    # If phase5 section is missing, return empty dict (Phase 5 disabled)
    if not phase5_cfg:
        return {}
    
    # Apply defaults for each subsection
    defaults = {
        "candidates": {
            "vis_prior_topK": 50,
            "tau_vis": 0.20,
            "C_max": 128,
        },
        "safe_neighbors": {
            "tau_sim": 0.25,
            "use_mutual_knn": True,
            "min_df_vis": 5,
            "require_in_vis_prior": False,
        },
        "detector": {
            "model_id": "owlvit",
            "batch_size_images": 8,
            "batch_size_phrases": 128,
            "cache_dir": "data/vision/cache",
            "raw_shards_dir": "data/vision/raw_shards",
            "post_shards_dir": "data/vision/post_shards",
            "shard_size": 500,
        },
        "negative_controls": {
            "null_phrases": ["xyzzy nonsense phrase 1", "blark random phrase 2"],
            "noise_percentile": 95,
            "delta": 0.05,
        },
        "thresholds": {
            "strategy": "grouped",
            "objective": "fixed_fpr",
            "target_fpr": 0.05,
            "target_precision": 0.90,
            "min_support": 200,
        },
        "binding": {
            "enable": False,
            "topN_images": 50,
            "topB_boxes": 5,
            "hsv_ranges": {},
            "clip_fallback": {
                "enable": True,
                "prompt_template": "a photo of a {attr} {obj}",
            },
        },
        "pmi": {
            "smoothing_eps": 1.0,
            "npmi_min": 0.20,
            "npmi_min_stop": 0.20,
            "degree_cap": 500,
            "per_node_topk": 200,
            "stop_nodes_topM_df": 50,
        },
    }
    
    # Merge defaults with actual config (actual config takes precedence)
    result = {}
    for section_key, section_defaults in defaults.items():
        section_cfg = phase5_cfg.get(section_key, {})
        result[section_key] = {**section_defaults, **section_cfg}
    
    return result
