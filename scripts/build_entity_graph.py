#!/usr/bin/env python

"""
Entity graph builder script for Phase 4.

This is the canonical builder for the entity graph artifact (entity_graph.pt).
It loads entity embeddings and context, constructs the graph with semantic
and co-occurrence edges, and saves it to the configured path.

Usage (from repo root):
    python scripts/build_entity_graph.py [--data-folder PATH]
    
    or
    
    python -m scripts.build_entity_graph [--data-folder PATH]

Arguments:
    --data-folder PATH     Path to the data folder for reading inputs (default: /kaggle/input/flickr30k)
    --output-folder PATH   Path to the output folder for saving graph (default: /kaggle/working)

Output:
    - {output-folder}/entity_graph.pt (or path from config)
    
For testing the graph construction logic, use:
    pytest tests/test_entity_graph_build.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


def main() -> None:
    """
    Main builder function.
    
    Loads configuration, entity artifacts, builds the graph, and saves it.
    Prints a summary of the constructed graph.
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Build entity graph for Phase 4",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--data-folder",
        type=str,
        default="/kaggle/input/flickr30k",
        help="Path to the data folder for reading inputs (default: /kaggle/input/flickr30k)"
    )
    parser.add_argument(
        "--output-folder",
        type=str,
        default="/kaggle/working",
        help="Path to the output folder for saving graph (default: /kaggle/working)"
    )
    args = parser.parse_args()
    
    data_folder = Path(args.data_folder)
    output_folder = Path(args.output_folder)
    
    # Resolve project root and add to path
    project_root = Path(__file__).resolve().parent.parent
    if project_root not in sys.path:
        sys.path.insert(0, str(project_root))
    
    from src.graph.config import load_entity_graph_config, get_entity_graph_config
    from src.graph.build_entity_graph import build_entity_graph, save_entity_graph
    
    print("=" * 70)
    print("ENTITY GRAPH BUILDER - PHASE 4 DAY 5-7")
    print("=" * 70)
    print(f"\n[Paths]")
    print(f"  Input data folder: {data_folder}")
    print(f"  Output folder: {output_folder}")
    
    # Load configuration
    config_path = project_root / "configs" / "entity_graph.yaml"
    if not config_path.exists():
        print(f"\n✗ Config file not found: {config_path}")
        print("  Ensure configs/entity_graph.yaml exists in the project root.")
        sys.exit(1)
    
    print(f"\n[Config]")
    print(f"  Loading from: {config_path}")
    cfg = load_entity_graph_config(str(config_path))
    cfg_entity_graph = get_entity_graph_config(cfg)
    
    print(f"  ✓ Config loaded")
    print(f"    k_sem: {cfg_entity_graph['k_sem']}")
    print(f"    degree_cap: {cfg_entity_graph['degree_cap']}")
    
    # Resolve artifact paths
    embeddings_path_cfg = Path(cfg_entity_graph["entity_embeddings_path"])
    context_path_cfg = Path(cfg_entity_graph["context_path"])
    graph_path_cfg = Path(cfg_entity_graph["entity_graph_path"])
    
    # Input paths: use data_folder as base if paths are relative
    if embeddings_path_cfg.is_absolute():
        embeddings_path = embeddings_path_cfg
    else:
        embeddings_path = data_folder / embeddings_path_cfg
    
    if context_path_cfg.is_absolute():
        context_path = context_path_cfg
    else:
        context_path = data_folder / context_path_cfg
    
    # Output path: use output_folder as base if path is relative
    if graph_path_cfg.is_absolute():
        graph_path = graph_path_cfg
    else:
        # Use only the filename from config, save to output_folder
        graph_path = output_folder / graph_path_cfg.name
    
    # Ensure output directory exists
    graph_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Check prerequisites
    if not embeddings_path.exists():
        print(f"\n✗ Entity embeddings not found: {embeddings_path}")
        print("\n  Please run the entity vocabulary builder first:")
        print("    python -m scripts.build_entity_vocabulary")
        sys.exit(1)
    
    if not context_path.exists():
        print(f"\n✗ Entity context not found: {context_path}")
        print("\n  Please run the entity vocabulary builder first:")
        print("    python -m scripts.build_entity_vocabulary")
        sys.exit(1)
    
    # Load entity embeddings
    print(f"\n[Loading Artifacts]")
    print(f"  Embeddings: {embeddings_path}")
    entity_embeddings = torch.load(str(embeddings_path), map_location="cpu")
    print(f"  ✓ Loaded embeddings: shape {list(entity_embeddings.shape)}, dtype {entity_embeddings.dtype}")
    
    # Load entity context
    print(f"  Context: {context_path}")
    with context_path.open("r", encoding="utf-8") as f:
        entity_context_raw = json.load(f)
    
    # Convert string keys to int keys (if necessary)
    entity_context = {
        int(eid): ctx
        for eid, ctx in entity_context_raw.items()
    }
    
    print(f"  ✓ Loaded context for {len(entity_context)} entities")
    
    # Build entity graph
    print(f"\n[Building Graph]")
    graph = build_entity_graph(entity_embeddings, entity_context, cfg)
    
    # Save graph to canonical location
    print(f"\n[Saving Graph]")
    print(f"  Output: {graph_path}")
    save_entity_graph(graph, graph_path)
    
    # Print summary
    num_entities = graph["entity"].x.size(0)
    num_sem_edges = graph["entity", "sem", "entity"].edge_index.size(1)
    num_cooc_edges = graph["entity", "cooc", "entity"].edge_index.size(1)
    
    print("\n" + "=" * 70)
    print("BUILD COMPLETE")
    print("=" * 70)
    print(f"\n  ✓ Successfully built and saved entity graph!")
    print(f"\n  Output file: {graph_path}")
    print(f"\n  Graph summary:")
    print(f"    Entities: {num_entities:,}")
    print(f"    Semantic edges: {num_sem_edges:,}")
    print(f"    Co-occurrence edges: {num_cooc_edges:,}")
    print(f"\n  Next steps:")
    print(f"    - Run tests: pytest tests/test_entity_graph_build.py")
    print(f"    - Implement query enrichment (Phase 4 Day 8-9)")
    print(f"    - Implement graph search (Phase 4 Day 10-12)")
    print("=" * 70)


if __name__ == "__main__":
    main()
