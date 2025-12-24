#!/usr/bin/env python3
"""
Query the LightRAG-style knowledge graph.

Usage:
    python scripts/query_graph.py --graph_dir ./output --query "person riding horse"
    python scripts/query_graph.py --graph_dir ./output --query "dog on grass" --top_k 10
"""

import sys
from pathlib import Path

# Add src to path for direct execution
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lightrag_vg150.cli import query_main

if __name__ == "__main__":
    query_main()
