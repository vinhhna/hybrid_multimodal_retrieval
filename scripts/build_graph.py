#!/usr/bin/env python3
"""
Build LightRAG-style knowledge graph from VG150 dataset.

Usage:
    python scripts/build_graph.py --data_dir ./data --out_dir ./output
    python scripts/build_graph.py --data_dir ./data --out_dir ./output --sample 100
"""

import sys
from pathlib import Path

# Add src to path for direct execution
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lightrag_vg150.cli import build_main

if __name__ == "__main__":
    build_main()
