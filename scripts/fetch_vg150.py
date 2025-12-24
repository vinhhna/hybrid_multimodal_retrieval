#!/usr/bin/env python3
"""
Fetch and validate VG150 dataset files.

Usage:
    python scripts/fetch_vg150.py --data_dir ./data
    python scripts/fetch_vg150.py --data_dir ./data --sample 100
"""

import sys
from pathlib import Path

# Add src to path for direct execution
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lightrag_vg150.cli import fetch_main

if __name__ == "__main__":
    fetch_main()
