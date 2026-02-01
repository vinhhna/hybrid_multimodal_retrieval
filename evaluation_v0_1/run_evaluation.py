"""
Main entry point for the Evaluation Framework v0.1

This script serves as the main entry point to run the evaluation pipeline.
It properly sets up the Python path and imports to avoid relative import issues.

Usage:
    python -m evaluation_v0_1.run_evaluation --config evaluation_v0_1/configs/eval.yaml
    python evaluation_v0_1/run_evaluation.py --config evaluation_v0_1/configs/eval.yaml --verbose
"""

import sys
from pathlib import Path

# Add repo root to path
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Now import and run the main script
from evaluation_v0_1.scripts.run_all import main

if __name__ == "__main__":
    main()
