#!/usr/bin/env python
"""
CLI tool for running evaluations.

Usage:
    lightrag-gqa-eval --config evaluation_v0_1/configs/eval.yaml
"""

import argparse
import sys
from pathlib import Path

def main():
    """Main entry point for evaluation CLI."""
    parser = argparse.ArgumentParser(
        description='Run LightRAG evaluation framework',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='evaluation_v0_1/configs/eval.yaml',
        help='Path to evaluation config file'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        help='Output directory for results (optional)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    
    args = parser.parse_args()
    
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"❌ Config file not found: {config_path}", file=sys.stderr)
        return 1
    
    print(f"🔬 Running evaluation...")
    print(f"   Config: {config_path}")
    
    try:
        # Import evaluation runner
        # For now, use the existing runner directly
        import subprocess
        
        # Run the evaluation script
        cmd = [sys.executable, 'evaluation_v0_1/run_evaluation.py']
        if args.verbose:
            cmd.append('--verbose')
        
        result = subprocess.run(cmd, check=True)
        
        print(f"\n✅ Evaluation completed!")
        return result.returncode
        
    except Exception as e:
        print(f"\n❌ Error running evaluation: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

if __name__ == '__main__':
    sys.exit(main())
