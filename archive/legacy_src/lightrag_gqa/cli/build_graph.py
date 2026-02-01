#!/usr/bin/env python
"""
CLI tool for building knowledge graphs from GQA scene graphs.

Usage:
    lightrag-gqa-build --scale 10k --input sceneGraphs/train_sceneGraphs.json --output experiments/sample_10k
"""

import argparse
import sys
from pathlib import Path

def main():
    """Main entry point for graph building CLI."""
    parser = argparse.ArgumentParser(
        description='Build LightRAG knowledge graph from GQA scene graphs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        '--scale',
        choices=['1k', '10k', 'full'],
        default='10k',
        help='Scale of the graph to build (default: 10k)'
    )
    
    parser.add_argument(
        '--input',
        type=str,
        default='sceneGraphs/train_sceneGraphs.json',
        help='Path to GQA scene graphs JSON file'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        help='Output directory for the built graph'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    
    args = parser.parse_args()
    
    # Import here to avoid slow startup
    from lightrag_gqa.basic_queries import GQALightRAGGraphBuilder
    
    # Determine output path
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = Path('experiments') / f'sample_{args.scale}'
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"🚀 Building {args.scale} scale graph...")
    print(f"   Input: {args.input}")
    print(f"   Output: {output_dir}")
    
    try:
        builder = GQALightRAGGraphBuilder(scale=args.scale)
        graph_path = builder.build_from_file(
            scene_graph_file=args.input,
            output_path=str(output_dir)
        )
        
        print(f"\n✅ Graph built successfully!")
        print(f"   Saved to: {graph_path}")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Error building graph: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

if __name__ == '__main__':
    sys.exit(main())
