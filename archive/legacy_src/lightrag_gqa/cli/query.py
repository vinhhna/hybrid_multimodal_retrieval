#!/usr/bin/env python
"""
CLI tool for querying the knowledge graph.

Usage:
    lightrag-gqa-query --scale 10k --interactive
    lightrag-gqa-query --scale 10k --query "Find all images with dogs"
"""

import argparse
import sys
from pathlib import Path

def main():
    """Main entry point for query CLI."""
    parser = argparse.ArgumentParser(
        description='Query LightRAG knowledge graph',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        '--scale',
        choices=['1k', '10k', 'full'],
        default='10k',
        help='Scale of the graph to query (default: 10k)'
    )
    
    parser.add_argument(
        '--graph',
        type=str,
        help='Path to saved graph file (optional, overrides --scale)'
    )
    
    parser.add_argument(
        '--query',
        type=str,
        help='Natural language query to execute'
    )
    
    parser.add_argument(
        '--interactive',
        action='store_true',
        help='Launch interactive query interface'
    )
    
    parser.add_argument(
        '--limit',
        type=int,
        default=10,
        help='Maximum results to return (default: 10)'
    )
    
    args = parser.parse_args()
    
    # Import here to avoid slow startup
    from lightrag_gqa.basic_queries import QueryInterface
    
    # Determine graph path
    if args.graph:
        graph_path = args.graph
    else:
        graph_path = f'experiments/sample_{args.scale}/gqa_lightrag.gpickle'
    
    if not Path(graph_path).exists():
        print(f"❌ Graph file not found: {graph_path}", file=sys.stderr)
        print(f"   Run 'lightrag-gqa-build --scale {args.scale}' first", file=sys.stderr)
        return 1
    
    print(f"📊 Loading graph from: {graph_path}")
    
    try:
        interface = QueryInterface(scale=args.scale)
        
        if args.interactive:
            print("\n🎯 Launching interactive query interface...")
            print("   Type 'help' for available commands")
            print("   Type 'exit' or 'quit' to exit\n")
            interface.interactive_mode()
            
        elif args.query:
            print(f"\n❓ Query: {args.query}")
            response = interface.execute_natural_language_query(
                nl_query=args.query,
                limit=args.limit
            )
            print(f"\n{response.formatted_output}")
            
        else:
            parser.print_help()
            print("\n❌ Please specify --query or --interactive", file=sys.stderr)
            return 1
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    sys.exit(main())
