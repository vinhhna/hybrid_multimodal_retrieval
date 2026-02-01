#!/usr/bin/env python3
"""
Natural Language Query Interface Demo
Demonstrates all 9 query types supported by the GQA LightRAG system.
"""

import sys
import io
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Set UTF-8 encoding for stdout
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from basic_queries.gqa_query_interface import QueryInterface

def print_separator(char='=', length=80):
    print(char * length)

def print_header(title):
    print_separator()
    print(f"  {title}")
    print_separator()

def demo_query(interface, name, query, description):
    """Execute a demo query and print results."""
    print(f"\n{'='*80}")
    print(f"  {name}")
    print(f"  Description: {description}")
    print(f"  Query: \"{query}\"")
    print('='*80)
    
    result = interface.query(query)
    
    if result.success:
        print(f"\n  ✅ Query Type: {result.query_type}")
        print(f"  📊 Parse Confidence: {result.parse_confidence:.2f}")
        
        # Print reasoning trace (first 3 steps)
        print("\n  --- REASONING TRACE (excerpt) ---")
        for trace in result.reasoning_trace[:3]:
            print(f"    {trace}")
        if len(result.reasoning_trace) > 3:
            print(f"    ... and {len(result.reasoning_trace) - 3} more steps")
        
        # Print results summary
        print("\n  --- RESULTS SUMMARY ---")
        r = result.results
        if isinstance(r, list):
            print(f"    Found {len(r)} results")
            for item in r[:3]:
                if isinstance(item, dict):
                    name = item.get('name', item.get('instance_id', 'unknown'))
                    print(f"      • {name}")
                else:
                    print(f"      • {item}")
            if len(r) > 3:
                print(f"      ... and {len(r) - 3} more")
        elif isinstance(r, dict):
            for key, value in list(r.items())[:5]:
                if isinstance(value, dict):
                    print(f"    {key}: {len(value)} items")
                elif isinstance(value, list):
                    print(f"    {key}: {len(value)} items")
                else:
                    print(f"    {key}: {value}")
    else:
        print(f"\n  ❌ Error: {result.error_message}")
    
    return result.success

def main():
    print("\n" + "="*80)
    print("  GQA LightRAG Natural Language Query Interface Demo")
    print("  All queries must be in English")
    print("="*80)
    
    # Initialize interface
    print("\n[INFO] Loading Knowledge Graph (10k scale)...")
    interface = QueryInterface(scale='10k', verbose=False)
    print(f"[INFO] Graph loaded: {interface.engine.graph.number_of_nodes():,} nodes, "
          f"{interface.engine.graph.number_of_edges():,} edges")
    
    # Define demo queries
    demos = [
        ("1. ENTITY SEARCH", 
         "Find all red cars",
         "Search for entities with specific attributes (color: red, type: car)"),
        
        ("2. STATISTICAL KNOWLEDGE",
         "How often are dogs near people",
         "Compute probability/co-occurrence statistics between concepts"),
        
        ("3. SIMILARITY SEARCH",
         "Find objects similar to large green tree",
         "Find entities with similar attributes to a reference"),
        
        ("4. RELATIONAL PATH",
         "What paths connect man to shirt",
         "Discover relationships/paths between two concepts"),
        
        ("5. NEGATIVE CONSTRAINTS",
         "Find images with trees but without sky",
         "Find scenes containing one concept but not another"),
        
        ("6. COMPARATIVE ANALYSIS",
         "Compare the number of people indoors vs outdoors",
         "Compare distribution of concepts across different contexts"),
        
        ("7. HIERARCHICAL ENTITIES",
         "Show all types of vehicles",
         "Retrieve all entities belonging to a category hierarchy"),
        
        ("8. ANOMALY DETECTION",
         "What unusual object-relation pairs exist",
         "Find rare/unusual patterns and relationships"),
        
        ("9. MULTI-ATTRIBUTE SEARCH",
         "Find large wooden brown tables",
         "Search with multiple attribute constraints"),
    ]
    
    # Run demos
    success_count = 0
    for name, query, description in demos:
        if demo_query(interface, name, query, description):
            success_count += 1
    
    # Summary
    print("\n" + "="*80)
    print(f"  DEMO COMPLETE: {success_count}/{len(demos)} queries successful")
    print("="*80)
    
    # Additional example queries
    print("\n" + "="*80)
    print("  ADDITIONAL EXAMPLE QUERIES YOU CAN TRY")
    print("="*80)
    example_queries = [
        "Find white plates on tables",
        "What is the probability of window near building",
        "Find similar objects to a wooden chair",
        "How is plate connected to table",
        "Images with dogs but not cats",
        "Which has more chairs: kitchen or living room",
        "Show all types of animals",
        "Find unusual cases of dog on table",
        "Find small metal silver forks",
        "Connection between car and road",
    ]
    for i, q in enumerate(example_queries, 1):
        print(f"  {i:2}. {q}")
    
    print("\n" + "="*80)
    print("  To run your own queries, use:")
    print("    python gqa_query_interface.py --scale full --query \"your query here\"")
    print("  Or use interactive mode:")
    print("    python gqa_query_interface.py --scale full")
    print("="*80)

if __name__ == '__main__':
    main()
