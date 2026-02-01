#!/usr/bin/env python3
"""Test all 9 query types with the Natural Language Interface."""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from basic_queries.gqa_query_interface import QueryInterface

def main():
    interface = QueryInterface(scale='10k')
    
    test_queries = [
        ('Entity Search', 'Find all red cars'),
        ('Statistical', 'How often are dogs near people'),
        ('Similarity', 'Find objects similar to large green tree'),
        ('Relational Path', 'What paths connect man to shirt'),
        ('Negative Constraint', 'Find images with trees but without sky'),
        ('Comparative', 'Compare the number of people indoors vs outdoors'),
        ('Hierarchical', 'Show all types of vehicles'),
        ('Anomaly', 'What unusual object-relation pairs exist'),
        ('Multi-Attribute', 'Find large wooden brown tables'),
    ]
    
    print('=' * 70)
    print('COMPREHENSIVE QUERY TEST - ALL 9 TYPES')
    print('=' * 70)
    
    success_count = 0
    
    for i, (name, q) in enumerate(test_queries, 1):
        print(f'\n[{i}/9] {name}')
        print(f'      Query: "{q}"')
        try:
            result = interface.query(q)
            if result.success:
                r = result.results
                # Extract count based on result type
                if isinstance(r, dict):
                    if 'total_instances' in r:
                        count = r['total_instances']
                    elif 'count_context_a' in r:
                        count = f"A:{r.get('count_context_a',0)} B:{r.get('count_context_b',0)}"
                    elif 'frequency' in r:
                        count = f"{r.get('frequency', 0):.2%}"
                    elif 'total_results' in r:
                        count = r['total_results']
                    elif 'results' in r:
                        count = len(r['results'])
                    else:
                        count = len(r)
                elif isinstance(r, list):
                    count = len(r)
                else:
                    count = str(type(r))
                print(f'      ✅ Type: {result.query_type}, Results: {count}')
                success_count += 1
            else:
                print(f'      ❌ Error: {result.error_message}')
        except Exception as e:
            import traceback
            print(f'      ❌ Exception: {e}')
            traceback.print_exc()
    
    print('\n' + '=' * 70)
    print(f'TEST COMPLETE: {success_count}/9 queries successful')
    print('=' * 70)

if __name__ == '__main__':
    main()
