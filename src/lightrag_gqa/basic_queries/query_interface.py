"""
GQA LightRAG Query Interface
Main interface for natural language queries on the GQA Knowledge Graph.

This module provides:
1. NaturalLanguageParser - Parse English queries to structured format
2. QueryInterface - Execute queries and return results
3. Interactive CLI for testing queries

Usage:
    python gqa_query_interface.py --scale full
    python gqa_query_interface.py --query "Find all red cars"
"""

import argparse
import json
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

from .nl_parser import NaturalLanguageParser, ParseResult, QueryType
from .reasoning_engine import GQA_Reasoning_Engine, ReasoningResult


@dataclass
class QueryResponse:
    """Response from a query execution"""
    success: bool
    query_type: str
    original_query: str
    parsed_params: Dict[str, Any]
    parse_confidence: float
    results: Any
    reasoning_trace: List[str]
    metadata: Dict[str, Any]
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "query_type": self.query_type,
            "original_query": self.original_query,
            "parsed_params": self.parsed_params,
            "parse_confidence": self.parse_confidence,
            "results": self.results,
            "reasoning_trace": self.reasoning_trace,
            "metadata": self.metadata,
            "error_message": self.error_message
        }
    
    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)


class QueryInterface:
    """
    Main interface for executing natural language queries on GQA LightRAG.
    """
    
    def __init__(self, scale: str = 'full', verbose: bool = True):
        """
        Initialize the query interface.
        
        Args:
            scale: Dataset scale ('1k', '10k', or 'full')
            verbose: Whether to print status messages
        """
        self.scale = scale
        self.verbose = verbose
        
        if verbose:
            print(f"[INFO] Initializing Query Interface with scale: {scale}")
        
        # Initialize parser
        self.parser = NaturalLanguageParser()
        
        # Initialize reasoning engine
        if verbose:
            print("[INFO] Loading Knowledge Graph...")
        self.engine = GQA_Reasoning_Engine(scale=scale)
        
        if verbose:
            print("[INFO] Query Interface ready!")
            print(f"       Nodes: {self.engine.graph.number_of_nodes():,}")
            print(f"       Edges: {self.engine.graph.number_of_edges():,}")
    
    def query(self, natural_language_query: str, limit: int = 10) -> QueryResponse:
        """
        Process a natural language query and return results.
        
        Args:
            natural_language_query: English query string
            limit: Maximum number of results to return (default: 10)
            
        Returns:
            QueryResponse with results and metadata
        """
        # Step 1: Parse the query
        parse_result = self.parser.parse(natural_language_query)
        
        if not parse_result.is_supported():
            return QueryResponse(
                success=False,
                query_type="unknown",
                original_query=natural_language_query,
                parsed_params={},
                parse_confidence=0.0,
                results=None,
                reasoning_trace=[],
                metadata={},
                error_message="Could not understand the query. Please try rephrasing."
            )
        
        # Step 2: Execute the appropriate query
        try:
            result = self._execute_query(parse_result, limit=limit)
            
            return QueryResponse(
                success=True,
                query_type=parse_result.query_type.value,
                original_query=natural_language_query,
                parsed_params=parse_result.params,
                parse_confidence=parse_result.confidence,
                results=result.results,
                reasoning_trace=result.reasoning_trace,
                metadata=result.metadata
            )
            
        except Exception as e:
            return QueryResponse(
                success=False,
                query_type=parse_result.query_type.value,
                original_query=natural_language_query,
                parsed_params=parse_result.params,
                parse_confidence=parse_result.confidence,
                results=None,
                reasoning_trace=[],
                metadata={},
                error_message=f"Error executing query: {str(e)}"
            )
    
    def _execute_query(self, parse_result: ParseResult, limit: int = 10) -> ReasoningResult:
        """
        Execute query based on parsed result.
        
        Args:
            parse_result: Parsed query result
            limit: Maximum number of results to return
        """
        params = parse_result.params
        query_type = parse_result.query_type
        
        if query_type == QueryType.ENTITY_SEARCH:
            return self.engine.entity_search(
                concept=params.get('concept'),
                attributes=params.get('attributes', []),
                limit=limit
            )
            
        elif query_type == QueryType.STATISTICAL_KNOWLEDGE:
            return self.engine.statistical_knowledge(
                concept_a=params.get('concept_a'),
                concept_b=params.get('concept_b'),
                relation=params.get('relation')
            )
            
        elif query_type == QueryType.SIMILARITY_SEARCH:
            return self.engine.similarity_search(
                concept=params.get('concept'),
                attributes=params.get('attributes', []),
                limit=limit
            )
            
        elif query_type == QueryType.RELATIONAL_PATH:
            return self.engine.relational_path(
                source_concept=params.get('source_concept'),
                target_concept=params.get('target_concept'),
                via_relation=params.get('via_relation'),
                max_hops=2,
                limit=min(limit, 20)  # Cap at 20 for path queries to avoid too many results
            )
            
        elif query_type == QueryType.NEGATIVE_CONSTRAINTS:
            return self.engine.negative_constraints(
                concept_present=params.get('concept_present'),
                concept_absent=params.get('concept_absent'),
                limit=limit
            )
            
        elif query_type == QueryType.COMPARATIVE:
            target = params.get('target_concept')
            context_a = params.get('context_a')
            context_b = params.get('context_b')
            
            if target and context_a and context_b:
                return self.engine.compare_contexts(
                    context_a=context_a,
                    context_b=context_b,
                    target_concept=target
                )
            else:
                # Attribute comparison
                return self.engine.compare_attribute_distribution(
                    concept_a=params.get('concept_a'),
                    concept_b=params.get('concept_b'),
                    attribute=params.get('attribute', 'white')
                )
                
        elif query_type == QueryType.HIERARCHICAL:
            category = params.get('category')
            
            # Normalize category name (remove trailing 's' for plurals if needed)
            # The engine uses singular forms: 'vehicle', 'animal', etc.
            if category and category.endswith('s'):
                singular = category[:-1]
                if singular in self.engine.HIERARCHY_MAPPING:
                    category = singular
            
            return self.engine.get_hierarchical_entities(
                parent_category=category,
                limit=limit
            )
            
        elif query_type == QueryType.ANOMALY_DETECTION:
            if params.get('find_rare_relations'):
                return self.engine.find_anomalies(
                    min_frequency=2,
                    limit=params.get('top_n', limit)  # Use top_n if specified in a query, otherwise use limit parameter
                )
            else:
                return self.engine.find_specific_anomaly(
                    subject_concept=params.get('subject_concept'),
                    relation=params.get('relation'),
                    object_concept=params.get('object_concept')
                )
                
        elif query_type == QueryType.VISUAL_ATTRIBUTE_CONSTRAINT:
            if params.get('main_concept') and params.get('related_concept'):
                return self.engine.complex_constraint_search(
                    main_concept=params.get('main_concept'),
                    main_attributes=params.get('main_attributes', []),
                    relation=params.get('relation'),
                    related_concept=params.get('related_concept'),
                    related_attributes=params.get('related_attributes', []),
                    limit=limit
                )
            else:
                return self.engine.multi_constraint_search(
                    name=params.get('concept'),
                    required_attributes=params.get('attributes', []),
                    limit=limit
                )
        
        # Default: return an empty result
        return ReasoningResult(
            query_type="unknown",
            question=parse_result.original_query,
            results=[],
            reasoning_trace=["Query type not implemented"],
            metadata={}
        )
    
    def get_supported_queries(self) -> Dict[str, List[str]]:
        """Get example queries for each supported query type"""
        return self.parser.get_supported_patterns()
    
    def interactive_mode(self):
        """Run interactive query mode"""
        print("\n" + "=" * 70)
        print("GQA LIGHTRAG INTERACTIVE QUERY INTERFACE")
        print("=" * 70)
        print(f"Scale: {self.scale}")
        print(f"Nodes: {self.engine.graph.number_of_nodes():,}")
        print(f"Edges: {self.engine.graph.number_of_edges():,}")
        print("=" * 70)
        print("\nType 'help' for example queries")
        print("Type 'exit' or 'quit' to exit")
        print("=" * 70)
        
        while True:
            try:
                query = input("\n> Enter query: ").strip()
                
                if not query:
                    continue
                    
                if query.lower() in ['exit', 'quit', 'q']:
                    print("Goodbye!")
                    break
                    
                if query.lower() == 'help':
                    self._print_help()
                    continue
                
                # Execute query
                print("\n[INFO] Processing query...")
                response = self.query(query)
                
                # Print results
                self._print_response(response)
                
            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"\n[ERROR] {str(e)}")
    
    def _print_help(self):
        """Print help message with example queries"""
        examples = self.get_supported_queries()
        
        print("\n" + "=" * 70)
        print("SUPPORTED QUERY TYPES AND EXAMPLES")
        print("=" * 70)
        
        for query_type, examples_list in examples.items():
            print(f"\n{query_type.upper().replace('_', ' ')}:")
            for example in examples_list[:3]:
                print(f"  - {example}")
        
        print("\n" + "=" * 70)

    def _print_response(self, response: QueryResponse):
        """Print query response in a readable format"""
        print("\n" + "=" * 70)
        print("QUERY RESULTS")
        print("=" * 70)

        print(f"\nQuery Type: {response.query_type.upper().replace('_', ' ')}")
        print(f"Parse Confidence: {response.parse_confidence:.2f}")
        print(f"Parsed Parameters: {json.dumps(response.parsed_params, indent=2)}")

        if not response.success:
            print(f"\n[ERROR] {response.error_message}")
            return

        print("\n--- REASONING TRACE ---")
        for i, step in enumerate(response.reasoning_trace, 1):
            print(f"  {step}")

        print("\n--- RESULTS ---")
        if response.results is None:
            print("  No results found.")
        elif isinstance(response.results, list):
            if len(response.results) == 0:
                print("  No results found.")
            else:
                print(f"  Found {len(response.results)} results:")
                for i, item in enumerate(response.results[:5], 1):
                    if isinstance(item, dict):
                        # Compact display
                        display = {k: v for k, v in item.items() if k in [
                            'node_id', 'image_id', 'name', 'attributes',
                            'count', 'probability', 'path', 'relation'
                        ]}
                        print(f"  {i}. {display}")
                    else:
                        print(f"  {i}. {item}")
                if len(response.results) > 5:
                    print(f"  ... and {len(response.results) - 5} more")
        elif isinstance(response.results, dict):
            for key, value in response.results.items():
                if isinstance(value, dict) and 'count' in value:
                    print(f"  {key}: {value['count']} instances")
                elif isinstance(value, (int, float, str)):
                    print(f"  {key}: {value}")

        print("\n--- METADATA ---")
        meta = response.metadata
        if meta:
            for key, value in meta.items():
                if isinstance(value, list) and len(value) > 3:
                    print(f"  {key}: [{value[0]}, {value[1]}, {value[2]}, ...]")
                elif isinstance(value, dict) and len(value) > 3:
                    keys = list(value.keys())[:3]
                    print(f"  {key}: {{{keys[0]}: ..., ...}}")
                else:
                    print(f"  {key}: {value}")

        print("\n" + "=" * 70)


# ============================================================
# Command Line Interface
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="GQA LightRAG Query Interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python gqa_query_interface.py --scale full
  python gqa_query_interface.py --scale 10k --query "Find all red cars"
  python gqa_query_interface.py --query "What is the probability of finding shirt near man"
        """
    )
    
    parser.add_argument(
        '--scale',
        choices=['1k', '10k', 'full'],
        default='full',
        help='Dataset scale (default: full)'
    )
    
    parser.add_argument(
        '--query', '-q',
        type=str,
        help='Single query to execute (if not provided, enters interactive mode)'
    )
    
    parser.add_argument(
        '--json',
        action='store_true',
        help='Output results as JSON'
    )
    
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress loading messages'
    )
    
    args = parser.parse_args()
    
    # Initialize interface
    interface = QueryInterface(scale=args.scale, verbose=not args.quiet)
    
    if args.query:
        # Single query mode
        response = interface.query(args.query)
        
        if args.json:
            print(response.to_json())
        else:
            interface._print_response(response)
    else:
        # Interactive mode
        interface.interactive_mode()


if __name__ == "__main__":
    main()
