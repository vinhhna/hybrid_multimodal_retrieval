"""
Advanced Query Evaluation Script for Evaluation Framework v0.1

Evaluates advanced query types (6-9) using curated toy suites:
- Type 6: Chain Reasoning
- Type 7: Pattern Matching
- Type 8: Scene Comparison
- Type 9: Counterfactual Reasoning

These queries require:
1. Curated gold outputs (not automatically derivable from scene graphs)
2. Potentially subjective correctness criteria
3. Trace quality assessment (placeholder for future work)

Usage:
    python -m evaluation_v0_1.scripts.evaluate_advanced --config evaluation_v0_1/configs/eval.yaml
"""

import sys
import json
import csv
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass

# Add repo root to path
REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))


@dataclass
class TraceQualityRubric:
    """
    Rubric for assessing reasoning trace quality.
    
    FUTURE WORK: Currently a placeholder. Full implementation would require:
    - Manual annotation of expected trace elements
    - Automated or human scoring against rubric
    
    Criteria:
    - completeness: Mentions all key reasoning steps (0-1)
    - correctness: Intermediate reasoning is logically valid (0-1)
    - clarity: Trace fields are structured and readable (0-1)
    """
    completeness: float = 0.0
    correctness: float = 0.0
    clarity: float = 0.0
    
    # Flag indicating this is a placeholder
    is_placeholder: bool = True
    
    def overall_score(self) -> float:
        """Weighted average of rubric scores."""
        if self.is_placeholder:
            return 0.0
        return (self.completeness + self.correctness + self.clarity) / 3
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "completeness": self.completeness,
            "correctness": self.correctness,
            "clarity": self.clarity,
            "overall": self.overall_score(),
            "is_placeholder": self.is_placeholder
        }


class AdvancedQueryEvaluator:
    """
    Evaluator for advanced query types (6-9).
    """
    
    def __init__(self, config: Dict[str, Any], verbose: bool = False):
        """
        Initialize the evaluator.
        
        Args:
            config: Configuration dictionary
            verbose: Whether to print progress
        """
        self.config = config
        self.verbose = verbose
        
        # Load advanced engine if available
        self.engine = None
        self._load_engine()
    
    def _load_engine(self) -> None:
        """Load the advanced reasoning engine."""
        try:
            from src.lightrag_gqa.advanced_queries.reasoning_engine import (
                AdvancedReasoningEngine
            )
            
            paths = self.config.get("paths", {})
            graph_path = paths.get("kg_graph_path")
            
            if graph_path:
                full_path = REPO_ROOT / graph_path
                if full_path.exists():
                    if self.verbose:
                        print(f"[AdvancedEval] Loading engine from: {full_path}")
                    self.engine = AdvancedReasoningEngine(graph_path=str(full_path))
                else:
                    print(f"[AdvancedEval] WARNING: Graph not found: {full_path}")
            else:
                if self.verbose:
                    print("[AdvancedEval] Loading engine with default scale")
                self.engine = AdvancedReasoningEngine(scale='10k')
                
        except Exception as e:
            print(f"[AdvancedEval] WARNING: Could not load engine: {e}")
            self.engine = None
    
    def evaluate_suite(self, suite_path: str) -> Dict[str, Any]:
        """
        Evaluate the engine on an advanced query suite.
        
        Args:
            suite_path: Path to the JSON suite file
            
        Returns:
            Evaluation results
        """
        if self.verbose:
            print(f"\n[AdvancedEval] Loading suite: {suite_path}")
        
        with open(suite_path, 'r', encoding='utf-8') as f:
            suite_data = json.load(f)
        
        suite_info = suite_data.get("suite_info", {})
        queries = suite_data.get("queries", [])
        
        if self.verbose:
            print(f"[AdvancedEval] Suite: {suite_info.get('name', 'Unknown')}")
            print(f"[AdvancedEval] Queries: {len(queries)}")
        
        query_type = suite_info.get("query_type", "unknown")
        per_query_results = []
        
        for query in queries:
            result = self._evaluate_query(query, query_type)
            per_query_results.append(result)
        
        # Aggregate results
        aggregated = self._aggregate_results(per_query_results, query_type)
        
        return {
            "suite_path": suite_path,
            "suite_name": suite_info.get("name", Path(suite_path).stem),
            "query_type": query_type,
            "output_type": suite_info.get("output_type", "unknown"),
            "count": len(queries),
            "per_query": per_query_results,
            "aggregated": aggregated,
            "trace_rubric_status": "placeholder_not_implemented",
            "timestamp": datetime.now().isoformat()
        }
    
    def _evaluate_query(self, query: Dict[str, Any],
                        query_type: str) -> Dict[str, Any]:
        """
        Evaluate a single advanced query.
        
        Args:
            query: Query dict with input and gold_output
            query_type: Type of query (chain_reasoning, etc.)
            
        Returns:
            Per-query result dictionary
        """
        query_id = query.get("query_id", "unknown")
        input_params = query.get("input", {})
        gold_output = query.get("gold_output", {})
        
        result = {
            "query_id": query_id,
            "description": query.get("description", ""),
            "executed": False,
            "success": False,
            "metrics": {}
        }
        
        if self.engine is None:
            result["error"] = "Engine not loaded"
            return result
        
        try:
            # Execute query based on type
            if query_type == "chain_reasoning":
                engine_result = self._execute_chain_reasoning(input_params)
            elif query_type == "pattern_matching":
                engine_result = self._execute_pattern_matching(input_params)
            elif query_type == "scene_comparison":
                engine_result = self._execute_scene_comparison(input_params)
            elif query_type == "counterfactual":
                engine_result = self._execute_counterfactual(input_params)
            else:
                result["error"] = f"Unknown query type: {query_type}"
                return result
            
            result["executed"] = True
            result["success"] = True
            result["predicted"] = engine_result
            
            # Compute metrics
            metrics = self._compute_metrics(engine_result, gold_output, query_type)
            result["metrics"] = metrics
            
            # Placeholder trace quality rubric
            trace_expected = query.get("reasoning_trace_expected", {})
            result["trace_rubric"] = TraceQualityRubric(is_placeholder=True).to_dict()
            
        except Exception as e:
            result["error"] = str(e)
            result["executed"] = True
            result["success"] = False
        
        return result
    
    def _execute_chain_reasoning(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a chain reasoning query."""
        start_concept = params.get("start_concept", "")
        chain = params.get("chain", [])
        limit = params.get("limit", 10)
        
        result = self.engine.chain_reasoning(
            start_concept=start_concept,
            chain=chain,
            limit=limit
        )
        
        # Extract structured output
        return {
            "exists": len(result.results) > 0 if isinstance(result.results, list) else bool(result.results),
            "paths": result.results if isinstance(result.results, list) else [],
            "count": len(result.results) if isinstance(result.results, list) else (1 if result.results else 0),
            "reasoning_steps": [str(step) for step in result.reasoning_steps]
        }
    
    def _execute_pattern_matching(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a pattern matching query."""
        pattern_nodes = params.get("pattern_nodes", [])
        pattern_edges = params.get("pattern_edges", [])
        limit = params.get("limit", 10)
        
        result = self.engine.pattern_matching(
            pattern_nodes=pattern_nodes,
            pattern_edges=pattern_edges,
            limit=limit
        )
        
        return {
            "matches": result.results if isinstance(result.results, list) else [],
            "count": len(result.results) if isinstance(result.results, list) else 0,
            "reasoning_steps": [str(step) for step in result.reasoning_steps]
        }
    
    def _execute_scene_comparison(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a scene comparison query."""
        comparison_type = params.get("comparison_type", "structural")
        
        if comparison_type == "find_similar":
            ref_image = params.get("reference_image_id", "")
            min_sim = params.get("min_similarity", 0.1)
            limit = params.get("limit", 5)
            
            # Use find_similar_scenes if available
            if hasattr(self.engine, 'find_similar_scenes'):
                result = self.engine.find_similar_scenes(
                    reference_image=ref_image,
                    min_similarity=min_sim,
                    limit=limit
                )
            else:
                return {"error": "find_similar_scenes not implemented"}
        else:
            image_a = params.get("image_id_a", "")
            image_b = params.get("image_id_b", "")
            
            result = self.engine.scene_comparison(
                image_id_a=image_a,
                image_id_b=image_b
            )
        
        if hasattr(result, 'results'):
            return result.results if isinstance(result.results, dict) else {"data": result.results}
        return {"error": "Unexpected result format"}
    
    def _execute_counterfactual(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a counterfactual query."""
        image_id = params.get("image_id", "")
        modification = params.get("modification", {})
        
        result = self.engine.counterfactual_reasoning(
            image_id=image_id,
            modification=modification
        )
        
        if hasattr(result, 'results'):
            return result.results if isinstance(result.results, dict) else {"data": result.results}
        return {"error": "Unexpected result format"}
    
    def _compute_metrics(self, predicted: Dict[str, Any],
                         gold: Dict[str, Any],
                         query_type: str) -> Dict[str, float]:
        """
        Compute metrics for advanced query.
        
        Uses set-based metrics where applicable.
        """
        metrics = {}
        
        if query_type == "chain_reasoning":
            # Path existence accuracy
            pred_exists = predicted.get("exists", False)
            gold_exists = gold.get("exists", False)
            metrics["existence_accuracy"] = 1.0 if pred_exists == gold_exists else 0.0
            
            # Path count accuracy (within tolerance)
            pred_count = predicted.get("count", 0)
            gold_count = gold.get("count", 0)
            metrics["count_error"] = abs(pred_count - gold_count)
            metrics["count_exact"] = 1.0 if pred_count == gold_count else 0.0
            
        elif query_type == "pattern_matching":
            # Match count
            pred_count = predicted.get("count", 0)
            gold_count = gold.get("count", 0)
            metrics["count_error"] = abs(pred_count - gold_count)
            metrics["count_exact"] = 1.0 if pred_count == gold_count else 0.0
            
            # If we have detailed matches, compute overlap
            pred_matches = set(str(m) for m in predicted.get("matches", []))
            gold_matches = set(str(m) for m in gold.get("matches", []))
            
            if gold_matches:
                metrics["match_recall"] = len(pred_matches & gold_matches) / len(gold_matches)
            else:
                metrics["match_recall"] = 1.0 if not pred_matches else 0.0
                
        elif query_type == "scene_comparison":
            # Common concepts overlap
            pred_common = set(predicted.get("common_concepts", []))
            gold_common = set(gold.get("common_concepts", []))
            
            if gold_common:
                metrics["common_concepts_f1"] = self._f1_score(pred_common, gold_common)
            else:
                metrics["common_concepts_f1"] = 1.0 if not pred_common else 0.0
            
            # Similarity score error
            pred_sim = predicted.get("similarity_score", 0)
            gold_sim = gold.get("similarity_score", 0)
            metrics["similarity_error"] = abs(pred_sim - gold_sim)
            
        elif query_type == "counterfactual":
            # Counterfactual evaluation is largely subjective
            # We check for structural consistency
            metrics["executed"] = 1.0 if "error" not in predicted else 0.0
            
            # Check if modification was applied
            if "modified_scene" in predicted:
                metrics["modification_applied"] = 1.0
            else:
                metrics["modification_applied"] = 0.0
        
        return metrics
    
    def _f1_score(self, pred: set, gold: set) -> float:
        """Compute F1 score for sets."""
        if not pred and not gold:
            return 1.0
        if not pred or not gold:
            return 0.0
        
        precision = len(pred & gold) / len(pred)
        recall = len(pred & gold) / len(gold)
        
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)
    
    def _aggregate_results(self, results: List[Dict[str, Any]],
                           query_type: str) -> Dict[str, float]:
        """Aggregate per-query results."""
        if not results:
            return {"count": 0, "success_rate": 0.0}
        
        n = len(results)
        success_count = sum(1 for r in results if r.get("success", False))
        
        aggregated = {
            "count": n,
            "success_rate": success_count / n,
            "executed_rate": sum(1 for r in results if r.get("executed", False)) / n
        }
        
        # Aggregate metrics
        all_metrics = [r.get("metrics", {}) for r in results if r.get("metrics")]
        
        if all_metrics:
            # Get all metric keys
            metric_keys = set()
            for m in all_metrics:
                metric_keys.update(m.keys())
            
            # Average each metric
            for key in metric_keys:
                values = [m.get(key, 0) for m in all_metrics if key in m]
                if values:
                    aggregated[f"mean_{key}"] = sum(values) / len(values)
        
        return aggregated


def save_advanced_results(results: Dict[str, Any], output_dir: str,
                          suite_name: str) -> Dict[str, str]:
    """Save advanced evaluation results."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    base_name = f"advanced_{suite_name}"
    paths = {}
    
    # Save JSON
    json_path = output_path / f"{base_name}.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, default=str)
    paths["json"] = str(json_path)
    
    # Save CSV (per-query)
    csv_path = output_path / f"{base_name}.csv"
    per_query = results.get("per_query", [])
    if per_query:
        fieldnames = ["query_id", "description", "executed", "success"]
        # Add metric fields
        if per_query[0].get("metrics"):
            fieldnames.extend(per_query[0]["metrics"].keys())
        fieldnames.append("error")
        
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            for row in per_query:
                flat_row = {
                    "query_id": row["query_id"],
                    "description": row.get("description", ""),
                    "executed": row.get("executed", False),
                    "success": row.get("success", False),
                    "error": row.get("error", "")
                }
                flat_row.update(row.get("metrics", {}))
                writer.writerow(flat_row)
    paths["csv"] = str(csv_path)
    
    # Save Markdown summary
    md_path = output_path / f"{base_name}.md"
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(f"# Advanced Query Evaluation: {suite_name}\n\n")
        f.write(f"**Timestamp:** {results.get('timestamp', 'N/A')}\n\n")
        f.write(f"**Query Type:** {results.get('query_type', 'N/A')}\n\n")
        f.write(f"**Query Count:** {results.get('count', 0)}\n\n")
        
        f.write("## Aggregated Metrics\n\n")
        aggregated = results.get("aggregated", {})
        f.write("| Metric | Value |\n")
        f.write("|--------|-------|\n")
        for key, value in sorted(aggregated.items()):
            if isinstance(value, float):
                f.write(f"| {key} | {value:.4f} |\n")
            else:
                f.write(f"| {key} | {value} |\n")
        
        f.write("\n## Trace Quality Rubric\n\n")
        f.write(f"**Status:** {results.get('trace_rubric_status', 'not implemented')}\n\n")
        f.write("Note: Trace quality assessment is marked as **future work**. ")
        f.write("Implementation would require:\n")
        f.write("- Manual annotation of expected trace elements\n")
        f.write("- Automated or human scoring against rubric criteria:\n")
        f.write("  - Completeness: mentions key steps\n")
        f.write("  - Correctness: valid intermediate reasoning\n")
        f.write("  - Clarity: structured trace fields\n")
        
    paths["md"] = str(md_path)
    
    return paths


def evaluate_all_advanced_suites(config: Dict[str, Any],
                                  results_dir: str,
                                  verbose: bool = False) -> Dict[str, Any]:
    """
    Evaluate all advanced query suites.
    
    Args:
        config: Configuration dictionary
        results_dir: Directory to save results
        verbose: Whether to print progress
        
    Returns:
        Dictionary with all suite results
    """
    # Get suite paths from config
    advanced_config = config.get("advanced_queries", {})
    
    if not advanced_config.get("enabled", False):
        print("[AdvancedEval] Advanced queries disabled in config")
        return {"status": "disabled"}
    
    evaluator = AdvancedQueryEvaluator(config, verbose=verbose)
    all_results = {}
    
    types_config = advanced_config.get("types", {})
    
    for query_type, type_config in types_config.items():
        if not type_config.get("enabled", False):
            continue
        
        suite_path = type_config.get("suite_path", "")
        full_path = REPO_ROOT / suite_path
        
        if not full_path.exists():
            print(f"[AdvancedEval] Suite not found: {full_path}")
            continue
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"[AdvancedEval] Evaluating: {query_type}")
            print(f"{'='*60}")
        
        results = evaluator.evaluate_suite(str(full_path))
        save_advanced_results(results, results_dir, query_type)
        
        all_results[query_type] = {
            "count": results.get("count", 0),
            "aggregated": results.get("aggregated", {}),
            "trace_rubric_status": results.get("trace_rubric_status", "")
        }
    
    return all_results


def main():
    """Main entry point for advanced query evaluation."""
    import argparse
    import yaml
    
    parser = argparse.ArgumentParser(
        description="Evaluate advanced query types (6-9)"
    )
    parser.add_argument(
        "--config",
        default="evaluation_v0_1/configs/eval.yaml",
        help="Path to eval.yaml configuration file"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print detailed progress"
    )
    
    args = parser.parse_args()
    
    # Load config
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path
    
    if not config_path.exists():
        print(f"[ERROR] Config file not found: {config_path}")
        sys.exit(1)
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    paths = config.get("paths", {})
    results_dir = REPO_ROOT / paths.get("results_dir", "evaluation_v0_1/results")
    
    print("\n" + "=" * 60)
    print("Advanced Query Evaluation (Types 6-9)")
    print("=" * 60)
    
    results = evaluate_all_advanced_suites(config, str(results_dir), args.verbose)
    
    print("\n[AdvancedEval] Evaluation complete.")
    print(f"[AdvancedEval] Results saved to: {results_dir}")


if __name__ == "__main__":
    main()
