"""
Engine Evaluation Script for Evaluation Framework v0.1

Evaluates the query engine using GOLD CQR (structured queries from scene graphs).
This is Layer A (Engine-only) evaluation - no NL parsing involved.

Usage:
    python evaluate_engine.py --config evaluation_v0_1/configs/eval.yaml
    python evaluate_engine.py --config evaluation_v0_1/configs/eval.yaml --suite ranked_entity_attr
"""

import sys
import json
import csv
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
from collections import defaultdict

# Add repo root to path
REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from .cqr import CQR, GoldOutput, QuerySuiteItem, load_suite, OutputType
from .adapter_engine import EngineAdapter, EngineResult, create_adapter
from .metrics import (
    compute_ranked_metrics, aggregate_ranked_metrics,
    compute_scalar_metrics, aggregate_scalar_metrics,
    compute_path_metrics, aggregate_path_metrics,
    compute_subgraph_metrics, aggregate_subgraph_metrics
)


class EngineEvaluator:
    """
    Evaluator for engine-only evaluation using GOLD CQR.
    """

    def __init__(self, adapter: EngineAdapter, config: Dict[str, Any],
                 verbose: bool = False):
        """
        Initialize the evaluator.

        Args:
            adapter: EngineAdapter instance
            config: Configuration dictionary
            verbose: Whether to print progress
        """
        self.adapter = adapter
        self.config = config
        self.verbose = verbose

        # Get k values for ranking metrics
        eval_config = config.get("evaluation", {})
        self.k_values = eval_config.get("k_values", [1, 5, 10, 20, 50])

    def evaluate_suite(self, suite_path: str) -> Dict[str, Any]:
        """
        Evaluate the engine on a query suite.

        Args:
            suite_path: Path to the JSONL suite file

        Returns:
            Evaluation results with per-query and aggregate metrics
        """
        if self.verbose:
            print(f"\n[Evaluator] Loading suite: {suite_path}")

        suite = load_suite(suite_path)

        if not suite:
            return {"error": "Empty suite", "count": 0}

        if self.verbose:
            print(f"[Evaluator] Loaded {len(suite)} queries")

        # Determine output type from first item
        output_type = suite[0].cqr_gold.output_type

        per_query_results = []

        for i, item in enumerate(suite):
            if self.verbose and (i + 1) % 50 == 0:
                print(f"[Evaluator] Progress: {i+1}/{len(suite)}")

            result = self._evaluate_query(item)
            per_query_results.append(result)

        # Aggregate metrics based on output type
        aggregated = self._aggregate_results(per_query_results, output_type)

        return {
            "suite_path": suite_path,
            "suite_name": Path(suite_path).stem,
            "output_type": output_type,
            "count": len(suite),
            "per_query": per_query_results,
            "aggregated": aggregated,
            "timestamp": datetime.now().isoformat()
        }

    def _evaluate_query(self, item: QuerySuiteItem) -> Dict[str, Any]:
        """
        Evaluate a single query.

        Args:
            item: QuerySuiteItem with gold CQR and output

        Returns:
            Per-query result dictionary
        """
        cqr = item.cqr_gold
        gold = item.gold_output

        # Execute query on engine
        engine_result = self.adapter.execute(cqr)

        result = {
            "query_id": item.query_id,
            "success": engine_result.success,
            "output_type": cqr.output_type,
            "error": engine_result.error
        }

        if not engine_result.success:
            result["metrics"] = {}
            return result

        # Compute metrics based on output type
        if cqr.output_type == OutputType.RANKED_SET.value:
            predicted = engine_result.data.get("image_ids", [])
            gold_ids = set(gold.data.get("image_ids", []))
            metrics = compute_ranked_metrics(predicted, gold_ids, self.k_values)
            result["metrics"] = metrics
            result["predicted_count"] = len(predicted)
            result["gold_count"] = len(gold_ids)

        elif cqr.output_type == OutputType.SCALAR.value:
            predicted = engine_result.data.get("value", 0.0)
            gold_value = gold.data.get("value", 0.0)
            metrics = compute_scalar_metrics(predicted, gold_value)
            result["metrics"] = metrics

        elif cqr.output_type == OutputType.PATH.value:
            metrics = compute_path_metrics(engine_result.data, gold.data)
            result["metrics"] = metrics

        elif cqr.output_type == OutputType.SUBGRAPH.value:
            metrics = compute_subgraph_metrics(engine_result.data, gold.data)
            result["metrics"] = metrics

        return result

    def _aggregate_results(self, results: List[Dict[str, Any]],
                          output_type: str) -> Dict[str, float]:
        """
        Aggregate per-query results.
        """
        # Filter successful results with metrics
        valid_results = [r["metrics"] for r in results
                        if r.get("success") and r.get("metrics")]

        if not valid_results:
            return {"count": 0, "success_rate": 0.0}

        success_rate = len(valid_results) / len(results) if results else 0.0

        if output_type == OutputType.RANKED_SET.value:
            aggregated = aggregate_ranked_metrics(valid_results, self.k_values)
        elif output_type == OutputType.SCALAR.value:
            aggregated = aggregate_scalar_metrics(valid_results)
        elif output_type == OutputType.PATH.value:
            aggregated = aggregate_path_metrics(valid_results)
        elif output_type == OutputType.SUBGRAPH.value:
            aggregated = aggregate_subgraph_metrics(valid_results)
        else:
            aggregated = {}

        aggregated["success_rate"] = success_rate
        return aggregated


def save_results(results: Dict[str, Any], output_dir: str,
                 suite_name: str) -> Dict[str, str]:
    """
    Save evaluation results to files.

    Args:
        results: Evaluation results
        output_dir: Directory to save files
        suite_name: Name of the suite

    Returns:
        Dictionary of output file paths
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    base_name = f"engine_{suite_name}"
    paths = {}

    # Save JSON
    json_path = output_path / f"{base_name}.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        # Remove per_query for the main JSON to keep it smaller
        summary = {k: v for k, v in results.items() if k != "per_query"}
        json.dump(summary, f, indent=2, default=str)
    paths["json"] = str(json_path)

    # Save per-query CSV
    csv_path = output_path / f"{base_name}.csv"
    per_query = results.get("per_query", [])
    if per_query:
        # Flatten metrics into columns
        fieldnames = ["query_id", "success", "error"]
        if per_query[0].get("metrics"):
            fieldnames.extend(per_query[0]["metrics"].keys())

        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            for row in per_query:
                flat_row = {
                    "query_id": row["query_id"],
                    "success": row["success"],
                    "error": row.get("error", "")
                }
                flat_row.update(row.get("metrics", {}))
                writer.writerow(flat_row)
    paths["csv"] = str(csv_path)

    # Save Markdown summary
    md_path = output_path / f"{base_name}.md"
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(f"# Engine Evaluation: {suite_name}\n\n")
        f.write(f"**Timestamp:** {results.get('timestamp', 'N/A')}\n\n")
        f.write(f"**Output Type:** {results.get('output_type', 'N/A')}\n\n")
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
    paths["md"] = str(md_path)

    return paths


def evaluate_all_suites(config: Dict[str, Any], suites_dir: str,
                       results_dir: str, verbose: bool = False) -> Dict[str, Any]:
    """
    Evaluate the engine on all suites.

    Args:
        config: Configuration dictionary
        suites_dir: Directory containing suite JSONL files
        results_dir: Directory to save results
        verbose: Whether to print progress

    Returns:
        Dictionary with all suite results
    """
    # Create adapter
    adapter = create_adapter(config, verbose=verbose)

    # Create evaluator
    evaluator = EngineEvaluator(adapter, config, verbose=verbose)

    # Find all suites
    suites_path = Path(suites_dir)
    suite_files = list(suites_path.glob("suite_*.jsonl"))

    if not suite_files:
        print(f"[WARN] No suites found in {suites_dir}")
        return {"error": "No suites found"}

    if verbose:
        print(f"\n[Engine Eval] Found {len(suite_files)} suites")

    all_results = {}

    for suite_file in suite_files:
        suite_name = suite_file.stem
        if verbose:
            print(f"\n{'='*60}")
            print(f"[Engine Eval] Evaluating: {suite_name}")
            print(f"{'='*60}")

        results = evaluator.evaluate_suite(str(suite_file))
        save_results(results, results_dir, suite_name)

        all_results[suite_name] = {
            "count": results.get("count", 0),
            "output_type": results.get("output_type", ""),
            "aggregated": results.get("aggregated", {})
        }

        if verbose:
            print(f"\n[Engine Eval] {suite_name} complete:")
            agg = results.get("aggregated", {})
            for key, value in sorted(agg.items()):
                if isinstance(value, float):
                    print(f"  {key}: {value:.4f}")
                else:
                    print(f"  {key}: {value}")

    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate query engine using GOLD CQR"
    )
    parser.add_argument(
        "--config",
        default="evaluation_v0_1/configs/eval.yaml",
        help="Path to eval.yaml"
    )
    parser.add_argument(
        "--suite",
        default=None,
        help="Specific suite to evaluate (without extension)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print progress"
    )

    args = parser.parse_args()

    import yaml

    # Load config
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    paths = config.get("paths", {})
    suites_dir = REPO_ROOT / paths.get("suites_dir", "evaluation_v0_1/data/suites")
    results_dir = REPO_ROOT / paths.get("results_dir", "evaluation_v0_1/results")

    if args.suite:
        # Evaluate single suite
        suite_path = suites_dir / f"{args.suite}.jsonl"
        if not suite_path.exists():
            suite_path = suites_dir / f"suite_{args.suite}.jsonl"

        if not suite_path.exists():
            print(f"[ERROR] Suite not found: {suite_path}")
            sys.exit(1)

        adapter = create_adapter(config, verbose=args.verbose)
        evaluator = EngineEvaluator(adapter, config, verbose=args.verbose)

        results = evaluator.evaluate_suite(str(suite_path))
        save_results(results, str(results_dir), suite_path.stem)

        print(f"\n[Engine Eval] Evaluation complete!")
        print(f"[Engine Eval] Results saved to {results_dir}")
    else:
        # Evaluate all suites
        all_results = evaluate_all_suites(
            config, str(suites_dir), str(results_dir),
            verbose=args.verbose
        )

        print(f"\n[Engine Eval] All evaluations complete!")
        print(f"[Engine Eval] Results saved to {results_dir}")


if __name__ == "__main__":
    main()

