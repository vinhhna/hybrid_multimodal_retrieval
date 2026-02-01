"""
End-to-End Evaluation Script for Evaluation Framework v0.1

Evaluates the full pipeline: NL -> Parsed CQR -> Engine -> Outputs
Compares heuristic parser vs LLM stub parser on final retrieval quality.

This is Layer B (End-to-end) evaluation.

Usage:
    python evaluate_e2e.py --config evaluation_v0_1/configs/eval.yaml
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
from .adapter_engine import EngineAdapter, create_adapter
from .llm_parser_stub import LLMParserStub, HeuristicParserWrapper
from .metrics import (
    compute_ranked_metrics, aggregate_ranked_metrics,
    compute_scalar_metrics, aggregate_scalar_metrics,
    compute_path_metrics, aggregate_path_metrics,
    compute_subgraph_metrics, aggregate_subgraph_metrics
)


class E2EEvaluator:
    """
    End-to-end evaluator: NL -> Parser -> Engine -> Metrics against Gold.
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

        # Initialize parsers
        self.heuristic_parser = HeuristicParserWrapper(verbose=False)
        self.llm_parser = LLMParserStub(use_heuristic_fallback=False, verbose=False)

        # Get k values for ranking metrics
        eval_config = config.get("evaluation", {})
        self.k_values = eval_config.get("k_values", [1, 5, 10, 20, 50])

    def evaluate_suite(self, suite_path: str) -> Dict[str, Any]:
        """
        Evaluate end-to-end on a query suite.

        Args:
            suite_path: Path to the JSONL suite file

        Returns:
            Evaluation results comparing parsers
        """
        if self.verbose:
            print(f"\n[E2E Eval] Loading suite: {suite_path}")

        suite = load_suite(suite_path)

        if not suite:
            return {"error": "Empty suite", "count": 0}

        if self.verbose:
            print(f"[E2E Eval] Loaded {len(suite)} queries")

        output_type = suite[0].cqr_gold.output_type

        heuristic_results = []
        llm_results = []

        for i, item in enumerate(suite):
            if self.verbose and (i + 1) % 50 == 0:
                print(f"[E2E Eval] Progress: {i+1}/{len(suite)}")

            gold = item.gold_output

            # Use first NL template for evaluation
            nl_query = item.nl_templates[0] if item.nl_templates else ""

            # Heuristic parser path
            h_result = self._evaluate_with_parser(
                self.heuristic_parser, nl_query, item.query_id, gold, output_type
            )
            heuristic_results.append(h_result)

            # LLM parser path
            l_result = self._evaluate_with_parser(
                self.llm_parser, nl_query, item.query_id, gold, output_type
            )
            llm_results.append(l_result)

        # Aggregate results
        heuristic_agg = self._aggregate_results(heuristic_results, output_type)
        llm_agg = self._aggregate_results(llm_results, output_type)

        return {
            "suite_path": suite_path,
            "suite_name": Path(suite_path).stem,
            "output_type": output_type,
            "count": len(suite),
            "heuristic": {
                "per_query": heuristic_results,
                "aggregated": heuristic_agg
            },
            "llm_stub": {
                "per_query": llm_results,
                "aggregated": llm_agg
            },
            "timestamp": datetime.now().isoformat()
        }

    def _evaluate_with_parser(self, parser, nl_query: str, query_id: str,
                             gold: GoldOutput, output_type: str) -> Dict[str, Any]:
        """
        Run full pipeline with a specific parser.
        """
        result = {
            "query_id": query_id,
            "nl_query": nl_query,
            "success": False
        }

        try:
            # Parse NL to CQR
            parsed_cqr = parser.parse(nl_query, query_id)

            # Execute on engine
            engine_result = self.adapter.execute(parsed_cqr)

            if not engine_result.success:
                result["error"] = engine_result.error
                return result

            result["success"] = True

            # Compute metrics against GOLD output
            if output_type == OutputType.RANKED_SET.value:
                predicted = engine_result.data.get("image_ids", [])
                gold_ids = set(gold.data.get("image_ids", []))
                metrics = compute_ranked_metrics(predicted, gold_ids, self.k_values)
                result["metrics"] = metrics
                result["predicted_count"] = len(predicted)
                result["gold_count"] = len(gold_ids)

            elif output_type == OutputType.SCALAR.value:
                predicted = engine_result.data.get("value", 0.0)
                gold_value = gold.data.get("value", 0.0)
                metrics = compute_scalar_metrics(predicted, gold_value)
                result["metrics"] = metrics

            elif output_type == OutputType.PATH.value:
                metrics = compute_path_metrics(engine_result.data, gold.data)
                result["metrics"] = metrics

            elif output_type == OutputType.SUBGRAPH.value:
                metrics = compute_subgraph_metrics(engine_result.data, gold.data)
                result["metrics"] = metrics

        except Exception as e:
            result["error"] = str(e)

        return result

    def _aggregate_results(self, results: List[Dict[str, Any]],
                          output_type: str) -> Dict[str, float]:
        """
        Aggregate per-query results.
        """
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


def save_e2e_results(results: Dict[str, Any], output_dir: str) -> Dict[str, str]:
    """
    Save E2E evaluation results.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    suite_name = results.get("suite_name", "e2e")
    paths = {}

    # Save summary JSON
    json_path = output_path / f"e2e_{suite_name}.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        summary = {
            "suite_name": results.get("suite_name"),
            "output_type": results.get("output_type"),
            "count": results.get("count"),
            "heuristic_aggregated": results.get("heuristic", {}).get("aggregated", {}),
            "llm_stub_aggregated": results.get("llm_stub", {}).get("aggregated", {}),
            "timestamp": results.get("timestamp")
        }
        json.dump(summary, f, indent=2)
    paths["json"] = str(json_path)

    # Save per-query CSVs
    for parser_name in ["heuristic", "llm_stub"]:
        csv_path = output_path / f"e2e_{suite_name}_{parser_name}.csv"
        parser_results = results.get(parser_name, {}).get("per_query", [])

        if parser_results:
            # Build fieldnames
            fieldnames = ["query_id", "nl_query", "success", "error"]
            if parser_results[0].get("metrics"):
                fieldnames.extend(parser_results[0]["metrics"].keys())

            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
                writer.writeheader()
                for row in parser_results:
                    flat_row = {
                        "query_id": row["query_id"],
                        "nl_query": row.get("nl_query", "")[:100],  # Truncate
                        "success": row["success"],
                        "error": row.get("error", "")
                    }
                    flat_row.update(row.get("metrics", {}))
                    writer.writerow(flat_row)

        paths[f"{parser_name}_csv"] = str(csv_path)

    return paths


def save_e2e_summary(all_results: Dict[str, Any], output_dir: str) -> None:
    """
    Save overall E2E comparison summary.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # JSON summary
    json_path = output_path / "e2e_summary.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2)

    # Markdown summary
    md_path = output_path / "e2e_summary.md"
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write("# End-to-End Evaluation Summary\n\n")
        f.write(f"**Timestamp:** {all_results.get('timestamp', 'N/A')}\n\n")
        f.write(f"**Total Queries:** {all_results.get('total_queries', 0)}\n\n")

        f.write("## Comparison: Heuristic vs LLM Stub\n\n")

        for suite_name, data in all_results.items():
            if suite_name in ("timestamp", "total_queries"):
                continue

            f.write(f"### {suite_name}\n\n")
            f.write(f"**Output Type:** {data.get('output_type', 'N/A')}\n\n")

            h_agg = data.get("heuristic", {})
            l_agg = data.get("llm_stub", {})

            # Determine which metrics to show based on output type
            output_type = data.get("output_type", "")

            if output_type == OutputType.RANKED_SET.value:
                metrics = ["mrr", "map", "precision@10", "recall@10", "ndcg@10", "success_rate"]
            elif output_type == OutputType.SCALAR.value:
                metrics = ["mae", "rmse", "spearman", "success_rate"]
            elif output_type == OutputType.PATH.value:
                metrics = ["validity_rate", "hop_accuracy", "coverage", "success_rate"]
            elif output_type == OutputType.SUBGRAPH.value:
                metrics = ["node_recall", "edge_recall", "node_f1", "edge_f1", "success_rate"]
            else:
                metrics = ["success_rate"]

            f.write("| Metric | Heuristic | LLM Stub |\n")
            f.write("|--------|-----------|----------|\n")

            for m in metrics:
                h_val = h_agg.get(m, 0)
                l_val = l_agg.get(m, 0)
                if isinstance(h_val, float):
                    f.write(f"| {m} | {h_val:.4f} | {l_val:.4f} |\n")
                else:
                    f.write(f"| {m} | {h_val} | {l_val} |\n")

            f.write("\n")


def evaluate_all_suites(config: Dict[str, Any], suites_dir: str,
                       results_dir: str, verbose: bool = False) -> Dict[str, Any]:
    """
    Run E2E evaluation on all suites.
    """
    # Create adapter
    adapter = create_adapter(config, verbose=verbose)

    # Create evaluator
    evaluator = E2EEvaluator(adapter, config, verbose=verbose)

    # Find all suites
    suites_path = Path(suites_dir)
    suite_files = list(suites_path.glob("suite_*.jsonl"))

    if not suite_files:
        print(f"[WARN] No suites found in {suites_dir}")
        return {"error": "No suites found"}

    if verbose:
        print(f"\n[E2E Eval] Found {len(suite_files)} suites")

    all_results = {
        "timestamp": datetime.now().isoformat(),
        "total_queries": 0
    }

    for suite_file in suite_files:
        suite_name = suite_file.stem
        if verbose:
            print(f"\n{'='*60}")
            print(f"[E2E Eval] Evaluating: {suite_name}")
            print(f"{'='*60}")

        results = evaluator.evaluate_suite(str(suite_file))
        save_e2e_results(results, results_dir)

        all_results[suite_name] = {
            "output_type": results.get("output_type"),
            "count": results.get("count", 0),
            "heuristic": results.get("heuristic", {}).get("aggregated", {}),
            "llm_stub": results.get("llm_stub", {}).get("aggregated", {})
        }
        all_results["total_queries"] += results.get("count", 0)

        if verbose:
            print(f"\n[E2E Eval] {suite_name} complete:")
            h_agg = results.get("heuristic", {}).get("aggregated", {})
            l_agg = results.get("llm_stub", {}).get("aggregated", {})
            print(f"  Heuristic success rate: {h_agg.get('success_rate', 0):.4f}")
            print(f"  LLM stub success rate: {l_agg.get('success_rate', 0):.4f}")

    # Save overall summary
    save_e2e_summary(all_results, results_dir)

    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="End-to-end evaluation: NL -> Parser -> Engine"
    )
    parser.add_argument(
        "--config",
        default="evaluation_v0_1/configs/eval.yaml",
        help="Path to eval.yaml"
    )
    parser.add_argument(
        "--suite",
        default=None,
        help="Specific suite to evaluate"
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
        evaluator = E2EEvaluator(adapter, config, verbose=args.verbose)

        results = evaluator.evaluate_suite(str(suite_path))
        save_e2e_results(results, str(results_dir))

        print(f"\n[E2E Eval] Evaluation complete!")
    else:
        # Evaluate all suites
        all_results = evaluate_all_suites(
            config, str(suites_dir), str(results_dir),
            verbose=args.verbose
        )

        print(f"\n[E2E Eval] All evaluations complete!")
        print(f"[E2E Eval] Results saved to {results_dir}")


if __name__ == "__main__":
    main()

