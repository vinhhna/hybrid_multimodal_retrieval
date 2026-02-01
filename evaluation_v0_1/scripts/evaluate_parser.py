"""
Parser Evaluation Script for Evaluation Framework v0.1

Evaluates NL parsers (heuristic and LLM stub) on their ability to
produce correct CQR from natural language queries.

This is Parser-only evaluation (NL -> CQR accuracy against GOLD CQR).

Usage:
    python evaluate_parser.py --config evaluation_v0_1/configs/eval.yaml
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

from .cqr import CQR, QuerySuiteItem, load_suite, OutputType
from .llm_parser_stub import LLMParserStub, HeuristicParserWrapper
from .metrics import compute_parser_metrics, aggregate_parser_metrics


class ParserEvaluator:
    """
    Evaluator for NL parser comparison.
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

        # Initialize parsers
        self.heuristic_parser = HeuristicParserWrapper(verbose=verbose)
        self.llm_parser = LLMParserStub(use_heuristic_fallback=False, verbose=verbose)

    def evaluate_suite(self, suite_path: str) -> Dict[str, Any]:
        """
        Evaluate both parsers on a query suite.

        Args:
            suite_path: Path to the JSONL suite file

        Returns:
            Evaluation results with per-query and aggregate metrics
        """
        if self.verbose:
            print(f"\n[Parser Eval] Loading suite: {suite_path}")

        suite = load_suite(suite_path)

        if not suite:
            return {"error": "Empty suite", "count": 0}

        if self.verbose:
            print(f"[Parser Eval] Loaded {len(suite)} queries")

        heuristic_results = []
        llm_results = []

        for i, item in enumerate(suite):
            if self.verbose and (i + 1) % 100 == 0:
                print(f"[Parser Eval] Progress: {i+1}/{len(suite)}")

            # Evaluate on each NL template
            for nl_idx, nl_template in enumerate(item.nl_templates):
                query_id = f"{item.query_id}_nl{nl_idx}"
                gold_cqr_dict = item.cqr_gold.to_dict()

                # Heuristic parser
                try:
                    heuristic_cqr = self.heuristic_parser.parse(nl_template, query_id)
                    h_metrics = compute_parser_metrics(heuristic_cqr.to_dict(), gold_cqr_dict)
                    h_metrics["query_id"] = query_id
                    h_metrics["nl_template"] = nl_template
                    h_metrics["success"] = True
                    heuristic_results.append(h_metrics)
                except Exception as e:
                    heuristic_results.append({
                        "query_id": query_id,
                        "nl_template": nl_template,
                        "success": False,
                        "error": str(e)
                    })

                # LLM parser stub
                try:
                    llm_cqr = self.llm_parser.parse(nl_template, query_id)
                    l_metrics = compute_parser_metrics(llm_cqr.to_dict(), gold_cqr_dict)
                    l_metrics["query_id"] = query_id
                    l_metrics["nl_template"] = nl_template
                    l_metrics["success"] = True
                    llm_results.append(l_metrics)
                except Exception as e:
                    llm_results.append({
                        "query_id": query_id,
                        "nl_template": nl_template,
                        "success": False,
                        "error": str(e)
                    })

        # Aggregate
        valid_heuristic = [r for r in heuristic_results if r.get("success")]
        valid_llm = [r for r in llm_results if r.get("success")]

        return {
            "suite_path": suite_path,
            "suite_name": Path(suite_path).stem,
            "count": len(suite),
            "total_nl_templates": len(heuristic_results),
            "heuristic": {
                "per_query": heuristic_results,
                "aggregated": aggregate_parser_metrics(valid_heuristic)
            },
            "llm_stub": {
                "per_query": llm_results,
                "aggregated": aggregate_parser_metrics(valid_llm)
            },
            "timestamp": datetime.now().isoformat()
        }


def save_parser_results(results: Dict[str, Any], output_dir: str) -> Dict[str, str]:
    """
    Save parser evaluation results to files.

    Args:
        results: Evaluation results
        output_dir: Directory to save files

    Returns:
        Dictionary of output file paths
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    suite_name = results.get("suite_name", "parser")
    paths = {}

    # Save full JSON
    json_path = output_path / f"parser_{suite_name}.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        # Exclude per_query for compactness
        summary = {
            "suite_name": results.get("suite_name"),
            "count": results.get("count"),
            "total_nl_templates": results.get("total_nl_templates"),
            "heuristic_aggregated": results.get("heuristic", {}).get("aggregated", {}),
            "llm_stub_aggregated": results.get("llm_stub", {}).get("aggregated", {}),
            "timestamp": results.get("timestamp")
        }
        json.dump(summary, f, indent=2)
    paths["json"] = str(json_path)

    # Save heuristic CSV
    h_csv_path = output_path / f"parser_{suite_name}_heuristic.csv"
    h_results = results.get("heuristic", {}).get("per_query", [])
    if h_results:
        fieldnames = list(h_results[0].keys())
        with open(h_csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(h_results)
    paths["heuristic_csv"] = str(h_csv_path)

    # Save LLM CSV
    l_csv_path = output_path / f"parser_{suite_name}_llm.csv"
    l_results = results.get("llm_stub", {}).get("per_query", [])
    if l_results:
        fieldnames = list(l_results[0].keys())
        with open(l_csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(l_results)
    paths["llm_csv"] = str(l_csv_path)

    return paths


def save_parser_summary(all_results: Dict[str, Any], output_dir: str) -> None:
    """
    Save overall parser comparison summary.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # JSON summary
    json_path = output_path / "parser_summary.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2)

    # CSV summary
    csv_path = output_path / "parser_report.csv"
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            "Suite", "Parser", "Count", "OutputType Acc", "Op Acc",
            "Concept F1", "Attr F1", "Rel F1", "Negation F1", "Exact Match"
        ])

        for suite_name, data in all_results.items():
            if suite_name in ("timestamp", "total_queries"):
                continue

            h_agg = data.get("heuristic", {})
            l_agg = data.get("llm_stub", {})

            writer.writerow([
                suite_name, "heuristic", h_agg.get("count", 0),
                f"{h_agg.get('output_type_acc', 0):.4f}",
                f"{h_agg.get('op_acc', 0):.4f}",
                f"{h_agg.get('concept_f1', 0):.4f}",
                f"{h_agg.get('attr_f1', 0):.4f}",
                f"{h_agg.get('rel_f1', 0):.4f}",
                f"{h_agg.get('negation_f1', 0):.4f}",
                f"{h_agg.get('exact_match', 0):.4f}"
            ])

            writer.writerow([
                suite_name, "llm_stub", l_agg.get("count", 0),
                f"{l_agg.get('output_type_acc', 0):.4f}",
                f"{l_agg.get('op_acc', 0):.4f}",
                f"{l_agg.get('concept_f1', 0):.4f}",
                f"{l_agg.get('attr_f1', 0):.4f}",
                f"{l_agg.get('rel_f1', 0):.4f}",
                f"{l_agg.get('negation_f1', 0):.4f}",
                f"{l_agg.get('exact_match', 0):.4f}"
            ])

    # Markdown summary
    md_path = output_path / "parser_summary.md"
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write("# Parser Evaluation Summary\n\n")
        f.write(f"**Timestamp:** {all_results.get('timestamp', 'N/A')}\n\n")
        f.write(f"**Total Queries Evaluated:** {all_results.get('total_queries', 0)}\n\n")

        f.write("## Comparison by Suite\n\n")

        for suite_name, data in all_results.items():
            if suite_name in ("timestamp", "total_queries"):
                continue

            f.write(f"### {suite_name}\n\n")

            h_agg = data.get("heuristic", {})
            l_agg = data.get("llm_stub", {})

            f.write("| Metric | Heuristic | LLM Stub |\n")
            f.write("|--------|-----------|----------|\n")

            metrics = ["output_type_acc", "op_acc", "concept_f1", "attr_f1",
                      "rel_f1", "negation_f1", "exact_match"]

            for m in metrics:
                h_val = h_agg.get(m, 0)
                l_val = l_agg.get(m, 0)
                f.write(f"| {m} | {h_val:.4f} | {l_val:.4f} |\n")

            f.write("\n")


def evaluate_all_suites(config: Dict[str, Any], suites_dir: str,
                       results_dir: str, verbose: bool = False) -> Dict[str, Any]:
    """
    Evaluate parsers on all suites.

    Args:
        config: Configuration dictionary
        suites_dir: Directory containing suite JSONL files
        results_dir: Directory to save results
        verbose: Whether to print progress

    Returns:
        Dictionary with all suite results
    """
    evaluator = ParserEvaluator(config, verbose=verbose)

    # Find all suites
    suites_path = Path(suites_dir)
    suite_files = list(suites_path.glob("suite_*.jsonl"))

    if not suite_files:
        print(f"[WARN] No suites found in {suites_dir}")
        return {"error": "No suites found"}

    if verbose:
        print(f"\n[Parser Eval] Found {len(suite_files)} suites")

    all_results = {
        "timestamp": datetime.now().isoformat(),
        "total_queries": 0
    }

    for suite_file in suite_files:
        suite_name = suite_file.stem
        if verbose:
            print(f"\n{'='*60}")
            print(f"[Parser Eval] Evaluating: {suite_name}")
            print(f"{'='*60}")

        results = evaluator.evaluate_suite(str(suite_file))
        save_parser_results(results, results_dir)

        all_results[suite_name] = {
            "count": results.get("count", 0),
            "heuristic": results.get("heuristic", {}).get("aggregated", {}),
            "llm_stub": results.get("llm_stub", {}).get("aggregated", {})
        }
        all_results["total_queries"] += results.get("count", 0)

        if verbose:
            print(f"\n[Parser Eval] {suite_name} complete:")
            h_agg = results.get("heuristic", {}).get("aggregated", {})
            l_agg = results.get("llm_stub", {}).get("aggregated", {})
            print(f"  Heuristic exact match: {h_agg.get('exact_match', 0):.4f}")
            print(f"  LLM stub exact match: {l_agg.get('exact_match', 0):.4f}")

    # Save overall summary
    save_parser_summary(all_results, results_dir)

    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate NL parsers (NL -> CQR)"
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

        evaluator = ParserEvaluator(config, verbose=args.verbose)
        results = evaluator.evaluate_suite(str(suite_path))
        save_parser_results(results, str(results_dir))

        print(f"\n[Parser Eval] Evaluation complete!")
    else:
        # Evaluate all suites
        all_results = evaluate_all_suites(
            config, str(suites_dir), str(results_dir),
            verbose=args.verbose
        )

        print(f"\n[Parser Eval] All evaluations complete!")
        print(f"[Parser Eval] Results saved to {results_dir}")


if __name__ == "__main__":
    main()

