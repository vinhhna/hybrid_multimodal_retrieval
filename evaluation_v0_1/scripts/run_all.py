"""
Run All Evaluation Script for Evaluation Framework v0.1

Master script that runs the complete evaluation pipeline:
1. Generate query suites from scene graphs
2. Evaluate engine with GOLD CQR
3. Evaluate parsers (NL -> CQR)
4. Evaluate end-to-end (NL -> Parser -> Engine)
5. Evaluate baselines (naive_scan, relation_walk, parser_trivial)
6. Evaluate advanced queries (Types 6-9)
7. Generate summary reports

Supports:
- Data splits (train/val/test) - Task A
- Set-based metrics for unranked outputs - Task B
- Baseline methods for comparison - Task C
- Advanced query evaluation - Task D
- Timestamped outputs for reproducibility

Usage:
    python run_all.py --config evaluation_v0_1/configs/eval.yaml
    python run_all.py --config evaluation_v0_1/configs/eval.yaml --verbose
    python run_all.py --config evaluation_v0_1/configs/eval.yaml --split val
    python run_all.py --config evaluation_v0_1/configs/eval.yaml --method baseline_naive_scan
    python run_all.py --config evaluation_v0_1/configs/eval.yaml --step advanced
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List

# Add repo root to path
REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    import yaml

    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def run_generate_suites(config: Dict[str, Any], verbose: bool = False,
                        force_regen: bool = False) -> Dict[str, int]:
    """
    Run suite generation step.

    Args:
        config: Configuration dictionary
        verbose: Whether to print progress
        force_regen: Force regeneration even if suites exist

    Returns:
        Dictionary of suite_name -> count
    """
    # Import from the package
    from evaluation_v0_1.scripts import generate_suites as gen_module

    paths = config.get("paths", {})
    suites_dir = REPO_ROOT / paths.get("suites_dir", "evaluation_v0_1/data/suites")

    # Check if suites already exist
    suite_files = list(suites_dir.glob("suite_*.jsonl")) if suites_dir.exists() else []

    if suite_files and not force_regen:
        if verbose:
            print(f"\n[Run All] Found {len(suite_files)} existing suites. Skipping generation.")
            print("[Run All] Use --regen to force regeneration.")
        return {"skipped": len(suite_files)}

    if verbose:
        print("\n" + "="*60)
        print("[Run All] Step 1: Generating Query Suites")
        print("="*60)

    config_path = REPO_ROOT / "evaluation_v0_1" / "configs" / "eval.yaml"
    return gen_module.generate_suites(str(config_path))


def run_evaluate_engine(config: Dict[str, Any], verbose: bool = False) -> Dict[str, Any]:
    """
    Run engine-only evaluation step.

    Args:
        config: Configuration dictionary
        verbose: Whether to print progress

    Returns:
        Evaluation results
    """
    from evaluation_v0_1.scripts import evaluate_engine as eval_eng_module

    if verbose:
        print("\n" + "="*60)
        print("[Run All] Step 2: Engine-Only Evaluation (GOLD CQR)")
        print("="*60)

    paths = config.get("paths", {})
    suites_dir = REPO_ROOT / paths.get("suites_dir", "evaluation_v0_1/data/suites")
    results_dir = REPO_ROOT / paths.get("results_dir", "evaluation_v0_1/results")

    return eval_eng_module.evaluate_all_suites(config, str(suites_dir), str(results_dir), verbose=verbose)


def run_evaluate_parser(config: Dict[str, Any], verbose: bool = False) -> Dict[str, Any]:
    """
    Run parser evaluation step.

    Args:
        config: Configuration dictionary
        verbose: Whether to print progress

    Returns:
        Parser evaluation results
    """
    from evaluation_v0_1.scripts import evaluate_parser as eval_parser_module

    if verbose:
        print("\n" + "="*60)
        print("[Run All] Step 3: Parser Evaluation (NL -> CQR)")
        print("="*60)

    paths = config.get("paths", {})
    suites_dir = REPO_ROOT / paths.get("suites_dir", "evaluation_v0_1/data/suites")
    results_dir = REPO_ROOT / paths.get("results_dir", "evaluation_v0_1/results")

    return eval_parser_module.evaluate_all_suites(config, str(suites_dir), str(results_dir), verbose=verbose)


def run_evaluate_e2e(config: Dict[str, Any], verbose: bool = False) -> Dict[str, Any]:
    """
    Run end-to-end evaluation step.

    Args:
        config: Configuration dictionary
        verbose: Whether to print progress

    Returns:
        E2E evaluation results
    """
    from evaluation_v0_1.scripts import evaluate_e2e as eval_e2e_module

    if verbose:
        print("\n" + "="*60)
        print("[Run All] Step 4: End-to-End Evaluation (NL -> Parser -> Engine)")
        print("="*60)

    paths = config.get("paths", {})
    suites_dir = REPO_ROOT / paths.get("suites_dir", "evaluation_v0_1/data/suites")
    results_dir = REPO_ROOT / paths.get("results_dir", "evaluation_v0_1/results")

    return eval_e2e_module.evaluate_all_suites(config, str(suites_dir), str(results_dir), verbose=verbose)


def run_evaluate_baselines(config: Dict[str, Any], method: str,
                           verbose: bool = False) -> Dict[str, Any]:
    """
    Run baseline evaluation step.
    
    Args:
        config: Configuration dictionary
        method: Baseline method name (baseline_naive_scan, baseline_relation_walk)
        verbose: Whether to print progress
        
    Returns:
        Baseline evaluation results
    """
    from evaluation_v0_1.scripts.baselines import (
        BaselineAdapter, load_scene_graphs_for_baseline
    )
    from evaluation_v0_1.scripts.evaluate_engine import EngineEvaluator, save_results
    from evaluation_v0_1.scripts.cqr import load_suite
    
    if verbose:
        print("\n" + "="*60)
        print(f"[Run All] Baseline Evaluation: {method}")
        print("="*60)
    
    paths = config.get("paths", {})
    suites_dir = REPO_ROOT / paths.get("suites_dir", "evaluation_v0_1/data/suites")
    results_dir = REPO_ROOT / paths.get("results_dir", "evaluation_v0_1/results")
    
    # Determine scene graphs path based on split
    split = config.get("data_splits", {}).get("active_split", "train")
    if split == "val":
        sg_path = paths.get("scenegraphs_val", "sceneGraphs/val_sceneGraphs.json")
    else:
        sg_path = paths.get("scenegraphs_train", "sceneGraphs/train_sceneGraphs.json")
    
    full_sg_path = REPO_ROOT / sg_path
    
    if not full_sg_path.exists():
        print(f"[WARN] Scene graphs not found: {full_sg_path}")
        return {"error": f"Scene graphs not found: {sg_path}"}
    
    # Load scene graphs
    scene_graphs = load_scene_graphs_for_baseline(str(full_sg_path))
    
    # Create baseline adapter
    adapter = BaselineAdapter(
        method=method,
        scene_graphs=scene_graphs,
        config=config,
        verbose=verbose
    )
    
    # Find appropriate suites for this baseline
    suite_files = list(Path(suites_dir).glob("suite_*.jsonl"))
    
    all_results = {}
    
    for suite_file in suite_files:
        suite_name = suite_file.stem
        
        # Skip suites incompatible with this baseline
        if method == "baseline_naive_scan":
            # Naive scan works on ranked_set queries
            if "scalar" in suite_name or "path" in suite_name or "subgraph" in suite_name:
                continue
        elif method == "baseline_relation_walk":
            # Relation walk works on path queries
            if "path" not in suite_name:
                continue
        
        if verbose:
            print(f"\n[Baseline] Evaluating {suite_name} with {method}")
        
        # Load suite and evaluate
        suite = load_suite(str(suite_file))
        
        if not suite:
            continue
        
        # Simple evaluation loop
        from evaluation_v0_1.scripts.metrics import (
            compute_set_metrics, aggregate_set_metrics,
            compute_path_metrics, aggregate_path_metrics
        )
        
        per_query_results = []
        k_values = config.get("evaluation", {}).get("k_values", [1, 5, 10, 20, 50])
        
        for item in suite:
            result = adapter.execute(item.cqr_gold)
            
            query_result = {
                "query_id": item.query_id,
                "success": result.success,
                "metrics": {}
            }
            
            if result.success:
                if "path" in suite_name:
                    metrics = compute_path_metrics(result.data, item.gold_output.data)
                else:
                    predicted = result.data.get("image_ids", [])
                    gold = set(item.gold_output.data.get("image_ids", []))
                    metrics = compute_set_metrics(predicted, gold, k_values)
                
                query_result["metrics"] = metrics
            
            per_query_results.append(query_result)
        
        # Aggregate
        valid_metrics = [r["metrics"] for r in per_query_results if r.get("success") and r.get("metrics")]
        
        if valid_metrics:
            if "path" in suite_name:
                aggregated = aggregate_path_metrics(valid_metrics)
            else:
                aggregated = aggregate_set_metrics(valid_metrics, k_values)
        else:
            aggregated = {"count": 0}
        
        aggregated["method"] = method
        aggregated["success_rate"] = len(valid_metrics) / len(per_query_results) if per_query_results else 0
        
        # Save results
        baseline_results = {
            "suite_name": suite_name,
            "method": method,
            "count": len(suite),
            "aggregated": aggregated,
            "timestamp": datetime.now().isoformat()
        }
        
        result_file = results_dir / f"baseline_{method}_{suite_name}.json"
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(baseline_results, f, indent=2, default=str)
        
        all_results[suite_name] = aggregated
    
    return all_results


def run_evaluate_advanced(config: Dict[str, Any], verbose: bool = False) -> Dict[str, Any]:
    """
    Run advanced query evaluation step.
    
    Args:
        config: Configuration dictionary
        verbose: Whether to print progress
        
    Returns:
        Advanced evaluation results
    """
    from evaluation_v0_1.scripts.evaluate_advanced import evaluate_all_advanced_suites
    
    if verbose:
        print("\n" + "="*60)
        print("[Run All] Step 6: Advanced Query Evaluation (Types 6-9)")
        print("="*60)
    
    paths = config.get("paths", {})
    results_dir = REPO_ROOT / paths.get("results_dir", "evaluation_v0_1/results")
    
    return evaluate_all_advanced_suites(config, str(results_dir), verbose=verbose)


def get_output_dir(config: Dict[str, Any], split: str, method: str) -> Path:
    """
    Get output directory with optional timestamping.
    
    Args:
        config: Configuration dictionary
        split: Data split (train/val/test)
        method: Method name (main/baseline_*)
        
    Returns:
        Path to output directory
    """
    paths = config.get("paths", {})
    reproducibility = config.get("reproducibility", {})
    
    base_dir = REPO_ROOT / paths.get("artifacts_dir", "artifacts")
    
    if reproducibility.get("timestamped_outputs", False):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pattern = reproducibility.get("output_pattern", "{timestamp}_{split}_{method}")
        dir_name = pattern.format(timestamp=timestamp, split=split, method=method)
        return base_dir / dir_name
    else:
        return REPO_ROOT / paths.get("results_dir", "evaluation_v0_1/results")


def generate_summary(config: Dict[str, Any],
                    suite_results: Dict[str, int],
                    engine_results: Dict[str, Any],
                    parser_results: Dict[str, Any],
                    e2e_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate the overall summary.

    Returns:
        Summary dictionary
    """
    paths = config.get("paths", {})

    summary = {
        "timestamp": datetime.now().isoformat(),
        "configuration": {
            "kg_graph_path": paths.get("kg_graph_path", ""),
            "scenegraphs_val": paths.get("scenegraphs_val", ""),
            "k_values": config.get("evaluation", {}).get("k_values", []),
            "random_seed": config.get("random_seed", 1337)
        },
        "suite_sizes": suite_results,
        "engine_evaluation": {},
        "parser_evaluation": {
            "heuristic": {},
            "llm_stub": {}
        },
        "e2e_evaluation": {
            "heuristic": {},
            "llm_stub": {}
        }
    }

    # Engine results by output type
    for suite_name, data in engine_results.items():
        if isinstance(data, dict) and "aggregated" in data:
            output_type = data.get("output_type", "")
            summary["engine_evaluation"][suite_name] = {
                "output_type": output_type,
                "count": data.get("count", 0),
                "metrics": data.get("aggregated", {})
            }

    # Parser results
    for suite_name, data in parser_results.items():
        if suite_name in ("timestamp", "total_queries"):
            continue
        if isinstance(data, dict):
            h_agg = data.get("heuristic", {})
            l_agg = data.get("llm_stub", {})
            summary["parser_evaluation"]["heuristic"][suite_name] = h_agg
            summary["parser_evaluation"]["llm_stub"][suite_name] = l_agg

    # E2E results
    for suite_name, data in e2e_results.items():
        if suite_name in ("timestamp", "total_queries"):
            continue
        if isinstance(data, dict):
            h_agg = data.get("heuristic", {})
            l_agg = data.get("llm_stub", {})
            summary["e2e_evaluation"]["heuristic"][suite_name] = h_agg
            summary["e2e_evaluation"]["llm_stub"][suite_name] = l_agg

    return summary


def save_summary(summary: Dict[str, Any], output_dir: str) -> None:
    """Save summary to files."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # JSON summary
    json_path = output_path / "summary.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, default=str)

    # Markdown summary
    md_path = output_path / "summary.md"
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write("# Evaluation Framework v0.1 - Summary Report\n\n")
        f.write(f"**Generated:** {summary['timestamp']}\n\n")

        # Configuration
        f.write("## Configuration\n\n")
        config = summary.get("configuration", {})
        f.write(f"- **KG Graph Path:** `{config.get('kg_graph_path', 'N/A')}`\n")
        f.write(f"- **Scene Graphs:** `{config.get('scenegraphs_val', 'N/A')}`\n")
        f.write(f"- **k values:** {config.get('k_values', [])}\n")
        f.write(f"- **Random Seed:** {config.get('random_seed', 'N/A')}\n\n")

        # Suite sizes
        f.write("## Query Suites\n\n")
        f.write("| Suite | Count |\n")
        f.write("|-------|-------|\n")
        for suite_name, count in summary.get("suite_sizes", {}).items():
            f.write(f"| {suite_name} | {count} |\n")
        f.write("\n")

        # Engine evaluation
        f.write("## Engine Evaluation (GOLD CQR)\n\n")
        for suite_name, data in summary.get("engine_evaluation", {}).items():
            f.write(f"### {suite_name}\n\n")
            f.write(f"**Output Type:** {data.get('output_type', 'N/A')}\n\n")

            metrics = data.get("metrics", {})
            if metrics:
                f.write("| Metric | Value |\n")
                f.write("|--------|-------|\n")
                for key, value in sorted(metrics.items()):
                    if isinstance(value, float):
                        f.write(f"| {key} | {value:.4f} |\n")
                    else:
                        f.write(f"| {key} | {value} |\n")
                f.write("\n")

        # Parser evaluation
        f.write("## Parser Evaluation (NL -> CQR)\n\n")
        f.write("### Heuristic Parser\n\n")
        h_parser = summary.get("parser_evaluation", {}).get("heuristic", {})
        if h_parser:
            f.write("| Suite | Exact Match | Concept F1 | Attr F1 |\n")
            f.write("|-------|-------------|------------|----------|\n")
            for suite_name, metrics in h_parser.items():
                em = metrics.get("exact_match", 0)
                cf1 = metrics.get("concept_f1", 0)
                af1 = metrics.get("attr_f1", 0)
                f.write(f"| {suite_name} | {em:.4f} | {cf1:.4f} | {af1:.4f} |\n")
            f.write("\n")

        f.write("### LLM Stub Parser\n\n")
        l_parser = summary.get("parser_evaluation", {}).get("llm_stub", {})
        if l_parser:
            f.write("| Suite | Exact Match | Concept F1 | Attr F1 |\n")
            f.write("|-------|-------------|------------|----------|\n")
            for suite_name, metrics in l_parser.items():
                em = metrics.get("exact_match", 0)
                cf1 = metrics.get("concept_f1", 0)
                af1 = metrics.get("attr_f1", 0)
                f.write(f"| {suite_name} | {em:.4f} | {cf1:.4f} | {af1:.4f} |\n")
            f.write("\n")

        # E2E evaluation
        f.write("## End-to-End Evaluation\n\n")
        f.write("### Heuristic Parser Pipeline\n\n")
        h_e2e = summary.get("e2e_evaluation", {}).get("heuristic", {})
        if h_e2e:
            f.write("| Suite | Success Rate | MRR/MAE/Validity |\n")
            f.write("|-------|--------------|------------------|\n")
            for suite_name, metrics in h_e2e.items():
                sr = metrics.get("success_rate", 0)
                # Pick appropriate primary metric based on output type
                primary = metrics.get("mrr", metrics.get("mae", metrics.get("validity_rate", 0)))
                f.write(f"| {suite_name} | {sr:.4f} | {primary:.4f} |\n")
            f.write("\n")

        f.write("### LLM Stub Parser Pipeline\n\n")
        l_e2e = summary.get("e2e_evaluation", {}).get("llm_stub", {})
        if l_e2e:
            f.write("| Suite | Success Rate | MRR/MAE/Validity |\n")
            f.write("|-------|--------------|------------------|\n")
            for suite_name, metrics in l_e2e.items():
                sr = metrics.get("success_rate", 0)
                primary = metrics.get("mrr", metrics.get("mae", metrics.get("validity_rate", 0)))
                f.write(f"| {suite_name} | {sr:.4f} | {primary:.4f} |\n")
            f.write("\n")

        f.write("---\n\n")
        f.write("*Report generated by Evaluation Framework v0.1*\n")

    print(f"\n[Run All] Summary saved to:")
    print(f"  - {json_path}")
    print(f"  - {md_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Run complete evaluation pipeline for LightRAG-GQA"
    )
    parser.add_argument(
        "--config",
        default="evaluation_v0_1/configs/eval.yaml",
        help="Path to eval.yaml configuration file"
    )
    parser.add_argument(
        "--regen",
        action="store_true",
        help="Force regeneration of query suites"
    )
    parser.add_argument(
        "--step",
        choices=["generate", "engine", "parser", "e2e", "baseline", "advanced", "all"],
        default="all",
        help="Run specific step only (default: all)"
    )
    parser.add_argument(
        "--split",
        choices=["train", "val", "test", "train+val"],
        default=None,
        help="Data split to evaluate on (overrides config)"
    )
    parser.add_argument(
        "--method",
        choices=["main", "baseline_naive_scan", "baseline_relation_walk", "baseline_parser_trivial"],
        default=None,
        help="Method to evaluate (overrides config)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print detailed progress"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (overrides config)"
    )

    args = parser.parse_args()

    # Load config
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path

    if not config_path.exists():
        print(f"[ERROR] Config file not found: {config_path}")
        sys.exit(1)

    config = load_config(str(config_path))
    
    # Apply command-line overrides
    if args.split:
        if "data_splits" not in config:
            config["data_splits"] = {}
        config["data_splits"]["active_split"] = args.split
    
    if args.method:
        if "method" not in config:
            config["method"] = {}
        config["method"]["active"] = args.method
    
    if args.seed is not None:
        config["random_seed"] = args.seed
        if "reproducibility" not in config:
            config["reproducibility"] = {}
        config["reproducibility"]["random_seed"] = args.seed
    
    # Set random seed for reproducibility
    import random
    seed = config.get("reproducibility", {}).get("random_seed", config.get("random_seed", 1337))
    random.seed(seed)
    
    # Determine active split and method
    active_split = config.get("data_splits", {}).get("active_split", "train")
    active_method = config.get("method", {}).get("active", "main")

    print("\n" + "="*60)
    print("LightRAG-GQA Evaluation Framework v0.1")
    print("="*60)
    print(f"\nConfig: {config_path}")
    print(f"Split: {active_split}")
    print(f"Method: {active_method}")
    print(f"Seed: {seed}")
    print(f"Verbose: {args.verbose}")
    print(f"Step: {args.step}")

    paths = config.get("paths", {})
    
    # Get output directory (with optional timestamping)
    results_dir = get_output_dir(config, active_split, active_method)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Results: {results_dir}")

    # Initialize results
    suite_results = {}
    engine_results = {}
    parser_results = {}
    e2e_results = {}
    baseline_results = {}
    advanced_results = {}

    try:
        # Step 1: Generate suites
        if args.step in ("generate", "all"):
            suite_results = run_generate_suites(config, args.verbose, args.regen)

        # Step 2: Engine evaluation (main method)
        if args.step in ("engine", "all") and active_method == "main":
            engine_results = run_evaluate_engine(config, args.verbose)

        # Step 3: Parser evaluation
        if args.step in ("parser", "all"):
            parser_results = run_evaluate_parser(config, args.verbose)

        # Step 4: E2E evaluation
        if args.step in ("e2e", "all"):
            e2e_results = run_evaluate_e2e(config, args.verbose)
        
        # Step 5: Baseline evaluation
        if args.step in ("baseline", "all") or active_method.startswith("baseline_"):
            if active_method.startswith("baseline_"):
                baseline_results = run_evaluate_baselines(config, active_method, args.verbose)
            elif args.step == "baseline":
                # Run all enabled baselines
                baseline_config = config.get("method", {}).get("baselines", {})
                for baseline_name, bl_config in baseline_config.items():
                    if bl_config.get("enabled", False):
                        method_name = f"baseline_{baseline_name}"
                        if args.verbose:
                            print(f"\n[Run All] Running baseline: {method_name}")
                        baseline_results[method_name] = run_evaluate_baselines(
                            config, method_name, args.verbose
                        )
        
        # Step 6: Advanced query evaluation
        if args.step in ("advanced", "all"):
            advanced_results = run_evaluate_advanced(config, args.verbose)

        # Generate summary if running all or multiple steps
        if args.step == "all":
            print("\n" + "="*60)
            print("[Run All] Generating Summary Report")
            print("="*60)

            summary = generate_summary(
                config, suite_results, engine_results, parser_results, e2e_results
            )
            
            # Add baseline and advanced results to summary
            summary["baseline_evaluation"] = baseline_results
            summary["advanced_evaluation"] = advanced_results
            summary["data_split"] = active_split
            summary["method"] = active_method
            summary["random_seed"] = seed
            
            save_summary(summary, str(results_dir))

        print("\n" + "="*60)
        print("[Run All] Evaluation Complete!")
        print("="*60)
        print(f"\nResults saved to: {results_dir}")
        
        # Print quick summary
        if engine_results:
            print("\nEngine Evaluation Summary:")
            for suite_name, data in engine_results.items():
                if isinstance(data, dict) and "aggregated" in data:
                    agg = data["aggregated"]
                    print(f"  {suite_name}: F1={agg.get('f1', agg.get('node_f1', 'N/A')):.3f}")
        
        if baseline_results:
            print("\nBaseline Evaluation Summary:")
            for method, results in baseline_results.items():
                if isinstance(results, dict):
                    for suite, agg in results.items():
                        if isinstance(agg, dict):
                            print(f"  {method}/{suite}: F1={agg.get('f1', 'N/A')}")
        
        if advanced_results and advanced_results.get("status") != "disabled":
            print("\nAdvanced Query Evaluation Summary:")
            for qtype, data in advanced_results.items():
                if isinstance(data, dict) and "aggregated" in data:
                    agg = data["aggregated"]
                    print(f"  {qtype}: success_rate={agg.get('success_rate', 0):.3f}")

    except Exception as e:
        print(f"\n[ERROR] Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

