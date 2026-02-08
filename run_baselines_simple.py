"""
Simple Baseline Evaluation Script

Runs the three baselines (NaiveScan, RelationWalk, TrivialParser) on compatible
query types and saves results for comparison.
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any
from datetime import datetime

# Add repo root to path
REPO_ROOT = Path(__file__).parent
sys.path.insert(0, str(REPO_ROOT))

from evaluation_v0_1.scripts.baselines import (
    NaiveScanBaseline, RelationWalkBaseline, TrivialParserBaseline,
    load_scene_graphs_for_baseline
)
from evaluation_v0_1.scripts.cqr import load_suite
from evaluation_v0_1.scripts.metrics import compute_set_metrics, aggregate_set_metrics


def run_naive_scan_baseline():
    """Run NaiveScanBaseline on ranked_set queries."""
    print("\n" + "="*70)
    print("BASELINE 1: NAIVE SCAN (Direct Scene Graph Scan)")
    print("="*70)
    
    # Load scene graphs
    sg_path = REPO_ROOT / "sceneGraphs" / "train_sceneGraphs.json"
    print(f"\nLoading scene graphs from: {sg_path}")
    
    scene_graphs = load_scene_graphs_for_baseline(str(sg_path), max_images=1000)
    print(f"✓ Loaded {len(scene_graphs)} scene graphs")
    
    # Initialize baseline
    baseline = NaiveScanBaseline(scene_graphs, verbose=True)
    
    # Load ranked_entity_attr suite (this is a ranked_set query type)
    suite_path = REPO_ROOT / "evaluation_v0_1" / "data" / "suites" / "suite_ranked_entity_attr.jsonl"
    
    if not suite_path.exists():
        print(f"❌ Suite not found: {suite_path}")
        return {}
    
    suite = load_suite(str(suite_path))
    print(f"✓ Loaded suite with {len(suite)} queries")
    
    # Run baseline on first 100 queries (for speed)
    print(f"\nRunning NaiveScan on first 100 queries...")
    k_values = [1, 5, 10, 20, 50]
    per_query_results = []
    
    for i, item in enumerate(suite[:100]):
        if i % 20 == 0:
            print(f"  Processed {i}/100 queries...")
        
        result = baseline.execute(item.cqr_gold)
        
        if result.success:
            predicted = result.data.get("image_ids", [])
            gold = set(item.gold_output.data.get("image_ids", []))
            
            metrics = compute_set_metrics(predicted, gold, k_values)
            per_query_results.append(metrics)
    
    print(f"✓ Completed {len(per_query_results)} queries successfully")
    
    # Aggregate metrics
    aggregated = aggregate_set_metrics(per_query_results, k_values)
    aggregated["method"] = "naive_scan"
    aggregated["success_rate"] = len(per_query_results) / 100
    aggregated["timestamp"] = datetime.now().isoformat()
    
    return aggregated


def run_relation_walk_baseline():
    """Run RelationWalkBaseline on path queries."""
    print("\n" + "="*70)
    print("BASELINE 2: RELATION WALK (Within-Image BFS)")
    print("="*70)
    
    # Load scene graphs
    sg_path = REPO_ROOT / "sceneGraphs" / "train_sceneGraphs.json"
    print(f"\nLoading scene graphs from: {sg_path}")
    
    scene_graphs = load_scene_graphs_for_baseline(str(sg_path), max_images=1000)
    print(f"✓ Loaded {len(scene_graphs)} scene graphs")
    
    # Initialize baseline
    baseline = RelationWalkBaseline(scene_graphs, max_hops=3, verbose=True)
    
    # Load path suite
    suite_path = REPO_ROOT / "evaluation_v0_1" / "data" / "suites" / "suite_path.jsonl"
    
    if not suite_path.exists():
        print(f"❌ Suite not found: {suite_path}")
        return {}
    
    suite = load_suite(str(suite_path))
    print(f"✓ Loaded suite with {len(suite)} queries")
    
    # Run baseline on first 50 queries (path queries are slower)
    print(f"\nRunning RelationWalk on first 50 queries...")
    per_query_results = []
    
    for i, item in enumerate(suite[:50]):
        if i % 10 == 0:
            print(f"  Processed {i}/50 queries...")
        
        result = baseline.execute(item.cqr_gold)
        
        if result.success:
            metrics = {
                "exists": result.data.get("exists", False),
                "path_count": result.data.get("total_paths_found", 0),
                "shortest_hops": result.data.get("shortest_hops", None)
            }
            per_query_results.append(metrics)
    
    print(f"✓ Completed {len(per_query_results)} queries successfully")
    
    # Aggregate metrics
    exists_count = sum(1 for m in per_query_results if m["exists"])
    total_paths = sum(m["path_count"] for m in per_query_results)
    avg_hops = sum(m["shortest_hops"] for m in per_query_results if m["shortest_hops"] is not None)
    avg_hops = avg_hops / len(per_query_results) if per_query_results else 0
    
    aggregated = {
        "method": "relation_walk",
        "success_rate": len(per_query_results) / 50,
        "path_exists_rate": exists_count / len(per_query_results) if per_query_results else 0,
        "total_paths_found": total_paths,
        "avg_shortest_hops": avg_hops,
        "timestamp": datetime.now().isoformat()
    }
    
    return aggregated


def run_trivial_parser_baseline():
    """Run TrivialParserBaseline on NL queries."""
    print("\n" + "="*70)
    print("BASELINE 3: TRIVIAL PARSER (Simple Regex)")
    print("="*70)
    
    # Initialize baseline
    baseline = TrivialParserBaseline(verbose=True)
    
    # Create test queries (we'd need NL queries with gold CQR for proper evaluation)
    test_queries = [
        {"nl": "find red cars", "expected_concept": "cars"},
        {"nl": "show images with a dog", "expected_concept": "dog"},
        {"nl": "get blue chairs", "expected_concept": "chairs"},
        {"nl": "find large tables", "expected_concept": "tables"},
        {"nl": "how many cats", "expected_concept": "cats"},
    ]
    
    print(f"\nTesting on {len(test_queries)} sample queries...")
    
    results = []
    for query_data in test_queries:
        nl = query_data["nl"]
        expected = query_data["expected_concept"]
        
        parsed = baseline.parse(nl)
        
        success = parsed.get("success", False)
        actual = parsed.get("params", {}).get("concept", "")
        
        match = (actual == expected) if success else False
        
        results.append({
            "nl": nl,
            "success": success,
            "expected": expected,
            "actual": actual,
            "match": match
        })
        
        status = "✓" if match else "❌"
        print(f"  {status} '{nl}' -> {actual} (expected: {expected})")
    
    # Aggregate
    parse_success = sum(1 for r in results if r["success"]) / len(results)
    concept_accuracy = sum(1 for r in results if r["match"]) / len(results)
    
    aggregated = {
        "method": "trivial_parser",
        "test_queries": len(results),
        "parse_success_rate": parse_success,
        "concept_accuracy": concept_accuracy,
        "timestamp": datetime.now().isoformat()
    }
    
    return aggregated


def main():
    print("\n" + "="*70)
    print("LIGHTRAG-GQA BASELINE EVALUATION (Simplified)")
    print("="*70)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    all_results = {}
    
    try:
        # Run Baseline 1: Naive Scan
        naive_results = run_naive_scan_baseline()
        all_results["naive_scan"] = naive_results
        
        # Run Baseline 2: Relation Walk
        relation_results = run_relation_walk_baseline()
        all_results["relation_walk"] = relation_results
        
        # Run Baseline 3: Trivial Parser
        parser_results = run_trivial_parser_baseline()
        all_results["trivial_parser"] = parser_results
        
    except Exception as e:
        print(f"\n❌ Error during baseline evaluation: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Save results
    output_dir = REPO_ROOT / "evaluation_v0_1" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / "baseline_results_summary.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    print("\n" + "="*70)
    print("BASELINE EVALUATION COMPLETE")
    print("="*70)
    print(f"\nResults saved to: {output_file}")
    
    # Print summary
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    
    if "naive_scan" in all_results:
        ns = all_results["naive_scan"]
        print(f"\n📊 Naive Scan Baseline:")
        print(f"   Success Rate: {ns.get('success_rate', 0):.2%}")
        print(f"   Precision@10: {ns.get('precision@10', 0):.3f}")
        print(f"   Recall@50: {ns.get('recall@50', 0):.3f}")
        print(f"   F1: {ns.get('f1', 0):.3f}")
    
    if "relation_walk" in all_results:
        rw = all_results["relation_walk"]
        print(f"\n📊 Relation Walk Baseline:")
        print(f"   Success Rate: {rw.get('success_rate', 0):.2%}")
        print(f"   Path Exists Rate: {rw.get('path_exists_rate', 0):.2%}")
        print(f"   Avg Shortest Hops: {rw.get('avg_shortest_hops', 0):.2f}")
    
    if "trivial_parser" in all_results:
        tp = all_results["trivial_parser"]
        print(f"\n📊 Trivial Parser Baseline:")
        print(f"   Parse Success Rate: {tp.get('parse_success_rate', 0):.2%}")
        print(f"   Concept Accuracy: {tp.get('concept_accuracy', 0):.2%}")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    main()
