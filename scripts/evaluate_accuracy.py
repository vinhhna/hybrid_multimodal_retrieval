"""
Accuracy Evaluation Script for T2.8

This script evaluates and compares the accuracy of different search methods:
- CLIP-only (baseline)
- Hybrid Search (CLIP + BLIP-2)
- Ground truth comparison

Metrics calculated:
- Recall@1, Recall@5, Recall@10
- Mean Reciprocal Rank (MRR)
- Mean Average Precision (MAP)
- Latency statistics
- Accuracy vs Latency trade-off

Usage (Kaggle):
    import sys
    sys.path.append('/kaggle/working/hybrid_multimodal_retrieval')
    %run scripts/evaluate_accuracy.py
"""

import sys
import os
import time
import json
from pathlib import Path
from typing import List, Tuple, Dict, Any, Set
import numpy as np
from collections import defaultdict

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Auto-detect environment
if Path('/kaggle/input').exists():
    DATA_DIR = Path('/kaggle/input/flickr30k/data')
    print("Running on Kaggle")
else:
    DATA_DIR = project_root / 'data'
    print("Running locally")

from src.retrieval.bi_encoder import BiEncoder
from src.retrieval.cross_encoder import CrossEncoder
from src.retrieval.faiss_index import FAISSIndex
from src.retrieval.hybrid_search import HybridSearchEngine
from src.flickr30k.dataset import Flickr30KDataset


def load_components():
    """Load all required components."""
    print("\n" + "="*70)
    print("LOADING COMPONENTS")
    print("="*70)
    
    start_time = time.time()
    
    print("\n[1/4] Loading Flickr30K dataset...")
    dataset = Flickr30KDataset(
        images_dir=str(DATA_DIR / 'images'),
        annotations_file=str(DATA_DIR / 'results.csv')
    )
    print(f"  ✓ Loaded {len(dataset)} images")
    
    print("\n[2/4] Loading CLIP bi-encoder...")
    bi_encoder = BiEncoder(model_name='ViT-B/32', device='cuda')
    print(f"  ✓ Model: {bi_encoder.model_name}")
    
    print("\n[3/4] Loading FAISS index...")
    image_index = FAISSIndex(device='cuda')
    index_path = DATA_DIR / 'indices' / 'image_index.faiss'
    image_index.load(str(index_path))
    print(f"  ✓ Loaded {image_index.index.ntotal:,} vectors")
    
    print("\n[4/4] Loading BLIP-2 cross-encoder...")
    cross_encoder = CrossEncoder(
        model_name='Salesforce/blip2-opt-2.7b',
        device='cuda',
        use_fp16=True
    )
    print(f"  ✓ Model: {cross_encoder.model_name}")
    
    load_time = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"Components loaded in {load_time:.2f}s")
    print(f"{'='*70}\n")
    
    return bi_encoder, cross_encoder, image_index, dataset


def select_test_queries(dataset: Flickr30KDataset, n: int = 100) -> List[Dict[str, Any]]:
    """
    Select diverse test queries from dataset.
    
    Each query includes:
    - query text (caption)
    - ground truth image_id
    - alternative captions for same image
    
    Args:
        dataset: Flickr30K dataset
        n: Number of test queries
    
    Returns:
        List of test query dictionaries
    """
    print("\n" + "="*70)
    print(f"SELECTING {n} TEST QUERIES")
    print("="*70)
    
    test_queries = []
    
    # Sample evenly across dataset
    step = len(dataset) // n
    
    for i in range(n):
        idx = i * step
        item = dataset[idx]
        
        if not item['captions']:
            continue
        
        # Use first caption as query, others as alternatives
        query_text = item['captions'][0]
        ground_truth_id = item['image_id']
        alternative_captions = item['captions'][1:] if len(item['captions']) > 1 else []
        
        test_queries.append({
            'query': query_text,
            'ground_truth': ground_truth_id,
            'alternatives': alternative_captions,
            'dataset_idx': idx
        })
    
    print(f"\n✓ Selected {len(test_queries)} test queries")
    print(f"  Sample queries:")
    for i, q in enumerate(test_queries[:3], 1):
        print(f"    {i}. {q['query'][:60]}...")
        print(f"       Ground truth: {q['ground_truth']}")
    
    print(f"{'='*70}")
    
    return test_queries


def evaluate_clip_only(
    engine: HybridSearchEngine,
    test_queries: List[Dict[str, Any]],
    k: int = 10
) -> Dict[str, Any]:
    """
    Evaluate CLIP-only search (baseline).
    
    Args:
        engine: HybridSearchEngine instance
        test_queries: List of test queries
        k: Number of results to retrieve
    
    Returns:
        Evaluation results
    """
    print("\n" + "="*70)
    print("EVALUATING CLIP-ONLY SEARCH (Baseline)")
    print("="*70)
    
    print(f"\nEvaluating {len(test_queries)} queries with k={k}")
    
    results = []
    latencies = []
    
    for i, test_query in enumerate(test_queries):
        if (i + 1) % 10 == 0:
            print(f"  Progress: {i+1}/{len(test_queries)} queries")
        
        query = test_query['query']
        ground_truth = test_query['ground_truth']
        
        # Run CLIP search
        start = time.time()
        search_results = engine._stage1_retrieve(query, k1=k)
        latency = (time.time() - start) * 1000
        latencies.append(latency)
        
        # Extract image IDs
        retrieved_ids = [img_id for img_id, score in search_results]
        
        # Check if ground truth in results
        ground_truth_rank = None
        if ground_truth in retrieved_ids:
            ground_truth_rank = retrieved_ids.index(ground_truth) + 1
        
        results.append({
            'query': query,
            'ground_truth': ground_truth,
            'retrieved': retrieved_ids,
            'rank': ground_truth_rank,
            'latency': latency
        })
    
    # Calculate metrics
    metrics = calculate_metrics(results, k)
    metrics['latencies'] = {
        'mean': np.mean(latencies),
        'median': np.median(latencies),
        'p95': np.percentile(latencies, 95),
        'p99': np.percentile(latencies, 99)
    }
    
    print(f"\n{'='*70}")
    print(f"CLIP-only Results:")
    print(f"  Recall@1:  {metrics['recall@1']:.2%}")
    print(f"  Recall@5:  {metrics['recall@5']:.2%}")
    print(f"  Recall@10: {metrics['recall@10']:.2%}")
    print(f"  MRR:       {metrics['mrr']:.4f}")
    print(f"  MAP:       {metrics['map']:.4f}")
    print(f"  Latency:   {metrics['latencies']['mean']:.2f}ms (avg)")
    print(f"{'='*70}")
    
    return {
        'method': 'CLIP-only',
        'results': results,
        'metrics': metrics
    }


def evaluate_hybrid(
    engine: HybridSearchEngine,
    test_queries: List[Dict[str, Any]],
    k1: int = 100,
    k2: int = 10
) -> Dict[str, Any]:
    """
    Evaluate Hybrid search (CLIP + BLIP-2).
    
    Args:
        engine: HybridSearchEngine instance
        test_queries: List of test queries
        k1: Number of Stage 1 candidates
        k2: Number of final results
    
    Returns:
        Evaluation results
    """
    print("\n" + "="*70)
    print("EVALUATING HYBRID SEARCH (CLIP + BLIP-2)")
    print("="*70)
    
    print(f"\nEvaluating {len(test_queries)} queries with k1={k1}, k2={k2}")
    
    results = []
    latencies = []
    
    for i, test_query in enumerate(test_queries):
        if (i + 1) % 10 == 0:
            print(f"  Progress: {i+1}/{len(test_queries)} queries")
        
        query = test_query['query']
        ground_truth = test_query['ground_truth']
        
        # Run hybrid search
        start = time.time()
        search_results = engine.text_to_image_hybrid_search(
            query=query,
            k1=k1,
            k2=k2,
            show_progress=False
        )
        latency = (time.time() - start) * 1000
        latencies.append(latency)
        
        # Extract image IDs
        retrieved_ids = [img_id for img_id, score in search_results]
        
        # Check if ground truth in results
        ground_truth_rank = None
        if ground_truth in retrieved_ids:
            ground_truth_rank = retrieved_ids.index(ground_truth) + 1
        
        results.append({
            'query': query,
            'ground_truth': ground_truth,
            'retrieved': retrieved_ids,
            'rank': ground_truth_rank,
            'latency': latency
        })
    
    # Calculate metrics
    metrics = calculate_metrics(results, k2)
    metrics['latencies'] = {
        'mean': np.mean(latencies),
        'median': np.median(latencies),
        'p95': np.percentile(latencies, 95),
        'p99': np.percentile(latencies, 99)
    }
    
    print(f"\n{'='*70}")
    print(f"Hybrid Results:")
    print(f"  Recall@1:  {metrics['recall@1']:.2%}")
    print(f"  Recall@5:  {metrics['recall@5']:.2%}")
    print(f"  Recall@10: {metrics['recall@10']:.2%}")
    print(f"  MRR:       {metrics['mrr']:.4f}")
    print(f"  MAP:       {metrics['map']:.4f}")
    print(f"  Latency:   {metrics['latencies']['mean']:.2f}ms (avg)")
    print(f"{'='*70}")
    
    return {
        'method': 'Hybrid',
        'results': results,
        'metrics': metrics
    }


def calculate_metrics(results: List[Dict], k: int) -> Dict[str, float]:
    """
    Calculate evaluation metrics.
    
    Metrics:
    - Recall@1, Recall@5, Recall@10
    - Mean Reciprocal Rank (MRR)
    - Mean Average Precision (MAP)
    
    Args:
        results: List of evaluation results
        k: Number of retrieved results
    
    Returns:
        Dictionary of metrics
    """
    n_queries = len(results)
    
    # Recall@k
    recall_at_1 = sum(1 for r in results if r['rank'] == 1) / n_queries
    recall_at_5 = sum(1 for r in results if r['rank'] and r['rank'] <= 5) / n_queries
    recall_at_10 = sum(1 for r in results if r['rank'] and r['rank'] <= 10) / n_queries
    
    # Mean Reciprocal Rank (MRR)
    reciprocal_ranks = []
    for r in results:
        if r['rank']:
            reciprocal_ranks.append(1.0 / r['rank'])
        else:
            reciprocal_ranks.append(0.0)
    mrr = np.mean(reciprocal_ranks)
    
    # Mean Average Precision (MAP)
    # For single ground truth, AP = 1/rank if found, else 0
    average_precisions = []
    for r in results:
        if r['rank']:
            average_precisions.append(1.0 / r['rank'])
        else:
            average_precisions.append(0.0)
    map_score = np.mean(average_precisions)
    
    return {
        'recall@1': recall_at_1,
        'recall@5': recall_at_5,
        'recall@10': recall_at_10,
        'mrr': mrr,
        'map': map_score,
        'n_queries': n_queries
    }


def compare_methods(clip_results: Dict, hybrid_results: Dict):
    """
    Compare CLIP-only vs Hybrid search.
    
    Args:
        clip_results: CLIP evaluation results
        hybrid_results: Hybrid evaluation results
    """
    print("\n" + "="*70)
    print("METHOD COMPARISON")
    print("="*70)
    
    clip_metrics = clip_results['metrics']
    hybrid_metrics = hybrid_results['metrics']
    
    # Accuracy comparison
    print("\n" + "-"*70)
    print("Accuracy Metrics:")
    print("-"*70)
    print(f"{'Metric':<15} {'CLIP-only':<15} {'Hybrid':<15} {'Improvement':<15}")
    print("-"*70)
    
    metrics = ['recall@1', 'recall@5', 'recall@10', 'mrr', 'map']
    metric_names = ['Recall@1', 'Recall@5', 'Recall@10', 'MRR', 'MAP']
    
    for metric, name in zip(metrics, metric_names):
        clip_val = clip_metrics[metric]
        hybrid_val = hybrid_metrics[metric]
        
        if clip_val > 0:
            improvement = ((hybrid_val - clip_val) / clip_val) * 100
            improvement_str = f"+{improvement:.1f}%"
        else:
            improvement_str = "N/A"
        
        if metric in ['recall@1', 'recall@5', 'recall@10']:
            print(f"{name:<15} {clip_val:<15.2%} {hybrid_val:<15.2%} {improvement_str:<15}")
        else:
            print(f"{name:<15} {clip_val:<15.4f} {hybrid_val:<15.4f} {improvement_str:<15}")
    
    # Latency comparison
    print("\n" + "-"*70)
    print("Latency Metrics:")
    print("-"*70)
    print(f"{'Metric':<15} {'CLIP-only':<15} {'Hybrid':<15} {'Difference':<15}")
    print("-"*70)
    
    latency_metrics = ['mean', 'median', 'p95', 'p99']
    latency_names = ['Mean', 'Median', 'P95', 'P99']
    
    for metric, name in zip(latency_metrics, latency_names):
        clip_val = clip_metrics['latencies'][metric]
        hybrid_val = hybrid_metrics['latencies'][metric]
        diff = hybrid_val - clip_val
        
        print(f"{name:<15} {clip_val:<15.2f}ms {hybrid_val:<15.2f}ms +{diff:.2f}ms")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    recall_improvement = ((hybrid_metrics['recall@10'] - clip_metrics['recall@10']) / 
                         clip_metrics['recall@10']) * 100
    latency_overhead = hybrid_metrics['latencies']['mean'] - clip_metrics['latencies']['mean']
    
    print(f"\n✓ Hybrid Search Improvements:")
    print(f"  • Recall@10: +{recall_improvement:.1f}% improvement")
    print(f"  • MRR: {hybrid_metrics['mrr']:.4f} (vs {clip_metrics['mrr']:.4f})")
    print(f"  • Latency overhead: +{latency_overhead:.2f}ms")
    print(f"  • Trade-off: {recall_improvement:.1f}% better accuracy for {latency_overhead:.0f}ms overhead")
    
    # Targets
    print(f"\n✓ Performance Targets:")
    target_recall = 0.65
    target_latency = 2000
    
    recall_met = hybrid_metrics['recall@10'] >= target_recall
    latency_met = hybrid_metrics['latencies']['mean'] < target_latency
    
    print(f"  • Recall@10 >65%: {'✓ MET' if recall_met else '✗ NOT MET'} "
          f"({hybrid_metrics['recall@10']:.2%})")
    print(f"  • Latency <2000ms: {'✓ MET' if latency_met else '✗ NOT MET'} "
          f"({hybrid_metrics['latencies']['mean']:.2f}ms)")
    
    print(f"\n{'='*70}")


def save_results(clip_results: Dict, hybrid_results: Dict, output_dir: Path):
    """Save evaluation results to JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save full results
    results_file = output_dir / 'accuracy_evaluation_results.json'
    
    data = {
        'clip_only': {
            'metrics': clip_results['metrics'],
            'method': clip_results['method']
        },
        'hybrid': {
            'metrics': hybrid_results['metrics'],
            'method': hybrid_results['method']
        },
        'comparison': {
            'recall@10_improvement': (
                (hybrid_results['metrics']['recall@10'] - 
                 clip_results['metrics']['recall@10']) /
                clip_results['metrics']['recall@10'] * 100
            ),
            'latency_overhead': (
                hybrid_results['metrics']['latencies']['mean'] -
                clip_results['metrics']['latencies']['mean']
            )
        }
    }
    
    with open(results_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"\n✓ Results saved to {results_file}")


def main():
    """Main evaluation execution."""
    print("\n" + "="*70)
    print("ACCURACY EVALUATION SUITE (T2.8)")
    print("="*70)
    print("\nThis script evaluates search accuracy using multiple metrics")
    print("and compares CLIP-only vs Hybrid search methods.")
    
    # Load components
    bi_encoder, cross_encoder, image_index, dataset = load_components()
    
    # Initialize hybrid search engine
    print("\n" + "="*60)
    print("Initializing HybridSearchEngine")
    print("="*60)
    
    engine = HybridSearchEngine(
        bi_encoder=bi_encoder,
        cross_encoder=cross_encoder,
        image_index=image_index,
        dataset=dataset,
        config={
            'k1': 100,
            'k2': 10,
            'batch_size': 4,
            'use_cache': False,
            'show_progress': False
        }
    )
    
    print(f"\n  ✓ Engine initialized")
    
    # Select test queries
    test_queries = select_test_queries(dataset, n=100)
    
    try:
        # Evaluate CLIP-only
        clip_results = evaluate_clip_only(engine, test_queries, k=10)
        
        # Evaluate Hybrid
        hybrid_results = evaluate_hybrid(engine, test_queries, k1=100, k2=10)
        
        # Compare methods
        compare_methods(clip_results, hybrid_results)
        
        # Save results
        output_dir = project_root / 'data' / 'evaluation'
        save_results(clip_results, hybrid_results, output_dir)
        
        print("\n" + "="*70)
        print("ACCURACY EVALUATION COMPLETE! 🎉")
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"\n✗ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
