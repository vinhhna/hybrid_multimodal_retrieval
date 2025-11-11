"""
Fast Smoke-Test Evaluator (T2.8)

This script provides a lightweight, reproducible evaluation for quick sanity checks:
- CLIP-only (baseline)
- Hybrid Search (CLIP + BLIP-2)

Metrics calculated:
- Recall@1, Recall@5, Recall@10
- Mean Reciprocal Rank (MRR)
- nDCG@10
- MAP (= MRR for single relevant)
- Mean and median latency

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

# Fast mode constants
FAST_SEED = 2025
FAST_N = 25        # number of queries for smoke test
FAST_K1 = 30       # Stage-1 candidates for Hybrid
FAST_K2 = 10       # Final k for both methods

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
        captions_file=str(DATA_DIR / 'results.csv')
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


def select_test_queries(dataset: Flickr30KDataset, n: int = FAST_N) -> List[Dict[str, Any]]:
    """
    Select random, reproducible test queries from dataset (FAST MODE).
    
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
    print(f"SELECTING {n} TEST QUERIES (FAST MODE)")
    print("="*70)

    unique_images = dataset.get_unique_images()
    rng = np.random.default_rng(FAST_SEED)
    n = min(n, len(unique_images))
    chosen = rng.choice(unique_images, size=n, replace=False)

    test_queries = []
    for image_id in chosen:
        captions = dataset.get_captions(image_id)
        if not captions:
            continue
        test_queries.append({
            'query': captions[0],
            'ground_truth': image_id,
            'alternatives': captions[1:] if len(captions) > 1 else []
        })

    print(f"\n✓ Selected {len(test_queries)} test queries")
    return test_queries


def evaluate_clip_only(
    engine: HybridSearchEngine,
    test_queries: List[Dict[str, Any]],
    k: int = FAST_K2
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
        'mean': float(np.mean(latencies)),
        'median': float(np.median(latencies)),
    }
    
    print(f"\n{'='*70}")
    print(f"CLIP-only Results:")
    print(f"  Recall@1:  {metrics['recall@1']:.2%}")
    print(f"  Recall@5:  {metrics['recall@5']:.2%}")
    print(f"  Recall@10: {metrics['recall@10']:.2%}")
    print(f"  MRR:       {metrics['mrr']:.4f}")
    print(f"  nDCG@10:   {metrics['ndcg@10']:.4f}")
    print(f"  MAP:       {metrics['map']:.4f}  (≡ MRR; single relevant)")
    print(f"  Latency:   {metrics['latencies']['mean']:.2f}ms (mean), {metrics['latencies']['median']:.2f}ms (median)")
    print(f"{'='*70}")
    
    return {
        'method': 'CLIP-only',
        'results': results,
        'metrics': metrics
    }


def evaluate_hybrid(
    engine: HybridSearchEngine,
    test_queries: List[Dict[str, Any]],
    k1: int = FAST_K1,
    k2: int = FAST_K2
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
        'mean': float(np.mean(latencies)),
        'median': float(np.median(latencies)),
    }
    
    print(f"\n{'='*70}")
    print(f"Hybrid Results:")
    print(f"  Recall@1:  {metrics['recall@1']:.2%}")
    print(f"  Recall@5:  {metrics['recall@5']:.2%}")
    print(f"  Recall@10: {metrics['recall@10']:.2%}")
    print(f"  MRR:       {metrics['mrr']:.4f}")
    print(f"  nDCG@10:   {metrics['ndcg@10']:.4f}")
    print(f"  MAP:       {metrics['map']:.4f}  (≡ MRR; single relevant)")
    print(f"  Latency:   {metrics['latencies']['mean']:.2f}ms (mean), {metrics['latencies']['median']:.2f}ms (median)")
    print(f"{'='*70}")
    
    return {
        'method': 'Hybrid',
        'results': results,
        'metrics': metrics
    }


def ndcg_at_k(rank, k=10):
    """Calculate nDCG@k for single relevant item."""
    if rank is None or rank > k:
        return 0.0
    # Single relevant item → IDCG=1
    return 1.0 / np.log2(rank + 1)


def calculate_metrics(results: List[Dict], k: int) -> Dict[str, float]:
    """
    Calculate evaluation metrics.
    
    Metrics:
    - Recall@1, Recall@5, Recall@10
    - Mean Reciprocal Rank (MRR)
    - nDCG@10
    - Mean Average Precision (MAP = MRR for single relevant)
    
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
    rr = [(1.0 / r['rank']) if r['rank'] else 0.0 for r in results]
    mrr = float(np.mean(rr))
    
    # nDCG@10
    ndcg = float(np.mean([ndcg_at_k(r['rank'], k=10) for r in results]))
    
    # In single-relevant setting, MAP == MRR
    map_score = mrr
    
    return {
        'recall@1': recall_at_1,
        'recall@5': recall_at_5,
        'recall@10': recall_at_10,
        'mrr': mrr,
        'ndcg@10': ndcg,
        'map': map_score,
        'n_queries': n_queries
    }


def compare_methods(clip_results: Dict, hybrid_results: Dict):
    """
    Compare CLIP-only vs Hybrid search (FAST MODE).
    
    Args:
        clip_results: CLIP evaluation results
        hybrid_results: Hybrid evaluation results
    """
    print("\n" + "="*70)
    print("METHOD COMPARISON (FAST)")
    print("="*70)
    
    clip_metrics = clip_results['metrics']
    hybrid_metrics = hybrid_results['metrics']
    
    # Accuracy comparison
    print("\n" + "-"*70)
    print("Accuracy Metrics:")
    print("-"*70)
    print(f"{'Metric':<15} {'CLIP-only':<15} {'Hybrid':<15} {'Improvement':<15}")
    print("-"*70)
    
    metrics = ['recall@1', 'recall@5', 'recall@10', 'mrr', 'ndcg@10', 'map']
    metric_names = ['Recall@1', 'Recall@5', 'Recall@10', 'MRR', 'nDCG@10', 'MAP']
    
    for metric, name in zip(metrics, metric_names):
        clip_val = clip_metrics[metric]
        hybrid_val = hybrid_metrics[metric]
        
        if clip_val > 0:
            improvement = ((hybrid_val - clip_val) / clip_val) * 100
            improvement_str = f"{improvement:+.1f}%"
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
    
    latency_metrics = ['mean', 'median']
    latency_names = ['Mean', 'Median']
    
    for metric, name in zip(latency_metrics, latency_names):
        clip_val = clip_metrics['latencies'][metric]
        hybrid_val = hybrid_metrics['latencies'][metric]
        diff = hybrid_val - clip_val
        
        print(f"{name:<15} {clip_val:<15.2f}ms {hybrid_val:<15.2f}ms {diff:+.2f}ms")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    delta_recall10 = (hybrid_metrics['recall@10'] - clip_metrics['recall@10']) * 100
    latency_overhead = hybrid_metrics['latencies']['mean'] - clip_metrics['latencies']['mean']
    
    print(f"\n✓ Hybrid vs CLIP (FAST):")
    print(f"  • Δ Recall@10: {delta_recall10:+.1f}%")
    print(f"  • Δ MRR:       {hybrid_metrics['mrr'] - clip_metrics['mrr']:+.4f}")
    print(f"  • Δ Latency:   {latency_overhead:+.2f}ms (mean)")
    
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
    """Main evaluation execution (FAST MODE)."""
    print("\n" + "="*70)
    print("FAST SMOKE-TEST EVALUATOR (T2.8)")
    print("="*70)
    print("\nQuick reproducible evaluation for sanity checks")
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
            'k1': FAST_K1,
            'k2': FAST_K2,
            'batch_size': 8,
            'use_cache': False,
            'show_progress': False,
            'fusion_method': 'weighted',
            'stage1_weight': 0.3,
            'stage2_weight': 0.7,
        }
    )
    
    print(f"\n  ✓ Engine initialized")
    
    # Warm-up (not included in metrics)
    print(f"\nWarming up engine...")
    try:
        _ = engine.text_to_image_hybrid_search(query="a dog", k1=min(FAST_K1, 10), k2=5, show_progress=False)
        _ = engine.text_to_image_hybrid_search(query="a person", k1=min(FAST_K1, 10), k2=5, show_progress=False)
        print(f"  ✓ Warm-up complete")
    except Exception:
        pass
    
    # Select test queries
    test_queries = select_test_queries(dataset, n=FAST_N)
    
    try:
        # Evaluate CLIP-only
        clip_results = evaluate_clip_only(engine, test_queries, k=FAST_K2)
        
        # Evaluate Hybrid
        hybrid_results = evaluate_hybrid(engine, test_queries, k1=FAST_K1, k2=FAST_K2)
        
        # Compare methods
        compare_methods(clip_results, hybrid_results)
        
        # Save results
        output_dir = project_root / 'data' / 'evaluation'
        save_results(clip_results, hybrid_results, output_dir)
        
        print("\n" + "="*70)
        print("FAST EVALUATION COMPLETE! 🎉")
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"\n✗ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
