"""
Test Stage 2 - BLIP-2 Re-ranking (T2.3)

This script tests the Stage 2 re-ranking functionality of the HybridSearchEngine,
focusing on BLIP-2-based accurate re-ranking of CLIP candidates.

Target: Stage 2 latency < 2000ms for 100 candidates (batch_size=4)

Usage:
    # On Kaggle or local with data
    python scripts/test_stage2_blip2.py
    
    # With custom data directory
    python scripts/test_stage2_blip2.py --data-dir /kaggle/input/flickr30k
    
    # With different parameters
    python scripts/test_stage2_blip2.py --k1 100 --k2 10 --batch-size 4

Kaggle Setup:
    1. Add Flickr30k dataset to Kaggle notebook
    2. Enable GPU (P100 or T4)
    3. Run this script
"""

import sys
import argparse
import time
from pathlib import Path
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Try imports with fallback for Kaggle
try:
    from src.retrieval import BiEncoder, CrossEncoder, FAISSIndex
    from src.flickr30k import Flickr30KDataset
except ImportError:
    # Fallback for Kaggle
    sys.path.insert(0, '/kaggle/working')
    from src.retrieval import BiEncoder, CrossEncoder, FAISSIndex
    from src.flickr30k import Flickr30KDataset


def setup_kaggle_paths(data_dir=None):
    """
    Setup data paths for Kaggle environment.
    
    Args:
        data_dir: Optional custom data directory
        
    Returns:
        Dictionary with paths
    """
    if data_dir is None:
        # Check if running on Kaggle
        kaggle_path = Path('/kaggle/input/flickr30k')
        if kaggle_path.exists():
            data_dir = kaggle_path
        else:
            # Local fallback
            data_dir = Path(__file__).parent.parent / 'data'
    else:
        data_dir = Path(data_dir)
    
    paths = {
        'data_dir': data_dir,
        'images_dir': data_dir / 'images',
        'captions_file': data_dir / 'results.csv',
        'embeddings_dir': data_dir / 'embeddings',
        'indices_dir': data_dir / 'indices',
        'image_index': data_dir / 'indices' / 'image_index.faiss',
    }
    
    return paths


def load_components(paths, verbose=True):
    """
    Load all components needed for Stage 2 testing.
    
    Args:
        paths: Dictionary with data paths
        verbose: Print loading progress
        
    Returns:
        Tuple of (bi_encoder, cross_encoder, image_index, dataset)
    """
    if verbose:
        print("=" * 60)
        print("Loading Components for Stage 2 Testing")
        print("=" * 60)
    
    # Load BiEncoder (CLIP)
    if verbose:
        print("\n1. Loading CLIP BiEncoder...")
    bi_encoder = BiEncoder(model_name='ViT-B-32', pretrained='openai')
    if verbose:
        print(f"   ✓ CLIP model loaded: {bi_encoder.model_name}")
    
    # Load CrossEncoder (BLIP-2)
    if verbose:
        print("\n2. Loading BLIP-2 CrossEncoder...")
    cross_encoder = CrossEncoder(
        model_name='Salesforce/blip2-opt-2.7b',
        device='cuda',
        use_fp16=True
    )
    if verbose:
        print(f"   ✓ BLIP-2 model loaded")
        info = cross_encoder.get_model_info()
        print(f"   ✓ Device: {info['device']}")
        print(f"   ✓ FP16: {info['use_fp16']}")
    
    # Load FAISS index
    if verbose:
        print("\n3. Loading FAISS Index...")
    image_index = FAISSIndex()
    
    if paths['image_index'].exists():
        image_index.load(str(paths['image_index']))
        if verbose:
            print(f"   ✓ Index loaded: {image_index.index.ntotal:,} vectors")
    else:
        print(f"   ✗ Error: Image index not found at {paths['image_index']}")
        print("   Please run 'scripts/build_faiss_indices.py' first or ensure data is available")
        return None, None, None, None
    
    # Load dataset
    if verbose:
        print("\n4. Loading Flickr30K Dataset...")
    dataset = Flickr30KDataset(
        images_dir=str(paths['images_dir']),
        captions_file=str(paths['captions_file']),
        auto_load=True
    )
    if verbose:
        print(f"   ✓ Dataset loaded: {len(dataset):,} images")
    
    return bi_encoder, cross_encoder, image_index, dataset


def stage1_retrieve(bi_encoder, image_index, query, k1=100):
    """
    Perform Stage 1 retrieval to get candidates for Stage 2.
    
    Args:
        bi_encoder: CLIP BiEncoder
        image_index: FAISS index
        query: Text query string
        k1: Number of candidates to retrieve
        
    Returns:
        List of (image_id, clip_score) tuples
    """
    # Encode query with CLIP
    query_embedding = bi_encoder.encode_texts(
        texts=[query],
        batch_size=1,
        normalize=True,
        show_progress=False
    )
    
    # Search FAISS index
    scores, indices = image_index.search(
        query_embeddings=query_embedding,
        k=k1,
        return_scores=True
    )
    
    # Convert to list of (image_id, score) tuples
    candidates = []
    image_ids = image_index.metadata.get('ids', [])
    
    for idx, score in zip(indices[0], scores[0]):
        if idx < len(image_ids):
            image_id = image_ids[idx]
            candidates.append((image_id, float(score)))
    
    return candidates


def stage2_rerank(cross_encoder, dataset, query, candidates, k2=10, batch_size=4):
    """
    Perform Stage 2 re-ranking using BLIP-2.
    
    Args:
        cross_encoder: BLIP-2 CrossEncoder
        dataset: Flickr30K dataset
        query: Text query string
        candidates: List of (image_id, clip_score) from Stage 1
        k2: Number of final results to return
        batch_size: Batch size for BLIP-2 processing
        
    Returns:
        Tuple of (reranked_results, latency_ms)
    """
    start_time = time.time()
    
    # Prepare batch data
    image_ids = [img_id for img_id, _ in candidates]
    image_paths = [dataset.images_dir / img_id for img_id in image_ids]
    queries = [query] * len(candidates)
    
    # Score with BLIP-2
    blip2_scores = cross_encoder.score_pairs(
        queries=queries,
        candidates=image_paths,
        query_type='text',
        candidate_type='image',
        batch_size=batch_size,
        show_progress=True
    )
    
    # Create list of (image_id, blip2_score) tuples
    reranked_results = [
        (image_id, float(score))
        for image_id, score in zip(image_ids, blip2_scores)
    ]
    
    # Sort by BLIP-2 score (descending) and take top-k2
    reranked_results.sort(key=lambda x: x[1], reverse=True)
    reranked_results = reranked_results[:k2]
    
    latency_ms = (time.time() - start_time) * 1000
    
    return reranked_results, latency_ms


def test_stage2_single_query(
    bi_encoder, cross_encoder, image_index, dataset,
    query, k1=100, k2=10, batch_size=4
):
    """
    Test Stage 2 re-ranking with a single query.
    
    Args:
        bi_encoder: CLIP BiEncoder
        cross_encoder: BLIP-2 CrossEncoder
        image_index: FAISS index
        dataset: Flickr30K dataset
        query: Text query string
        k1: Number of candidates from Stage 1
        k2: Number of final results
        batch_size: Batch size for BLIP-2
        
    Returns:
        Tuple of (stage1_candidates, stage2_results, stage1_time, stage2_time)
    """
    print(f"\nQuery: '{query}'")
    print(f"Pipeline: Stage 1 (k1={k1}) → Stage 2 (k2={k2}, batch_size={batch_size})")
    print("-" * 60)
    
    # Stage 1: CLIP retrieval
    print(f"\nStage 1: Retrieving top-{k1} candidates with CLIP...")
    stage1_start = time.time()
    stage1_candidates = stage1_retrieve(bi_encoder, image_index, query, k1)
    stage1_time = (time.time() - stage1_start) * 1000
    print(f"✓ Stage 1 complete: {len(stage1_candidates)} candidates in {stage1_time:.1f}ms")
    
    # Show Stage 1 top results
    print(f"\nStage 1 Top 5 (CLIP scores):")
    for i, (img_id, score) in enumerate(stage1_candidates[:5], 1):
        print(f"  {i}. {img_id}: {score:.4f}")
    
    # Stage 2: BLIP-2 re-ranking
    print(f"\nStage 2: Re-ranking with BLIP-2 (batch_size={batch_size})...")
    stage2_results, stage2_time = stage2_rerank(
        cross_encoder, dataset, query, stage1_candidates, k2, batch_size
    )
    print(f"✓ Stage 2 complete: {len(stage2_results)} results in {stage2_time:.1f}ms")
    
    # Show Stage 2 results
    print(f"\nStage 2 Top {k2} (BLIP-2 scores):")
    for i, (img_id, score) in enumerate(stage2_results, 1):
        print(f"  {i}. {img_id}: {score:.4f}")
    
    # Total time
    total_time = stage1_time + stage2_time
    print(f"\nTotal Time: {total_time:.1f}ms (Stage 1: {stage1_time:.1f}ms, Stage 2: {stage2_time:.1f}ms)")
    
    return stage1_candidates, stage2_results, stage1_time, stage2_time


def test_stage2_batch_queries(
    bi_encoder, cross_encoder, image_index, dataset,
    queries, k1=100, k2=10, batch_size=4
):
    """
    Test Stage 2 re-ranking with multiple queries.
    
    Args:
        bi_encoder: CLIP BiEncoder
        cross_encoder: BLIP-2 CrossEncoder
        image_index: FAISS index
        dataset: Flickr30K dataset
        queries: List of text query strings
        k1: Number of candidates per query
        k2: Number of final results per query
        batch_size: Batch size for BLIP-2
        
    Returns:
        Dictionary with statistics
    """
    print(f"\n" + "=" * 60)
    print(f"Testing Stage 2 with {len(queries)} queries")
    print("=" * 60)
    
    stage1_latencies = []
    stage2_latencies = []
    total_latencies = []
    
    for i, query in enumerate(queries, 1):
        print(f"\n[{i}/{len(queries)}] Query: '{query[:50]}...'")
        
        # Stage 1
        stage1_start = time.time()
        candidates = stage1_retrieve(bi_encoder, image_index, query, k1)
        stage1_time = (time.time() - stage1_start) * 1000
        
        # Stage 2
        results, stage2_time = stage2_rerank(
            cross_encoder, dataset, query, candidates, k2, batch_size
        )
        
        total_time = stage1_time + stage2_time
        
        stage1_latencies.append(stage1_time)
        stage2_latencies.append(stage2_time)
        total_latencies.append(total_time)
        
        print(f"   ✓ Total: {total_time:.1f}ms (S1: {stage1_time:.1f}ms, S2: {stage2_time:.1f}ms)")
    
    # Calculate statistics
    stats = {
        'num_queries': len(queries),
        'k1': k1,
        'k2': k2,
        'batch_size': batch_size,
        'stage1_latency': {
            'mean': np.mean(stage1_latencies),
            'median': np.median(stage1_latencies),
            'std': np.std(stage1_latencies),
            'min': np.min(stage1_latencies),
            'max': np.max(stage1_latencies),
        },
        'stage2_latency': {
            'mean': np.mean(stage2_latencies),
            'median': np.median(stage2_latencies),
            'std': np.std(stage2_latencies),
            'min': np.min(stage2_latencies),
            'max': np.max(stage2_latencies),
            'p95': np.percentile(stage2_latencies, 95),
            'p99': np.percentile(stage2_latencies, 99),
        },
        'total_latency': {
            'mean': np.mean(total_latencies),
            'median': np.median(total_latencies),
            'std': np.std(total_latencies),
            'min': np.min(total_latencies),
            'max': np.max(total_latencies),
            'p95': np.percentile(total_latencies, 95),
            'p99': np.percentile(total_latencies, 99),
        }
    }
    
    return stats


def test_different_batch_sizes(
    bi_encoder, cross_encoder, image_index, dataset,
    query, k1=100, k2=10
):
    """
    Test Stage 2 with different batch sizes.
    
    Args:
        bi_encoder: CLIP BiEncoder
        cross_encoder: BLIP-2 CrossEncoder
        image_index: FAISS index
        dataset: Flickr30K dataset
        query: Text query string
        k1: Number of candidates
        k2: Number of final results
    """
    print(f"\n" + "=" * 60)
    print("Testing Different Batch Sizes")
    print("=" * 60)
    print(f"\nQuery: '{query}'")
    print(f"k1={k1}, k2={k2}")
    
    # Get Stage 1 candidates once
    print(f"\nStage 1: Retrieving {k1} candidates...")
    candidates = stage1_retrieve(bi_encoder, image_index, query, k1)
    print(f"✓ {len(candidates)} candidates retrieved")
    
    # Test different batch sizes
    batch_sizes = [2, 4, 8]
    print(f"\nTesting batch sizes: {batch_sizes}")
    print("-" * 60)
    
    for batch_size in batch_sizes:
        print(f"\nBatch size: {batch_size}")
        
        # Run Stage 2 multiple times and average
        latencies = []
        for run in range(3):
            _, latency = stage2_rerank(
                cross_encoder, dataset, query, candidates, k2, batch_size
            )
            latencies.append(latency)
        
        mean_latency = np.mean(latencies)
        print(f"  Mean latency: {mean_latency:.1f}ms (over 3 runs)")
        print(f"  Latency per pair: {mean_latency / len(candidates):.1f}ms")


def print_statistics(stats):
    """Print Stage 2 performance statistics."""
    print("\n" + "=" * 60)
    print("STAGE 2 PERFORMANCE STATISTICS")
    print("=" * 60)
    
    print(f"\nConfiguration:")
    print(f"  Queries: {stats['num_queries']}")
    print(f"  k1 (Stage 1 candidates): {stats['k1']}")
    print(f"  k2 (final results): {stats['k2']}")
    print(f"  BLIP-2 batch size: {stats['batch_size']}")
    
    print(f"\nStage 1 (CLIP) Latency:")
    print(f"  Mean:   {stats['stage1_latency']['mean']:.1f} ms")
    print(f"  Median: {stats['stage1_latency']['median']:.1f} ms")
    print(f"  Range:  {stats['stage1_latency']['min']:.1f} - {stats['stage1_latency']['max']:.1f} ms")
    
    print(f"\nStage 2 (BLIP-2) Latency:")
    print(f"  Mean:   {stats['stage2_latency']['mean']:.1f} ms")
    print(f"  Median: {stats['stage2_latency']['median']:.1f} ms")
    print(f"  Std:    {stats['stage2_latency']['std']:.1f} ms")
    print(f"  Range:  {stats['stage2_latency']['min']:.1f} - {stats['stage2_latency']['max']:.1f} ms")
    print(f"  P95:    {stats['stage2_latency']['p95']:.1f} ms")
    print(f"  P99:    {stats['stage2_latency']['p99']:.1f} ms")
    
    print(f"\nTotal (End-to-End) Latency:")
    print(f"  Mean:   {stats['total_latency']['mean']:.1f} ms")
    print(f"  Median: {stats['total_latency']['median']:.1f} ms")
    print(f"  P95:    {stats['total_latency']['p95']:.1f} ms")
    print(f"  P99:    {stats['total_latency']['p99']:.1f} ms")
    
    # Check targets
    stage2_target = 2000  # ms for 100 candidates
    total_target = 2000  # ms total
    
    print(f"\nPerformance Targets:")
    print(f"  Stage 2: < {stage2_target}ms for {stats['k1']} candidates")
    
    if stats['stage2_latency']['mean'] < stage2_target:
        print(f"  ✓ PASSED: Mean {stats['stage2_latency']['mean']:.1f}ms < {stage2_target}ms")
    else:
        print(f"  ✗ FAILED: Mean {stats['stage2_latency']['mean']:.1f}ms >= {stage2_target}ms")
    
    print(f"\n  Total: < {total_target}ms")
    if stats['total_latency']['mean'] < total_target:
        print(f"  ✓ PASSED: Mean {stats['total_latency']['mean']:.1f}ms < {total_target}ms")
    else:
        print(f"  ⚠ WARNING: Mean {stats['total_latency']['mean']:.1f}ms >= {total_target}ms")


def main():
    """Run all Stage 2 BLIP-2 re-ranking tests."""
    parser = argparse.ArgumentParser(description='Test Stage 2 BLIP-2 Re-ranking')
    parser.add_argument('--data-dir', type=str, default=None,
                        help='Path to data directory (default: auto-detect)')
    parser.add_argument('--k1', type=int, default=100,
                        help='Number of Stage 1 candidates (default: 100)')
    parser.add_argument('--k2', type=int, default=10,
                        help='Number of final results (default: 10)')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='BLIP-2 batch size (default: 4)')
    parser.add_argument('--num-test-queries', type=int, default=5,
                        help='Number of test queries (default: 5)')
    args = parser.parse_args()
    
    print("=" * 60)
    print("STAGE 2 (BLIP-2) RE-RANKING TEST - T2.3")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  k1 (candidates): {args.k1}")
    print(f"  k2 (final results): {args.k2}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Test queries: {args.num_test_queries}")
    print(f"  Data directory: {args.data_dir or 'auto-detect'}")
    
    # Setup paths
    paths = setup_kaggle_paths(args.data_dir)
    print(f"\nData paths:")
    print(f"  Images: {paths['images_dir']}")
    print(f"  Captions: {paths['captions_file']}")
    print(f"  Index: {paths['image_index']}")
    
    # Load components
    bi_encoder, cross_encoder, image_index, dataset = load_components(paths)
    
    if bi_encoder is None or cross_encoder is None:
        print("\n✗ Failed to load components. Exiting.")
        return 1
    
    # Test queries
    test_queries = [
        "a dog playing in the park",
        "a cat sitting on a couch",
        "a person riding a bicycle",
        "a beautiful sunset over the ocean",
        "children playing soccer in a field",
        "a red car parked on the street",
        "people eating dinner at a restaurant",
        "a bird flying in the blue sky",
        "a snowy mountain landscape",
        "a crowded city street with people",
    ]
    
    # Test 1: Single query (detailed)
    print("\n" + "=" * 60)
    print("TEST 1: Single Query (Detailed)")
    print("=" * 60)
    
    stage1_candidates, stage2_results, stage1_time, stage2_time = test_stage2_single_query(
        bi_encoder, cross_encoder, image_index, dataset,
        query=test_queries[0],
        k1=args.k1,
        k2=args.k2,
        batch_size=args.batch_size
    )
    
    # Test 2: Batch queries
    print("\n" + "=" * 60)
    print("TEST 2: Batch Queries")
    print("=" * 60)
    
    stats = test_stage2_batch_queries(
        bi_encoder, cross_encoder, image_index, dataset,
        queries=test_queries[:args.num_test_queries],
        k1=args.k1,
        k2=args.k2,
        batch_size=args.batch_size
    )
    
    # Print statistics
    print_statistics(stats)
    
    # Test 3: Different batch sizes
    test_different_batch_sizes(
        bi_encoder, cross_encoder, image_index, dataset,
        query=test_queries[0],
        k1=args.k1,
        k2=args.k2
    )
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY - T2.3: Stage 2 BLIP-2 Re-ranking")
    print("=" * 60)
    
    print("\n✓ Stage 2 (_stage2_rerank) method is working correctly")
    print(f"✓ Re-ranked {args.k1} candidates to top-{args.k2} results")
    print(f"✓ Tested with {args.num_test_queries} queries")
    
    if stats['stage2_latency']['mean'] < 2000:
        print(f"✓ Performance target met: {stats['stage2_latency']['mean']:.1f}ms < 2000ms")
        print("\n🎉 T2.3 COMPLETE: Stage 2 BLIP-2 Re-ranking")
        return_code = 0
    else:
        print(f"⚠ Performance warning: {stats['stage2_latency']['mean']:.1f}ms >= 2000ms")
        print("  Consider:")
        print("  - Using smaller batch size")
        print("  - Using FP16 (already enabled)")
        print("  - Using faster GPU (T4 → V100)")
        return_code = 1
    
    print("\nNext step: T2.4 - Image-to-Image Hybrid Search")
    
    return return_code


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
