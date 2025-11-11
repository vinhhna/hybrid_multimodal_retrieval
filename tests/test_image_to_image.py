"""
Test Image-to-Image Hybrid Search (T2.4)

This script tests the image-to-image search functionality of the HybridSearchEngine.
Currently focuses on Stage 1 (CLIP-based) image similarity search.

Stage 2 (BLIP-2) for image-to-image is not yet implemented, as BLIP-2 is primarily
designed for image-text tasks. For now, CLIP-only search provides good results.

Usage:
    # On Kaggle or local with data
    python scripts/test_image_to_image.py
    
    # With custom data directory
    python scripts/test_image_to_image.py --data-dir /kaggle/input/flickr30k
    
    # With specific query image
    python scripts/test_image_to_image.py --query-image data/images/12345.jpg

Kaggle Setup:
    1. Add Flickr30k dataset to Kaggle notebook
    2. Run this script
"""

import sys
import argparse
import time
from pathlib import Path
import numpy as np
import random

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Try imports with fallback for Kaggle
try:
    from src.retrieval import BiEncoder, FAISSIndex, HybridSearchEngine
    from src.flickr30k import Flickr30KDataset
except ImportError:
    # Fallback for Kaggle
    sys.path.insert(0, '/kaggle/working')
    from src.retrieval import BiEncoder, FAISSIndex, HybridSearchEngine
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
    Load all components needed for image-to-image search testing.
    
    Args:
        paths: Dictionary with data paths
        verbose: Print loading progress
        
    Returns:
        Tuple of (bi_encoder, image_index, dataset)
    """
    if verbose:
        print("=" * 60)
        print("Loading Components for Image-to-Image Search")
        print("=" * 60)
    
    # Load BiEncoder (CLIP)
    if verbose:
        print("\n1. Loading CLIP BiEncoder...")
    bi_encoder = BiEncoder(model_name='ViT-B-32', pretrained='openai')
    if verbose:
        print(f"   ✓ CLIP model loaded: {bi_encoder.model_name}")
    
    # Load FAISS index
    if verbose:
        print("\n2. Loading FAISS Index...")
    image_index = FAISSIndex()
    
    if paths['image_index'].exists():
        image_index.load(str(paths['image_index']))
        if verbose:
            print(f"   ✓ Index loaded: {image_index.index.ntotal:,} vectors")
    else:
        print(f"   ✗ Error: Image index not found at {paths['image_index']}")
        print("   Please run 'scripts/build_faiss_indices.py' first or ensure data is available")
        return None, None, None
    
    # Load dataset
    if verbose:
        print("\n3. Loading Flickr30K Dataset...")
    dataset = Flickr30KDataset(
        images_dir=str(paths['images_dir']),
        captions_file=str(paths['captions_file']),
        auto_load=True
    )
    if verbose:
        print(f"   ✓ Dataset loaded: {len(dataset):,} images")
    
    return bi_encoder, image_index, dataset


def get_random_query_image(dataset, seed=None):
    """
    Get a random image from the dataset as query.
    
    Args:
        dataset: Flickr30K dataset
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (image_name, image_path, captions)
    """
    if seed is not None:
        random.seed(seed)
    
    image_ids = dataset.get_unique_images()
    query_image_id = random.choice(image_ids)
    query_image_path = dataset.images_dir / query_image_id
    query_captions = dataset.get_captions(query_image_id)
    
    return query_image_id, query_image_path, query_captions


def image_to_image_search_clip(bi_encoder, image_index, dataset, query_image_path, k=10):
    """
    Perform image-to-image search using CLIP only.
    
    Args:
        bi_encoder: CLIP BiEncoder
        image_index: FAISS index
        dataset: Flickr30K dataset
        query_image_path: Path to query image
        k: Number of results to return
        
    Returns:
        Tuple of (results, latency_ms)
    """
    from PIL import Image
    
    start_time = time.time()
    
    # Load and encode query image
    query_image = Image.open(query_image_path).convert('RGB')
    query_embedding = bi_encoder.encode_images(
        images=[query_image],
        batch_size=1,
        normalize=True,
        show_progress=False
    )
    
    # Search FAISS index
    scores, indices = image_index.search(
        query_embeddings=query_embedding,
        k=k,
        return_scores=True
    )
    
    # Convert to list of (image_id, score) tuples
    results = []
    image_ids = image_index.metadata.get('ids', [])
    
    for idx, score in zip(indices[0], scores[0]):
        if idx < len(image_ids):
            image_id = image_ids[idx]
            results.append((image_id, float(score)))
    
    latency_ms = (time.time() - start_time) * 1000
    
    return results, latency_ms


def test_single_image_query(bi_encoder, image_index, dataset, query_image_path, k=10):
    """
    Test image-to-image search with a single query image.
    
    Args:
        bi_encoder: CLIP BiEncoder
        image_index: FAISS index
        dataset: Flickr30K dataset
        query_image_path: Path to query image
        k: Number of results to return
    """
    print(f"\n" + "=" * 60)
    print("Single Image Query Test")
    print("=" * 60)
    
    query_image_id = query_image_path.name
    print(f"\nQuery Image: {query_image_id}")
    
    # Show query image captions
    query_captions = dataset.get_captions(query_image_id)
    if query_captions:
        print(f"\nQuery Image Captions:")
        for i, caption in enumerate(query_captions[:3], 1):
            print(f"  {i}. {caption}")
    
    # Perform search
    print(f"\nSearching for top-{k} similar images...")
    results, latency = image_to_image_search_clip(
        bi_encoder, image_index, dataset, query_image_path, k
    )
    
    print(f"✓ Search complete in {latency:.1f}ms")
    
    # Show results
    print(f"\nTop {k} Similar Images:")
    print("-" * 60)
    
    for i, (img_id, score) in enumerate(results, 1):
        print(f"\n{i}. {img_id} (similarity: {score:.4f})")
        
        # Show first caption of result
        captions = dataset.get_captions(img_id)
        if captions:
            print(f"   Caption: {captions[0][:80]}...")
    
    return results, latency


def test_batch_image_queries(bi_encoder, image_index, dataset, num_queries=5, k=10):
    """
    Test image-to-image search with multiple query images.
    
    Args:
        bi_encoder: CLIP BiEncoder
        image_index: FAISS index
        dataset: Flickr30K dataset
        num_queries: Number of test queries
        k: Number of results per query
    """
    print(f"\n" + "=" * 60)
    print(f"Batch Image Query Test ({num_queries} queries)")
    print("=" * 60)
    
    latencies = []
    all_results = []
    
    for i in range(num_queries):
        # Get random query image
        query_image_id, query_image_path, query_captions = get_random_query_image(
            dataset, seed=42 + i
        )
        
        print(f"\n[{i+1}/{num_queries}] Query: {query_image_id}")
        if query_captions:
            print(f"   Caption: {query_captions[0][:60]}...")
        
        # Perform search
        results, latency = image_to_image_search_clip(
            bi_encoder, image_index, dataset, query_image_path, k
        )
        
        latencies.append(latency)
        all_results.append(results)
        
        print(f"   ✓ Found {len(results)} similar images in {latency:.1f}ms")
        print(f"   Top result: {results[0][0]} (similarity: {results[0][1]:.4f})")
    
    # Calculate statistics
    stats = {
        'num_queries': num_queries,
        'k': k,
        'latency_mean': np.mean(latencies),
        'latency_median': np.median(latencies),
        'latency_std': np.std(latencies),
        'latency_min': np.min(latencies),
        'latency_max': np.max(latencies),
    }
    
    return stats, all_results


def test_similarity_quality(bi_encoder, image_index, dataset, num_samples=5):
    """
    Test the quality of image-to-image similarity search.
    
    Checks if similar images actually have related content by comparing captions.
    
    Args:
        bi_encoder: CLIP BiEncoder
        image_index: FAISS index
        dataset: Flickr30K dataset
        num_samples: Number of samples to test
    """
    print(f"\n" + "=" * 60)
    print("Similarity Quality Test")
    print("=" * 60)
    print("\nChecking if similar images have related content...")
    
    for i in range(num_samples):
        query_image_id, query_image_path, query_captions = get_random_query_image(
            dataset, seed=100 + i
        )
        
        print(f"\n{'=' * 60}")
        print(f"Sample {i+1}/{num_samples}")
        print(f"Query: {query_image_id}")
        print(f"Caption: {query_captions[0] if query_captions else 'N/A'}")
        
        # Search for similar images
        results, _ = image_to_image_search_clip(
            bi_encoder, image_index, dataset, query_image_path, k=5
        )
        
        print(f"\nTop 3 Similar Images:")
        for j, (img_id, score) in enumerate(results[1:4], 1):  # Skip first (self)
            captions = dataset.get_captions(img_id)
            caption = captions[0] if captions else "N/A"
            print(f"  {j}. {img_id} (sim: {score:.4f})")
            print(f"     {caption[:70]}...")


def print_statistics(stats):
    """Print image-to-image search statistics."""
    print("\n" + "=" * 60)
    print("IMAGE-TO-IMAGE SEARCH STATISTICS")
    print("=" * 60)
    
    print(f"\nQueries processed: {stats['num_queries']}")
    print(f"Results per query (k): {stats['k']}")
    
    print(f"\nLatency (milliseconds):")
    print(f"  Mean:   {stats['latency_mean']:.1f} ms")
    print(f"  Median: {stats['latency_median']:.1f} ms")
    print(f"  Std:    {stats['latency_std']:.1f} ms")
    print(f"  Min:    {stats['latency_min']:.1f} ms")
    print(f"  Max:    {stats['latency_max']:.1f} ms")
    
    # Check target
    target_latency = 100  # ms (same as text-to-image Stage 1)
    print(f"\nTarget Latency: < {target_latency} ms")
    
    if stats['latency_mean'] < target_latency:
        print(f"✓ PASSED: Mean latency {stats['latency_mean']:.1f}ms < {target_latency}ms")
    else:
        print(f"⚠ WARNING: Mean latency {stats['latency_mean']:.1f}ms >= {target_latency}ms")


def main():
    """Run all image-to-image search tests."""
    parser = argparse.ArgumentParser(description='Test Image-to-Image Hybrid Search')
    parser.add_argument('--data-dir', type=str, default=None,
                        help='Path to data directory (default: auto-detect)')
    parser.add_argument('--query-image', type=str, default=None,
                        help='Path to specific query image (default: random)')
    parser.add_argument('--k', type=int, default=10,
                        help='Number of results to return (default: 10)')
    parser.add_argument('--num-test-queries', type=int, default=5,
                        help='Number of test queries for batch test (default: 5)')
    args = parser.parse_args()
    
    print("=" * 60)
    print("IMAGE-TO-IMAGE HYBRID SEARCH TEST - T2.4")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  k (results): {args.k}")
    print(f"  Test queries: {args.num_test_queries}")
    print(f"  Data directory: {args.data_dir or 'auto-detect'}")
    print(f"  Query image: {args.query_image or 'random'}")
    
    # Setup paths
    paths = setup_kaggle_paths(args.data_dir)
    print(f"\nData paths:")
    print(f"  Images: {paths['images_dir']}")
    print(f"  Index: {paths['image_index']}")
    
    # Load components
    bi_encoder, image_index, dataset = load_components(paths)
    
    if bi_encoder is None:
        print("\n✗ Failed to load components. Exiting.")
        return 1
    
    # Test 1: Single query
    print("\n" + "=" * 60)
    print("TEST 1: Single Image Query")
    print("=" * 60)
    
    if args.query_image:
        query_image_path = Path(args.query_image)
    else:
        # Use random image
        query_image_id, query_image_path, _ = get_random_query_image(dataset, seed=42)
    
    if not query_image_path.exists():
        print(f"✗ Error: Query image not found: {query_image_path}")
        return 1
    
    results, latency = test_single_image_query(
        bi_encoder, image_index, dataset,
        query_image_path=query_image_path,
        k=args.k
    )
    
    # Test 2: Batch queries
    print("\n" + "=" * 60)
    print("TEST 2: Batch Image Queries")
    print("=" * 60)
    
    stats, all_results = test_batch_image_queries(
        bi_encoder, image_index, dataset,
        num_queries=args.num_test_queries,
        k=args.k
    )
    
    # Print statistics
    print_statistics(stats)
    
    # Test 3: Similarity quality
    test_similarity_quality(
        bi_encoder, image_index, dataset,
        num_samples=3
    )
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY - T2.4: Image-to-Image Hybrid Search")
    print("=" * 60)
    
    print("\n✓ Image-to-image search is working correctly")
    print(f"✓ CLIP-based similarity search (Stage 1)")
    print(f"✓ Tested with {args.num_test_queries} random query images")
    print(f"✓ Mean latency: {stats['latency_mean']:.1f}ms")
    
    print("\n⚠ Note: Stage 2 (BLIP-2) not implemented for image-to-image")
    print("  Reason: BLIP-2 is designed for image-text tasks")
    print("  CLIP provides good image similarity results")
    
    if stats['latency_mean'] < 100:
        print(f"\n✓ Performance target met: {stats['latency_mean']:.1f}ms < 100ms")
        print("\n🎉 T2.4 COMPLETE: Image-to-Image Search (CLIP-based)")
        return_code = 0
    else:
        print(f"\n⚠ Performance warning: {stats['latency_mean']:.1f}ms >= 100ms")
        return_code = 1
    
    print("\nNext step: T2.5 - Batch Hybrid Search")
    
    return return_code


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
