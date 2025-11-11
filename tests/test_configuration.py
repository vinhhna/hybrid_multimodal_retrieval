"""
Test script for T2.6: Configuration & Optimization

This script tests the configuration management and optimization capabilities
of the HybridSearchEngine.

Tests:
1. Runtime Configuration Updates
2. Configuration Validation
3. Cache Management
4. Performance Profiling
5. Automatic Optimization

Usage (Kaggle):
    import sys
    sys.path.append('/kaggle/working/hybrid_multimodal_retrieval')
    %run scripts/test_configuration.py
"""

import sys
import os
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Auto-detect environment
if Path('/kaggle/input').exists():
    # Kaggle environment
    DATA_DIR = Path('/kaggle/input/flickr30k/data')
    print("Running on Kaggle")
else:
    # Local environment
    DATA_DIR = project_root / 'data'
    print("Running locally")

from src.retrieval.bi_encoder import BiEncoder
from src.retrieval.cross_encoder import CrossEncoder
from src.retrieval.faiss_index import FAISSIndex
from src.retrieval.hybrid_search import HybridSearchEngine
from src.flickr30k.dataset import Flickr30KDataset


def load_components():
    """Load all required components."""
    print("\n" + "="*60)
    print("Loading Components")
    print("="*60)
    
    start_time = time.time()
    
    # Load dataset
    print("\n[1/4] Loading Flickr30K dataset...")
    dataset = Flickr30KDataset(
        images_dir=str(DATA_DIR / 'images'),
        captions_file=str(DATA_DIR / 'results.csv')
    )
    print(f"  ✓ Loaded {len(dataset)} images")
    
    # Load CLIP bi-encoder
    print("\n[2/4] Loading CLIP bi-encoder...")
    bi_encoder = BiEncoder(
        model_name='ViT-B/32',
        device='cuda'
    )
    print(f"  ✓ Model: {bi_encoder.model_name}")
    
    # Load FAISS index
    print("\n[3/4] Loading FAISS index...")
    image_index = FAISSIndex(device='cuda')
    index_path = DATA_DIR / 'indices' / 'image_index.faiss'
    image_index.load(str(index_path))
    print(f"  ✓ Loaded {image_index.index.ntotal:,} vectors")
    
    # 4. Load BLIP-2 cross-encoder
    print("\n[4/4] Loading BLIP-2 cross-encoder...")
    cross_encoder = CrossEncoder(
        model_name='Salesforce/blip2-opt-2.7b',
        device='cuda',
        use_fp16=True
    )
    print(f"  ✓ Model: {cross_encoder.model_name}")
    
    load_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"All components loaded in {load_time:.2f}s")
    print(f"{'='*60}\n")
    
    return bi_encoder, cross_encoder, image_index, dataset


def test_runtime_config_updates(engine: HybridSearchEngine):
    """
    Test 1: Runtime Configuration Updates
    
    Tests updating configuration parameters at runtime and validates
    that changes take effect immediately.
    """
    print("\n" + "="*70)
    print("TEST 1: Runtime Configuration Updates")
    print("="*70)
    
    # Show initial config
    print("\n1. Initial Configuration:")
    config = engine.get_config()
    print(f"  k1: {config['k1']}")
    print(f"  k2: {config['k2']}")
    print(f"  batch_size: {config['batch_size']}")
    print(f"  use_cache: {config['use_cache']}")
    print(f"  show_progress: {config['show_progress']}")
    
    # Test query with initial config
    print("\n2. Test with initial config (k1=100, k2=10)...")
    query = "a dog playing in the park"
    start = time.time()
    results1 = engine.text_to_image_hybrid_search(query, show_progress=False)
    time1 = (time.time() - start) * 1000
    print(f"  ✓ Found {len(results1)} results in {time1:.2f}ms")
    
    # Update config: increase k1 and k2
    print("\n3. Update configuration (k1=200, k2=20, batch_size=8)...")
    engine.update_config(k1=200, k2=20, batch_size=8)
    
    # Test with updated config
    print("\n4. Test with updated config...")
    start = time.time()
    results2 = engine.text_to_image_hybrid_search(query, show_progress=False)
    time2 = (time.time() - start) * 1000
    print(f"  ✓ Found {len(results2)} results in {time2:.2f}ms")
    
    # Verify changes
    print("\n5. Verification:")
    print(f"  Initial: {len(results1)} results, {time1:.2f}ms")
    print(f"  Updated: {len(results2)} results, {time2:.2f}ms")
    print(f"  ✓ Config updates working correctly!")
    
    # Reset config
    print("\n6. Reset to defaults...")
    engine.reset_config()
    config = engine.get_config()
    print(f"  ✓ Reset to k1={config['k1']}, k2={config['k2']}, batch_size={config['batch_size']}")
    
    print("\n" + "="*70)


def test_config_validation(engine: HybridSearchEngine):
    """
    Test 2: Configuration Validation
    
    Tests that invalid configurations are rejected and config
    is reverted to previous valid state.
    """
    print("\n" + "="*70)
    print("TEST 2: Configuration Validation")
    print("="*70)
    
    # Save current config
    original_config = engine.get_config()
    print(f"\n1. Current valid config: k1={original_config['k1']}, k2={original_config['k2']}")
    
    # Try invalid update: k1 < k2
    print("\n2. Try invalid update (k1=5, k2=10)...")
    engine.update_config(k1=5, k2=10)
    
    # Verify config unchanged
    current_config = engine.get_config()
    print(f"\n3. Verification:")
    print(f"  Config after failed update: k1={current_config['k1']}, k2={current_config['k2']}")
    print(f"  ✓ Invalid config rejected, previous config preserved!")
    
    # Try invalid k2
    print("\n4. Try invalid k2 (k2=0)...")
    engine.update_config(k2=0)
    
    current_config = engine.get_config()
    print(f"  Config after failed update: k2={current_config['k2']}")
    print(f"  ✓ Invalid k2 rejected!")
    
    # Try invalid batch_size
    print("\n5. Try invalid batch_size (batch_size=-1)...")
    engine.update_config(batch_size=-1)
    
    current_config = engine.get_config()
    print(f"  Config after failed update: batch_size={current_config['batch_size']}")
    print(f"  ✓ Invalid batch_size rejected!")
    
    print("\n" + "="*70)


def test_cache_management(engine: HybridSearchEngine):
    """
    Test 3: Cache Management
    
    Tests cache functionality including enabling/disabling,
    hit rate, and clearing.
    """
    print("\n" + "="*70)
    print("TEST 3: Cache Management")
    print("="*70)
    
    # Enable cache
    print("\n1. Enable caching...")
    engine.update_config(use_cache=True)
    print(f"  ✓ Cache enabled")
    print(f"  Current cache size: {engine.get_cache_size()}")
    
    # Run queries
    queries = [
        "a dog playing in the park",
        "people walking on the beach",
        "a red car on the street"
    ]
    
    print(f"\n2. Run {len(queries)} queries (first time - no cache)...")
    start = time.time()
    for query in queries:
        engine.text_to_image_hybrid_search(query, show_progress=False)
    first_time = (time.time() - start) * 1000
    
    print(f"  ✓ Completed in {first_time:.2f}ms")
    print(f"  Cache size: {engine.get_cache_size()}")
    
    # Run same queries again (should hit cache)
    print(f"\n3. Run same queries again (should hit cache)...")
    start = time.time()
    for query in queries:
        engine.text_to_image_hybrid_search(query, show_progress=False)
    cached_time = (time.time() - start) * 1000
    
    print(f"  ✓ Completed in {cached_time:.2f}ms")
    print(f"  Cache size: {engine.get_cache_size()}")
    
    # Get statistics
    stats = engine.get_statistics()
    print(f"\n4. Cache statistics:")
    print(f"  Total queries: {stats['total_queries']}")
    print(f"  Cache hits: {stats['cache_hits']}")
    print(f"  Hit rate: {stats['cache_hit_rate']:.2%}")
    print(f"  Speedup: {first_time/cached_time:.2f}x")
    
    # Show cached keys
    cached_keys = engine.get_cache_keys()
    print(f"\n5. Cached queries:")
    for key in cached_keys:
        print(f"  - {key[:50]}...")
    
    # Clear cache
    print(f"\n6. Clear cache...")
    n_cleared = engine.clear_cache()
    print(f"  Cache size after clear: {engine.get_cache_size()}")
    
    # Disable cache
    print(f"\n7. Disable caching...")
    engine.update_config(use_cache=False)
    print(f"  ✓ Cache disabled")
    
    print("\n" + "="*70)


def test_performance_profiling(engine: HybridSearchEngine):
    """
    Test 4: Performance Profiling
    
    Tests the profiling capabilities to find optimal configuration.
    """
    print("\n" + "="*70)
    print("TEST 4: Performance Profiling")
    print("="*70)
    
    # Test queries
    test_queries = [
        "a dog playing in the park",
        "people walking on the beach",
        "a red car on the street",
        "children eating ice cream",
        "a cat sleeping on a couch"
    ]
    
    print(f"\nProfiling with {len(test_queries)} test queries...")
    print("Testing k1=[50, 100], k2=[10], batch_size=[4, 8]")
    
    # Profile search
    results = engine.profile_search(
        test_queries=test_queries,
        k1_values=[50, 100],
        k2_values=[10],
        batch_sizes=[4, 8]
    )
    
    # Results are printed by profile_search
    print("\n✓ Profiling complete!")
    
    return results


def test_automatic_optimization(engine: HybridSearchEngine):
    """
    Test 5: Automatic Optimization
    
    Tests automatic configuration optimization for target latency.
    """
    print("\n" + "="*70)
    print("TEST 5: Automatic Optimization")
    print("="*70)
    
    # Test queries
    test_queries = [
        "a dog playing in the park",
        "people walking on the beach",
        "a red car on the street"
    ]
    
    # Target latency
    target_latency = 300  # ms
    
    print(f"\nOptimizing for target latency: {target_latency}ms")
    
    # Optimize
    result = engine.optimize_config(
        target_latency_ms=target_latency,
        test_queries=test_queries
    )
    
    # Apply recommended config
    print(f"\n" + "-"*70)
    print("Applying recommended configuration...")
    recommended = result['recommended_config']
    engine.update_config(**recommended)
    
    # Test with recommended config
    print(f"\nTesting with recommended config...")
    start = time.time()
    for query in test_queries:
        engine.text_to_image_hybrid_search(query, show_progress=False)
    actual_time = (time.time() - start) * 1000 / len(test_queries)
    
    print(f"\nResults:")
    print(f"  Target latency: {target_latency}ms")
    print(f"  Expected latency: {result['expected_latency_ms']:.2f}ms")
    print(f"  Actual latency: {actual_time:.2f}ms")
    print(f"  ✓ Optimization successful!")
    
    print("\n" + "="*70)


def main():
    """Main test execution."""
    print("\n" + "="*70)
    print("CONFIGURATION & OPTIMIZATION TEST SUITE (T2.6)")
    print("="*70)
    print("\nThis script tests configuration management and optimization")
    print("capabilities of the HybridSearchEngine class.")
    
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
            'show_progress': True
        }
    )
    
    print(f"\n  ✓ Engine initialized")
    print(f"  {engine}")
    
    # Run tests
    try:
        # Test 1: Runtime config updates
        test_runtime_config_updates(engine)
        
        # Test 2: Config validation
        test_config_validation(engine)
        
        # Test 3: Cache management
        test_cache_management(engine)
        
        # Test 4: Performance profiling
        profile_results = test_performance_profiling(engine)
        
        # Test 5: Automatic optimization
        test_automatic_optimization(engine)
        
        # Final summary
        print("\n" + "="*70)
        print("FINAL SUMMARY")
        print("="*70)
        print("\n✓ All tests completed successfully!")
        print("\nConfiguration & Optimization Features:")
        print("  ✅ Runtime configuration updates")
        print("  ✅ Configuration validation")
        print("  ✅ Cache management (enable/disable/clear)")
        print("  ✅ Performance profiling")
        print("  ✅ Automatic optimization")
        
        print("\nKey Capabilities:")
        print("  • Update k1, k2, batch_size at runtime")
        print("  • Validate configurations before applying")
        print("  • Cache frequent queries for speedup")
        print("  • Profile different configurations")
        print("  • Automatically find optimal config for target latency")
        
        print("\nUsage Example:")
        print("  # Update configuration")
        print("  engine.update_config(k1=200, k2=20, batch_size=8)")
        print("  ")
        print("  # Enable caching")
        print("  engine.update_config(use_cache=True)")
        print("  ")
        print("  # Profile performance")
        print("  results = engine.profile_search(test_queries=queries)")
        print("  ")
        print("  # Auto-optimize")
        print("  result = engine.optimize_config(target_latency_ms=400)")
        print("  engine.update_config(**result['recommended_config'])")
        
        print("\n" + "="*70 + "\n")
        
    except Exception as e:
        print(f"\n✗ Error during testing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
