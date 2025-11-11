"""
Integration Example: Using Hybrid Search with Configuration (T3.1 + T3.2)

This script demonstrates how to use the integrated hybrid search system
with configuration management.

Features:
1. Load configuration from YAML or use presets
2. Initialize HybridSearchEngine with configuration
3. Perform various search operations
4. Update configuration at runtime
5. Compare different configurations

Usage:
    python scripts/integration_example.py [--config CONFIG_PATH] [--preset PRESET_NAME]
"""

import sys
import time
import argparse
from pathlib import Path
from typing import List, Tuple

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

from src.retrieval import (
    BiEncoder,
    CrossEncoder,
    FAISSIndex,
    HybridSearchEngine,
    load_config,
    load_preset
)
from src.flickr30k.dataset import Flickr30KDataset


def load_components():
    """Load all required components."""
    print("\n" + "="*70)
    print("LOADING COMPONENTS")
    print("="*70)
    
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
    
    print(f"\n{'='*70}")
    print("Components loaded successfully!")
    print(f"{'='*70}\n")
    
    return bi_encoder, cross_encoder, image_index, dataset


def example_1_basic_usage(config, components):
    """Example 1: Basic hybrid search with configuration."""
    print("\n" + "="*70)
    print("EXAMPLE 1: Basic Hybrid Search with Configuration")
    print("="*70)
    
    bi_encoder, cross_encoder, image_index, dataset = components
    
    # Create engine with configuration
    print("\nInitializing HybridSearchEngine with configuration...")
    engine = HybridSearchEngine(
        bi_encoder=bi_encoder,
        cross_encoder=cross_encoder,
        image_index=image_index,
        dataset=dataset,
        config={
            'k1': config.get('stage1.k1'),
            'k2': config.get('stage2.k2'),
            'batch_size': config.get('stage2.batch_size'),
            'use_cache': config.get('performance.use_cache'),
            'show_progress': config.get('performance.show_progress')
        }
    )
    
    print(f"✓ Engine initialized with configuration")
    print(f"\nConfiguration:")
    print(f"  k1 (Stage 1 candidates): {config.get('stage1.k1')}")
    print(f"  k2 (Final results): {config.get('stage2.k2')}")
    print(f"  Batch size: {config.get('stage2.batch_size')}")
    print(f"  Cache: {config.get('performance.use_cache')}")
    
    # Perform search
    query = "a dog playing in the park"
    print(f"\nQuery: '{query}'")
    
    start = time.time()
    results = engine.text_to_image_hybrid_search(
        query=query,
        k1=config.get('stage1.k1'),
        k2=config.get('stage2.k2'),
        show_progress=True
    )
    latency = (time.time() - start) * 1000
    
    print(f"\n✓ Search completed in {latency:.2f}ms")
    print(f"\nTop 5 Results:")
    for i, (image_id, score) in enumerate(results[:5], 1):
        print(f"  {i}. {image_id} - Score: {score:.4f}")


def example_2_preset_configs(components):
    """Example 2: Using preset configurations."""
    print("\n" + "="*70)
    print("EXAMPLE 2: Using Preset Configurations")
    print("="*70)
    
    bi_encoder, cross_encoder, image_index, dataset = components
    
    presets = ['fast', 'balanced', 'accurate']
    query = "sunset over the ocean"
    
    print(f"\nQuery: '{query}'")
    print(f"\nTesting different presets:\n")
    
    for preset_name in presets:
        print(f"--- Preset: {preset_name} ---")
        
        # Load preset configuration
        config = load_preset(preset_name)
        
        print(f"  Configuration:")
        print(f"    k1: {config.get('stage1.k1')}")
        print(f"    k2: {config.get('stage2.k2')}")
        print(f"    batch_size: {config.get('stage2.batch_size')}")
        
        # Create engine with preset config
        engine = HybridSearchEngine(
            bi_encoder=bi_encoder,
            cross_encoder=cross_encoder,
            image_index=image_index,
            dataset=dataset,
            config={
                'k1': config.get('stage1.k1'),
                'k2': config.get('stage2.k2'),
                'batch_size': config.get('stage2.batch_size'),
                'use_cache': False,
                'show_progress': False
            }
        )
        
        # Run search
        start = time.time()
        results = engine.text_to_image_hybrid_search(
            query=query,
            k1=config.get('stage1.k1'),
            k2=config.get('stage2.k2'),
            show_progress=False
        )
        latency = (time.time() - start) * 1000
        
        print(f"  Latency: {latency:.2f}ms")
        print(f"  Top result: {results[0][0]} ({results[0][1]:.4f})")
        print()


def example_3_runtime_config(config, components):
    """Example 3: Runtime configuration updates."""
    print("\n" + "="*70)
    print("EXAMPLE 3: Runtime Configuration Updates")
    print("="*70)
    
    bi_encoder, cross_encoder, image_index, dataset = components
    
    # Create engine
    engine = HybridSearchEngine(
        bi_encoder=bi_encoder,
        cross_encoder=cross_encoder,
        image_index=image_index,
        dataset=dataset,
        config={
            'k1': config.get('stage1.k1'),
            'k2': config.get('stage2.k2'),
            'batch_size': config.get('stage2.batch_size'),
            'use_cache': True,
            'show_progress': False
        }
    )
    
    query = "a cat sitting on a couch"
    print(f"\nQuery: '{query}'")
    
    # Test with original config
    print(f"\n1. Original configuration (k1={config.get('stage1.k1')}, k2={config.get('stage2.k2')}):")
    start = time.time()
    results1 = engine.text_to_image_hybrid_search(query=query, show_progress=False)
    latency1 = (time.time() - start) * 1000
    print(f"   Latency: {latency1:.2f}ms")
    print(f"   Top result: {results1[0][0]}")
    
    # Update configuration at runtime
    print(f"\n2. Updated configuration (k1=50, k2=5):")
    engine.update_config(k1=50, k2=5)
    start = time.time()
    results2 = engine.text_to_image_hybrid_search(query=query, k1=50, k2=5, show_progress=False)
    latency2 = (time.time() - start) * 1000
    print(f"   Latency: {latency2:.2f}ms (speedup: {latency1/latency2:.2f}x)")
    print(f"   Top result: {results2[0][0]}")
    
    # Another update
    print(f"\n3. Updated configuration (k1=200, k2=20):")
    engine.update_config(k1=200, k2=20)
    start = time.time()
    results3 = engine.text_to_image_hybrid_search(query=query, k1=200, k2=20, show_progress=False)
    latency3 = (time.time() - start) * 1000
    print(f"   Latency: {latency3:.2f}ms")
    print(f"   Top result: {results3[0][0]}")
    
    print(f"\n✓ Configuration can be updated at runtime without recreating engine")


def example_4_batch_search(config, components):
    """Example 4: Batch search with configuration."""
    print("\n" + "="*70)
    print("EXAMPLE 4: Batch Search with Configuration")
    print("="*70)
    
    bi_encoder, cross_encoder, image_index, dataset = components
    
    # Create engine
    engine = HybridSearchEngine(
        bi_encoder=bi_encoder,
        cross_encoder=cross_encoder,
        image_index=image_index,
        dataset=dataset,
        config={
            'k1': config.get('stage1.k1'),
            'k2': config.get('stage2.k2'),
            'batch_size': config.get('batch_search.stage2_batch_size'),
            'use_cache': False,
            'show_progress': True
        }
    )
    
    queries = [
        "a dog running on the beach",
        "people at a party",
        "sunset over mountains",
        "a child eating ice cream",
        "city street at night"
    ]
    
    print(f"\nProcessing {len(queries)} queries in batch...")
    print(f"Configuration:")
    print(f"  Stage 1 batch size: {config.get('batch_search.stage1_batch_size')}")
    print(f"  Stage 2 batch size: {config.get('batch_search.stage2_batch_size')}")
    
    start = time.time()
    batch_results = engine.batch_text_to_image_search(
        queries=queries,
        k1=config.get('stage1.k1'),
        k2=config.get('stage2.k2'),
        show_progress=True
    )
    batch_latency = time.time() - start
    
    print(f"\n✓ Batch search completed in {batch_latency:.2f}s")
    print(f"  Average per query: {(batch_latency/len(queries))*1000:.2f}ms")
    
    print(f"\nTop result for each query:")
    for i, query in enumerate(queries, 1):
        top_result = batch_results[query][0]
        print(f"  {i}. '{query[:40]}...'")
        print(f"     → {top_result[0]} ({top_result[1]:.4f})")


def example_5_save_config(config):
    """Example 5: Save custom configuration."""
    print("\n" + "="*70)
    print("EXAMPLE 5: Save Custom Configuration")
    print("="*70)
    
    # Modify configuration
    print("\nCreating custom configuration...")
    config.set('stage1.k1', 150)
    config.set('stage2.k2', 15)
    config.set('stage2.batch_size', 8)
    config.set('performance.use_cache', True)
    
    print(f"Custom settings:")
    print(f"  k1: {config.get('stage1.k1')}")
    print(f"  k2: {config.get('stage2.k2')}")
    print(f"  batch_size: {config.get('stage2.batch_size')}")
    print(f"  use_cache: {config.get('performance.use_cache')}")
    
    # Save configuration
    custom_path = project_root / 'configs' / 'my_custom_config.yaml'
    config.save(str(custom_path))
    print(f"\n✓ Configuration saved to {custom_path}")
    
    # Load it back
    loaded_config = load_config(str(custom_path))
    print(f"✓ Configuration loaded from file")
    
    # Verify
    assert loaded_config.get('stage1.k1') == 150
    assert loaded_config.get('stage2.k2') == 15
    print(f"✓ Saved and loaded values match")


def main():
    """Main execution."""
    parser = argparse.ArgumentParser(
        description="Integration example for hybrid search with configuration"
    )
    parser.add_argument(
        '--config',
        type=str,
        help='Path to configuration YAML file'
    )
    parser.add_argument(
        '--preset',
        type=str,
        choices=['fast', 'balanced', 'accurate', 'memory_efficient'],
        help='Use a preset configuration'
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("HYBRID SEARCH INTEGRATION EXAMPLE (T3.1 + T3.2)")
    print("="*70)
    print("\nDemonstrating integrated hybrid search with configuration management")
    
    # Load configuration
    if args.preset:
        print(f"\nLoading preset configuration: {args.preset}")
        config = load_preset(args.preset)
    elif args.config:
        print(f"\nLoading configuration from: {args.config}")
        config = load_config(args.config)
    else:
        print(f"\nLoading default configuration from: configs/hybrid_config.yaml")
        config_path = project_root / 'configs' / 'hybrid_config.yaml'
        config = load_config(str(config_path))
    
    print(f"\n{config}")
    
    # Load components
    components = load_components()
    
    # Run examples
    try:
        example_1_basic_usage(config, components)
        example_2_preset_configs(components)
        example_3_runtime_config(config, components)
        example_4_batch_search(config, components)
        example_5_save_config(config)
        
        print("\n" + "="*70)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY! 🎉")
        print("="*70)
        print("\nKey takeaways:")
        print("  ✓ Configuration can be loaded from YAML files")
        print("  ✓ Preset configurations available for common use cases")
        print("  ✓ Configuration can be updated at runtime")
        print("  ✓ Custom configurations can be saved and reused")
        print("  ✓ Hybrid search integrates seamlessly with configuration")
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
