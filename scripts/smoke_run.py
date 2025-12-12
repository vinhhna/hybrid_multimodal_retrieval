#!/usr/bin/env python3
"""
Smoke Test for Phase 5 Hybrid Multimodal Retrieval

Tests that the core pipeline is runnable with graceful fallbacks.
Validates: CLIP retrieval works, KG/BLIP-2 fail gracefully if missing.

Usage:
    python scripts\\smoke_run.py
    python scripts\\smoke_run.py --config configs\\phase5.yaml
    python scripts\\smoke_run.py --query "a dog running on grass"
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import yaml
import numpy as np
from typing import Optional, Dict, Any


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def check_artifacts(config: Dict[str, Any]) -> Dict[str, bool]:
    """
    Check which artifacts are available.
    
    Returns:
        Dict indicating availability of: clip, faiss, kg, blip2
    """
    results = {
        'clip': False,
        'faiss': False,
        'kg': False,
        'blip2': False,
        'dataset': False
    }
    
    # Check FAISS index
    index_path = Path("data/indices/image_index.faiss")
    meta_path = Path("data/indices/image_index.json")
    results['faiss'] = index_path.exists() and meta_path.exists()
    
    # Check KG artifacts
    graph_path = Path("data/graph/entity_graph.pt")
    entity_vocab_path = Path("data/entities/entity_vocab.json")
    results['kg'] = graph_path.exists() and entity_vocab_path.exists()
    
    # Check dataset
    dataset_path = Path("data/images/")
    results['dataset'] = dataset_path.exists() and any(dataset_path.iterdir())
    
    # CLIP is always available (downloads on demand)
    results['clip'] = True
    
    # BLIP-2 would be loaded on demand, assume available
    results['blip2'] = True
    
    return results


def run_smoke_test(config_path: str, query: str = "a dog running on grass") -> bool:
    """
    Run smoke test of the retrieval pipeline.
    
    Args:
        config_path: Path to config YAML
        query: Test query string
        
    Returns:
        True if test passed, False otherwise
    """
    print("=" * 70)
    print("PHASE 5 SMOKE TEST - Hybrid Multimodal Retrieval")
    print("=" * 70)
    print()
    
    # Load config
    print(f"📁 Loading config from: {config_path}")
    try:
        config = load_config(config_path)
        print("✓ Config loaded successfully")
    except Exception as e:
        print(f"✗ Failed to load config: {e}")
        return False
    
    print()
    
    # Check artifacts
    print("🔍 Checking artifacts...")
    artifacts = check_artifacts(config)
    for name, available in artifacts.items():
        status = "✓" if available else "✗"
        print(f"  {status} {name.upper()}: {'Available' if available else 'Missing'}")
    print()
    
    # Validate minimum requirements
    if not artifacts['faiss']:
        print("⚠️  FAISS index missing - cannot run retrieval!")
        print("   Run: python scripts\\build_faiss_indices.py")
        return False
    
    if not artifacts['dataset']:
        print("⚠️  Image dataset missing - cannot run retrieval!")
        print("   Run: python scripts\\download_flickr30k.py")
        return False
    
    # Import modules
    print("📦 Importing modules...")
    try:
        from src.retrieval.bi_encoder import BiEncoder
        from src.retrieval.faiss_index import FAISSIndex
        from src.flickr30k.dataset import Flickr30KDataset
        print("✓ Core modules imported")
        
        # Optional imports with fallback
        try:
            from src.retrieval.cross_encoder import CrossEncoder
            has_cross_encoder = True
            print("✓ Cross-encoder (BLIP-2) available")
        except Exception as e:
            has_cross_encoder = False
            print(f"⚠️  Cross-encoder unavailable (will skip reranking): {e}")
        
        try:
            from src.retrieval.hybrid_search import HybridSearchEngine
            print("✓ Hybrid search engine imported")
        except Exception as e:
            print(f"✗ Failed to import HybridSearchEngine: {e}")
            return False
            
    except Exception as e:
        print(f"✗ Failed to import modules: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print()
    
    # Initialize components
    print("🚀 Initializing retrieval pipeline...")
    try:
        # Load CLIP encoder
        print("  Loading CLIP encoder...")
        bi_encoder = BiEncoder()
        print("  ✓ CLIP loaded")
        
        # Load FAISS index
        print("  Loading FAISS index...")
        image_index = FAISSIndex()
        image_index.load("data/indices/image_index.faiss")
        print(f"  ✓ FAISS loaded ({image_index.index.ntotal} vectors)")
        
        # Load dataset
        print("  Loading dataset...")
        dataset = Flickr30KDataset()
        print(f"  ✓ Dataset loaded ({len(dataset)} images)")
        
        # Load cross-encoder if available
        cross_encoder = None
        if has_cross_encoder and artifacts['blip2']:
            try:
                print("  Loading BLIP-2 cross-encoder...")
                cross_encoder = CrossEncoder()
                print("  ✓ BLIP-2 loaded")
            except Exception as e:
                print(f"  ⚠️  BLIP-2 loading failed (will skip): {e}")
        
        # Load entity graph if available
        entity_graph = None
        if artifacts['kg']:
            try:
                print("  Loading entity graph...")
                import torch
                entity_graph = torch.load("data/graph/entity_graph.pt")
                n_entities = entity_graph["entity"].x.shape[0] if hasattr(entity_graph, "__getitem__") else 0
                print(f"  ✓ Entity graph loaded ({n_entities} entities)")
            except Exception as e:
                print(f"  ⚠️  Entity graph loading failed (will skip): {e}")
        
        # Create search engine
        print("  Creating hybrid search engine...")
        
        # Use a minimal cross-encoder stub if not available
        if cross_encoder is None:
            # Create minimal stub
            class CrossEncoderStub:
                def __init__(self):
                    pass
                def score_image_text_pairs(self, *args, **kwargs):
                    return []
            cross_encoder = CrossEncoderStub()
        
        engine = HybridSearchEngine(
            bi_encoder=bi_encoder,
            cross_encoder=cross_encoder,
            image_index=image_index,
            dataset=dataset,
            entity_graph=entity_graph,
            config={'k1': 50, 'k2': 10}
        )
        print("✓ Pipeline initialized")
        
    except Exception as e:
        print(f"✗ Failed to initialize pipeline: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print()
    
    # Run test query
    print(f"🔎 Running test query: \"{query}\"")
    try:
        # Use text_to_image_graph_search if KG available, else text_to_image_hybrid_search
        if entity_graph is not None:
            print("  Using KG-enhanced search...")
            results = engine.text_to_image_graph_search(
                query=query,
                k1=50,
                k2=10,
                alpha_clip=0.5,
                alpha_kg=0.3,
                alpha_xenc=0.2
            )
        else:
            print("  Using CLIP + BLIP-2 hybrid search...")
            results = engine.text_to_image_hybrid_search(
                query=query,
                k1=50,
                k2=10
            )
        
        print(f"✓ Query executed successfully!")
        print()
        print(f"📊 Top-5 Results:")
        for i, (image_id, score) in enumerate(results[:5], 1):
            print(f"  {i}. {image_id:<20} (score: {score:.4f})")
        
    except Exception as e:
        print(f"✗ Query execution failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print()
    print("=" * 70)
    print("✓ SMOKE TEST PASSED")
    print("=" * 70)
    return True


def main():
    parser = argparse.ArgumentParser(description="Phase 5 smoke test")
    parser.add_argument(
        '--config',
        type=str,
        default='configs/hybrid_config.yaml',
        help='Path to config file (default: configs/hybrid_config.yaml)'
    )
    parser.add_argument(
        '--query',
        type=str,
        default='a dog running on grass',
        help='Test query string'
    )
    
    args = parser.parse_args()
    
    # Run smoke test
    success = run_smoke_test(args.config, args.query)
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
