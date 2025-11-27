"""
Integration test for Phase 4 query enrichment with HybridSearchEngine.

This script tests the full pipeline integration including:
- Config loading
- Entity context loading
- Query enrichment
- Graph mode search
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import numpy as np
from unittest.mock import MagicMock, patch

from graph.config import load_entity_graph_config
from retrieval.hybrid_search import HybridSearchEngine


def create_mock_components():
    """Create mock BiEncoder, CrossEncoder, and FAISS index."""
    
    # Mock BiEncoder
    bi_encoder = MagicMock()
    bi_encoder.encode_texts = lambda texts, batch_size=None, normalize=True, show_progress=False: (
        torch.nn.functional.normalize(
            torch.randn(len(texts), 512, dtype=torch.float32),
            p=2, dim=-1
        ).numpy()
    )
    bi_encoder.encode_images = lambda images, batch_size=None, normalize=True, show_progress=False: (
        torch.nn.functional.normalize(
            torch.randn(len(images), 512, dtype=torch.float32),
            p=2, dim=-1
        ).numpy()
    )
    
    # Mock CrossEncoder
    cross_encoder = MagicMock()
    cross_encoder.higher_is_better = True
    cross_encoder.score_pairs = lambda queries, candidates, **kwargs: (
        [0.5 + 0.1 * i for i in range(len(queries))]
    )
    
    # Mock FAISS index
    faiss_index = MagicMock()
    faiss_index.index.ntotal = 1000
    faiss_index.metadata = {'ids': [f'img_{i}.jpg' for i in range(1000)]}
    faiss_index.search = lambda query_embeddings, k, return_scores=True: (
        np.random.rand(query_embeddings.shape[0], k).astype(np.float32),
        np.random.randint(0, 1000, (query_embeddings.shape[0], k))
    )
    
    # Mock dataset
    dataset = MagicMock()
    dataset.images_dir = Path("data/images")
    dataset.get_unique_images = lambda: [f'img_{i}.jpg' for i in range(100)]
    dataset.get_captions = lambda img_id: [f"Caption for {img_id}"]
    
    return bi_encoder, cross_encoder, faiss_index, dataset


def create_synthetic_entity_data():
    """Create synthetic entity context, embeddings, and metadata."""
    entity_context = {
        0: {"entity": "dog", "image_ids": ["img_0.jpg", "img_1.jpg"], "caption_ids": []},
        1: {"entity": "cat", "image_ids": ["img_1.jpg", "img_2.jpg"], "caption_ids": []},
        2: {"entity": "tree", "image_ids": ["img_0.jpg", "img_2.jpg"], "caption_ids": []},
        3: {"entity": "park", "image_ids": ["img_0.jpg", "img_3.jpg"], "caption_ids": []},
    }
    
    torch.manual_seed(42)
    entity_embeddings = torch.nn.functional.normalize(
        torch.randn(4, 512, dtype=torch.float32),
        p=2, dim=-1
    )
    
    entity_meta = {
        0: {"entity": "dog", "df_caption": 2, "df_image": 2},
        1: {"entity": "cat", "df_caption": 2, "df_image": 2},
        2: {"entity": "tree", "df_caption": 2, "df_image": 2},
        3: {"entity": "park", "df_caption": 2, "df_image": 2},
    }
    
    return entity_context, entity_embeddings, entity_meta


def test_graph_search_integration():
    """Test the full graph search integration."""
    print("=" * 70)
    print("Integration Test: Graph Search with Query Enrichment")
    print("=" * 70)
    
    # Load config
    config_path = Path(__file__).parent.parent / "configs" / "entity_graph.yaml"
    cfg = load_entity_graph_config(config_path)
    print("\n✓ Config loaded")
    
    # Create mock components
    bi_encoder, cross_encoder, faiss_index, dataset = create_mock_components()
    print("✓ Mock components created")
    
    # Create entity data
    entity_context, entity_embeddings, entity_meta = create_synthetic_entity_data()
    print("✓ Synthetic entity data created")
    
    # Initialize HybridSearchEngine
    engine = HybridSearchEngine(
        bi_encoder=bi_encoder,
        cross_encoder=cross_encoder,
        image_index=faiss_index,
        dataset=dataset,
        config={'k1': 50, 'k2': 10, 'batch_size': 4, 'show_progress': False}
    )
    print("✓ HybridSearchEngine initialized")
    
    # Test query
    query = "a dog playing in the park"
    print(f"\nTest Query: {query}")
    
    # Run graph search with enrichment
    print("\nRunning text_to_image_graph_search()...")
    
    with patch('graph.graph_search._load_entity_embeddings', return_value=entity_embeddings):
        with patch('graph.graph_search._load_entity_meta', return_value=entity_meta):
            try:
                results = engine.text_to_image_graph_search(
                    query=query,
                    entity_context=entity_context,
                    phase4_config=cfg,
                    k1=50,
                    k2=10,
                    show_progress=False
                )
                
                print("\n✓ Graph search completed successfully!")
                print(f"\nResults: {len(results)} images returned")
                
                if results:
                    print("\nTop 3 results:")
                    for i, (img_id, score) in enumerate(results[:3], 1):
                        print(f"  {i}. {img_id}: {score:.4f}")
                
                return True
                
            except Exception as e:
                print(f"\n✗ Error during graph search: {e}")
                import traceback
                traceback.print_exc()
                return False


def test_enrichment_disabled():
    """Test that search works when enrichment is disabled."""
    print("\n" + "=" * 70)
    print("Integration Test: Enrichment Disabled (Baseline Mode)")
    print("=" * 70)
    
    # Load config and disable enrichment
    config_path = Path(__file__).parent.parent / "configs" / "entity_graph.yaml"
    cfg = load_entity_graph_config(config_path)
    cfg['query_enrichment']['enabled'] = False
    print("\n✓ Config loaded (enrichment disabled)")
    
    # Create mock components
    bi_encoder, cross_encoder, faiss_index, dataset = create_mock_components()
    print("✓ Mock components created")
    
    # Create entity data
    entity_context, entity_embeddings, entity_meta = create_synthetic_entity_data()
    print("✓ Synthetic entity data created")
    
    # Initialize HybridSearchEngine
    engine = HybridSearchEngine(
        bi_encoder=bi_encoder,
        cross_encoder=cross_encoder,
        image_index=faiss_index,
        dataset=dataset,
        config={'k1': 50, 'k2': 10, 'batch_size': 4, 'show_progress': False}
    )
    print("✓ HybridSearchEngine initialized")
    
    # Test query
    query = "a cat on a tree"
    print(f"\nTest Query: {query}")
    
    # Run graph search (should fall back to hybrid search)
    print("\nRunning text_to_image_graph_search() with enrichment disabled...")
    
    try:
        results = engine.text_to_image_graph_search(
            query=query,
            entity_context=entity_context,
            phase4_config=cfg,
            k1=50,
            k2=10,
            show_progress=False
        )
        
        print("\n✓ Search completed (fallback to hybrid search)!")
        print(f"\nResults: {len(results)} images returned")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Error during search: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("\n" + "=" * 70)
    print("PHASE 4 INTEGRATION TESTS")
    print("=" * 70)
    
    success_count = 0
    total_tests = 2
    
    # Test 1: Graph search with enrichment
    if test_graph_search_integration():
        success_count += 1
    
    # Test 2: Graph search with enrichment disabled
    if test_enrichment_disabled():
        success_count += 1
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"Passed: {success_count}/{total_tests}")
    
    if success_count == total_tests:
        print("\n✓ All integration tests passed!")
        return 0
    else:
        print(f"\n✗ {total_tests - success_count} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
