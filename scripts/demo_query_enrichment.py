"""
Demo script for Phase 4 query enrichment.

This script demonstrates the query enrichment pipeline with synthetic data.
For real usage, you need:
  - entity_context.json
  - entity_embeddings.pt
  - entity_meta.json
  - A loaded BiEncoder (CLIP)
  - FAISS indices for CLIP search
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import numpy as np
from unittest.mock import patch
from graph.graph_search import enrich_query
from graph.config import load_entity_graph_config, get_query_enrichment_config


def create_dummy_encoder():
    """Create a dummy encoder for testing."""
    class DummyEncoder:
        def encode_texts(self, texts, normalize=True, show_progress=False):
            # Return random L2-normalized embeddings
            torch.manual_seed(len(texts[0]))
            embeddings = torch.randn(len(texts), 512, dtype=torch.float32)
            if normalize:
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=-1)
            return embeddings.numpy()
    
    return DummyEncoder()


def main():
    print("=" * 70)
    print("Phase 4 Query Enrichment Demo")
    print("=" * 70)
    
    # Load configuration
    config_path = Path(__file__).parent.parent / "configs" / "entity_graph.yaml"
    print(f"\nLoading config from: {config_path}")
    cfg = load_entity_graph_config(config_path)
    enrichment_cfg = get_query_enrichment_config(cfg)
    
    print("\nQuery Enrichment Configuration:")
    for key, value in enrichment_cfg.items():
        print(f"  {key}: {value}")
    
    # Create synthetic entity context
    print("\n" + "=" * 70)
    print("Creating synthetic entity context...")
    entity_context = {
        0: {
            "entity": "dog",
            "image_ids": ["img_0.jpg", "img_1.jpg", "img_5.jpg"],
            "caption_ids": ["cap_0", "cap_1"]
        },
        1: {
            "entity": "cat",
            "image_ids": ["img_1.jpg", "img_2.jpg"],
            "caption_ids": ["cap_2", "cap_3"]
        },
        2: {
            "entity": "tree",
            "image_ids": ["img_0.jpg", "img_2.jpg", "img_3.jpg"],
            "caption_ids": ["cap_0", "cap_4", "cap_5"]
        },
        3: {
            "entity": "park",
            "image_ids": ["img_0.jpg", "img_3.jpg", "img_4.jpg"],
            "caption_ids": ["cap_6", "cap_7"]
        }
    }
    print(f"Created {len(entity_context)} synthetic entities")
    
    # Create synthetic CLIP seeds (simulating CLIP search results)
    print("\nSimulating CLIP search seeds...")
    seeds = [
        ("img_0.jpg", 0.95),  # Will find: dog, tree, park
        ("img_1.jpg", 0.88),  # Will find: dog, cat
        ("img_2.jpg", 0.82),  # Will find: cat, tree
        ("img_3.jpg", 0.75),  # Will find: tree, park
    ]
    print(f"Generated {len(seeds)} seed results")
    for img_id, score in seeds:
        print(f"  {img_id}: {score:.3f}")
    
    # Create dummy encoder
    encoder = create_dummy_encoder()
    
    # Create synthetic entity embeddings and metadata
    print("\nCreating synthetic entity embeddings...")
    torch.manual_seed(42)
    entity_embeddings = torch.randn(len(entity_context), 512, dtype=torch.float32)
    entity_embeddings = torch.nn.functional.normalize(entity_embeddings, p=2, dim=-1)
    print(f"Created embeddings with shape: {entity_embeddings.shape}")
    
    entity_meta = {
        0: {"entity": "dog", "df_caption": 2, "df_image": 3},
        1: {"entity": "cat", "df_caption": 2, "df_image": 2},
        2: {"entity": "tree", "df_caption": 3, "df_image": 3},
        3: {"entity": "park", "df_caption": 2, "df_image": 3}
    }
    
    # Test query
    query = "a brown dog playing in the park"
    print("\n" + "=" * 70)
    print(f"Original Query: {query}")
    print("=" * 70)
    
    # Run enrichment with patched loaders
    print("\nRunning query enrichment...")
    try:
        with patch('graph.graph_search._load_entity_embeddings', return_value=entity_embeddings):
            with patch('graph.graph_search._load_entity_meta', return_value=entity_meta):
                result = enrich_query(
                    query=query,
                    seeds=seeds,
                    encoders=encoder,
                    entity_context=entity_context,
                    cfg=cfg
                )
        
        print("\n" + "=" * 70)
        print("Enrichment Result:")
        print("=" * 70)
        print(f"Original Query: {result.original_query}")
        print(f"Enriched Query: {result.enriched_query}")
        print(f"\nSelected Entities ({len(result.entity_ids)}):")
        for i, (eid, name, score) in enumerate(zip(
            result.entity_ids,
            result.entity_names,
            result.entity_scores.tolist()
        )):
            print(f"  {i+1}. {name} (id={eid}, score={score:.4f})")
        
        print(f"\nEmbedding Shapes:")
        print(f"  q0: {result.q0.shape}")
        print(f"  q_enriched: {result.q_enriched.shape}")
        
        print(f"\nEmbedding Norms (should be ~1.0 for L2-normalized):")
        print(f"  ||q0||_2: {torch.norm(result.q0, p=2).item():.6f}")
        print(f"  ||q_enriched||_2: {torch.norm(result.q_enriched, p=2).item():.6f}")
        
        print("\n✓ Query enrichment successful!")
        
    except Exception as e:
        print(f"\n✗ Error during enrichment: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    print("\n" + "=" * 70)
    print("Demo Complete")
    print("=" * 70)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
