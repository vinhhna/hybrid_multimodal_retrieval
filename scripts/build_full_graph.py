"""
Phase 4: Full Graph Build Script for Flickr30K
Builds multimodal knowledge graph with semantic and co-occurrence edges.

This script orchestrates the full build process:
1. Load dataset and embeddings
2. Build semantic edges (image-image, caption-caption)
3. Build co-occurrence edges (caption-image, caption-caption)
4. Assemble HeteroData graph
5. Save artifacts and validate

Usage:
    python scripts/build_full_graph.py
"""

import torch
import numpy as np
import time
import os
import json
import sys
from pathlib import Path
# Try to import resource (Unix-only), otherwise set to None
try:
    import resource
except ImportError:
    resource = None
from typing import List, Dict, Tuple

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.flickr30k import Flickr30KDataset
from src.graph.build import (
    build_semantic_edges,
    build_cooccurrence_edges,
    assemble_hetero_graph,
    l2_normalize
)
from src.graph.store import save_graph_artifacts, load_graph_artifacts


# === Configuration ===
# Load from graph_config.yaml or use defaults
def load_config() -> Dict:
    """Load configuration from YAML or use defaults."""
    config_path = Path(__file__).parent.parent / "configs" / "graph_config.yaml"
    
    if config_path.exists():
        try:
            import yaml
            with open(config_path, 'r') as f:
                cfg = yaml.safe_load(f)
            print(f"✓ Loaded config from {config_path}")
            return cfg
        except Exception as e:
            print(f"⚠ Failed to load YAML config: {e}")
            print("  Using defaults...")
    else:
        print(f"⚠ Config file not found: {config_path}")
        print("  Using defaults...")
    
    # Default configuration
    return {
        'dataset': {
            'data_root': 'data',
            'images_dir': 'data/images',
            'captions_csv': 'data/results.csv'
        },
        'clip': {'dim': 512},
        'graph': {
            'k_sem': 16,
            'degree_cap': 16,
            'edge_dtype': 'fp16'
        }
    }


# === Helper Functions ===
def build_maps(image_ids: List[str], caption_data: List[Dict]) -> Tuple[Dict, Dict, Dict, Dict]:
    """
    Create all necessary ID/index maps.
    
    Args:
        image_ids: List of image IDs (from embeddings)
        caption_data: List of caption dicts with 'image_name' and 'caption'
    
    Returns:
        image_id_to_idx, image_ids_for_caption, captions_by_image, caption_id_to_idx
    """
    # Image ID to index
    image_id_to_idx = {img_id: i for i, img_id in enumerate(image_ids)}
    
    # Caption to image mapping
    image_ids_for_caption = {}
    captions_by_image = {}
    caption_ids = []
    
    for i, item in enumerate(caption_data):
        img_id = item['image_name']
        caption = item['caption']
        caption_id = f"{img_id}_cap{i % 5}"  # Generate caption ID
        
        caption_ids.append(caption_id)
        image_ids_for_caption[caption_id] = img_id
        
        if img_id not in captions_by_image:
            captions_by_image[img_id] = []
        captions_by_image[img_id].append(caption_id)
    
    caption_id_to_idx = {cap_id: i for i, cap_id in enumerate(caption_ids)}
    
    return image_id_to_idx, image_ids_for_caption, captions_by_image, caption_id_to_idx


def get_memory_usage_mb() -> float:
    """Get current memory usage in MB (Cross-platform safe)."""
    if resource:
        # Linux/Unix
        try:
            usage = resource.getrusage(resource.RUSAGE_SELF)
            return usage.ru_maxrss / 1024 
        except:
            return -1.0
    else:
        # Windows fallback (requires psutil, or just return -1 if not strictly needed)
        try:
            import psutil
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / 1024 / 1024  # Bytes to MB
        except ImportError:
            return -1.0


def main():
    """Main build orchestration."""
    print("="*60)
    print("PHASE 4: Full Graph Build (Flickr30K)")
    print("="*60)
    
    start_time = time.time()
    
    # 1. Load Config
    print("\n[1/8] Loading configuration...")
    config = load_config()
    k_sem = config['graph']['k_sem']
    degree_cap = config['graph']['degree_cap']
    print(f"  k_sem={k_sem}, degree_cap={degree_cap}")
    
    # 2. Setup paths
    data_root = Path(config['dataset'].get('data_root', 'data'))
    embed_dir = data_root / "embeddings"
    graph_dir = data_root / "graph"
    
    image_embed_path = embed_dir / "image_embeddings.npy"
    text_embed_path = embed_dir / "text_embeddings.npy"
    image_meta_path = embed_dir / "image_embeddings.json"
    text_meta_path = embed_dir / "text_embeddings.json"
    
    # Validate paths
    if not image_embed_path.exists():
        raise FileNotFoundError(f"Image embeddings not found: {image_embed_path}")
    if not text_embed_path.exists():
        raise FileNotFoundError(f"Text embeddings not found: {text_embed_path}")
    if not image_meta_path.exists():
        raise FileNotFoundError(f"Image metadata not found: {image_meta_path}")
    if not text_meta_path.exists():
        raise FileNotFoundError(f"Text metadata not found: {text_meta_path}")
    
    # 3. Load Embeddings and Metadata
    print(f"\n[2/8] Loading embeddings from {embed_dir}...")
    t_load = time.time()
    
    image_embeds = np.load(image_embed_path)
    text_embeds = np.load(text_embed_path)
    
    with open(image_meta_path, 'r') as f:
        image_meta = json.load(f)
    with open(text_meta_path, 'r') as f:
        text_meta = json.load(f)
    
    image_ids = image_meta['image_names']
    caption_data = text_meta['caption_to_image_mapping']
    
    print(f"  ✓ Loaded {len(image_ids)} image embeddings (shape: {image_embeds.shape})")
    print(f"  ✓ Loaded {len(caption_data)} caption embeddings (shape: {text_embeds.shape})")
    print(f"  Time: {time.time() - t_load:.2f}s")
    
    # Verify L2 normalization
    if not image_meta.get('normalized', False):
        print("  ⚠ Image embeddings not normalized, applying L2 normalization...")
        image_embeds = l2_normalize(image_embeds)
    if not text_meta.get('normalized', False):
        print("  ⚠ Text embeddings not normalized, applying L2 normalization...")
        text_embeds = l2_normalize(text_embeds)
    
    # 4. Build Maps
    print("\n[3/8] Building ID mappings...")
    t_map = time.time()
    
    image_id_to_idx, image_ids_for_caption, captions_by_image, caption_id_to_idx = build_maps(
        image_ids, caption_data
    )
    
    # Build reverse maps
    idx_to_image_id = {i: img_id for img_id, i in image_id_to_idx.items()}
    idx_to_caption_id = {i: cap_id for cap_id, i in caption_id_to_idx.items()}
    
    # Package maps for storage
    nid_maps = {
        'image': image_id_to_idx,
        'caption': caption_id_to_idx
    }
    id_maps = {
        'image': idx_to_image_id,
        'caption': idx_to_caption_id
    }
    
    print(f"  ✓ Created mappings for {len(image_id_to_idx)} images, {len(caption_id_to_idx)} captions")
    print(f"  Time: {time.time() - t_map:.2f}s")
    
    # 5. Build Semantic Edges (Image)
    print(f"\n[4/8] Building IMAGE semantic edges (k={k_sem}, cap={degree_cap})...")
    t_sem_img = time.time()
    
    img_sem_idx, img_sem_w = build_semantic_edges(
        node_type='image',
        X=image_embeds,
        k_sem=k_sem,
        degree_cap=degree_cap,
        chunk_size=8192
    )
    
    print(f"  ✓ Built {img_sem_idx.shape[1]} image-image edges")
    print(f"  Time: {time.time() - t_sem_img:.2f}s")
    
    # 6. Build Semantic Edges (Caption)
    print(f"\n[5/8] Building CAPTION semantic edges (k={k_sem}, cap={degree_cap})...")
    t_sem_cap = time.time()
    
    cap_sem_idx, cap_sem_w = build_semantic_edges(
        node_type='caption',
        X=text_embeds,
        k_sem=k_sem,
        degree_cap=degree_cap,
        chunk_size=8192
    )
    
    print(f"  ✓ Built {cap_sem_idx.shape[1]} caption-caption semantic edges")
    print(f"  Time: {time.time() - t_sem_cap:.2f}s")
    
    # 7. Build Co-occurrence Edges
    print("\n[6/8] Building co-occurrence edges...")
    t_cooc = time.time()
    
    caption_ids_ordered = [idx_to_caption_id[i] for i in range(len(caption_id_to_idx))]
    
    cooccur_result = build_cooccurrence_edges(
        caption_ids=caption_ids_ordered,
        image_ids_for_caption=image_ids_for_caption,
        captions_by_image=captions_by_image,
        image_id_to_idx=image_id_to_idx,
        use_caption_caption=True
    )
    
    edge_index_ci = cooccur_result.get('edge_index_ci', torch.zeros((2, 0), dtype=torch.long))
    edge_index_cc_co = cooccur_result.get('edge_index_cc', torch.zeros((2, 0), dtype=torch.long))
    
    print(f"  ✓ Built {edge_index_ci.shape[1]} caption-image paired edges")
    print(f"  ✓ Built {edge_index_cc_co.shape[1]} caption-caption cooccur edges")
    print(f"  Time: {time.time() - t_cooc:.2f}s")
    
    # 8. Assemble HeteroData Graph
    print("\n[7/8] Assembling HeteroData object...")
    t_assemble = time.time()
    
    g = assemble_hetero_graph(
        x_image=image_embeds,
        x_caption=text_embeds,
        eidx_ii=img_sem_idx,
        w_ii=img_sem_w,
        eidx_cc=cap_sem_idx,
        w_cc=cap_sem_w,
        eidx_ci=edge_index_ci,
        eidx_cc_co=edge_index_cc_co if edge_index_cc_co.shape[1] > 0 else None,
        nid_maps=nid_maps,
        id_maps=id_maps,
        enforce_normalized=False  # Already normalized
    )
    
    print(f"  Time: {time.time() - t_assemble:.2f}s")
    print("\n" + "="*60)
    print("Graph Summary:")
    print("="*60)
    print(g)
    print("="*60)
    
    # 9. Save Artifacts
    print(f"\n[8/8] Saving artifacts to {graph_dir}...")
    t_save = time.time()
    
    graph_dir.mkdir(parents=True, exist_ok=True)
    
    # Package maps for save_graph_artifacts
    maps_dict = {
        'nid_maps': nid_maps,
        'id_maps': id_maps
    }
    save_graph_artifacts(g, maps_dict, str(graph_dir))
    
    print(f"  Time: {time.time() - t_save:.2f}s")
    
    # 10. Acceptance Test: Reload
    print("\n" + "="*60)
    print("ACCEPTANCE TEST: Reload Validation")
    print("="*60)
    
    try:
        print("Reloading graph from disk...")
        g_reloaded, maps_reloaded = load_graph_artifacts(str(graph_dir))
        
        # Validate shapes
        assert g_reloaded['image'].x.shape == g['image'].x.shape, \
            f"Image features mismatch: {g_reloaded['image'].x.shape} vs {g['image'].x.shape}"
        assert g_reloaded['caption'].x.shape == g['caption'].x.shape, \
            f"Caption features mismatch"
        assert g_reloaded['image', 'sem_sim', 'image'].edge_index.shape == img_sem_idx.shape, \
            f"Image edges mismatch"
        assert g_reloaded['caption', 'sem_sim', 'caption'].edge_index.shape == cap_sem_idx.shape, \
            f"Caption edges mismatch"
        
        # Validate maps
        assert maps_reloaded['nid_maps']['image'] == nid_maps['image'], "Image ID maps mismatch"
        assert maps_reloaded['nid_maps']['caption'] == nid_maps['caption'], "Caption ID maps mismatch"
        
        print("✅ Reload test PASSED")
        print("  - All shapes match")
        print("  - All maps match")
        
    except Exception as e:
        print(f"❌ Reload test FAILED: {e}")
        raise
    
    # 11. Final Summary
    print("\n" + "="*60)
    print("BUILD COMPLETE")
    print("="*60)
    
    total_time = time.time() - start_time
    mem_usage = get_memory_usage_mb()
    
    print(f"\n📊 Performance Metrics:")
    print(f"  Total build time: {total_time:.2f}s ({total_time/60:.1f}m)")
    print(f"  Memory usage: {mem_usage:.1f} MB" if mem_usage > 0 else "  Memory usage: N/A")
    print(f"\n📈 Graph Statistics:")
    print(f"  Image nodes: {g['image'].x.shape[0]:,}")
    print(f"  Caption nodes: {g['caption'].x.shape[0]:,}")
    print(f"  Image-image edges: {img_sem_idx.shape[1]:,}")
    print(f"  Caption-caption semantic edges: {cap_sem_idx.shape[1]:,}")
    print(f"  Caption-image paired edges: {edge_index_ci.shape[1]:,}")
    print(f"  Caption-caption cooccur edges: {edge_index_cc_co.shape[1]:,}")
    print(f"\n💾 Output:")
    print(f"  Graph directory: {graph_dir}")
    print(f"  Artifacts: x_image.pt, x_caption.pt, eidx_*.pt, w_*.pt, maps.json")
    print("\n✅ Full graph build completed successfully!")
    print("="*60)
    
    # Spot-check degree distributions
    print("\n📊 Degree Distribution Spot-Check:")
    
    # Image node out-degrees
    img_out_deg = torch.bincount(img_sem_idx[0])
    print(f"  Image nodes:")
    print(f"    Mean out-degree: {img_out_deg.float().mean():.2f}")
    print(f"    Max out-degree: {img_out_deg.max().item()}")
    print(f"    Min out-degree: {img_out_deg.min().item()}")
    
    # Caption node out-degrees (semantic only)
    cap_out_deg = torch.bincount(cap_sem_idx[0])
    print(f"  Caption nodes (semantic):")
    print(f"    Mean out-degree: {cap_out_deg.float().mean():.2f}")
    print(f"    Max out-degree: {cap_out_deg.max().item()}")
    print(f"    Min out-degree: {cap_out_deg.min().item()}")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    main()
