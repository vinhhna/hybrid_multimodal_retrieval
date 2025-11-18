"""
Build a tiny (≈100 images) multimodal graph slice from Flickr30K and save to data/graph_tiny/.

This script:
- Loads CLIP embeddings (assumed precomputed) or stubs with random vectors for smoke test
- Ensures L2-normalization
- Builds semantic & cooccurrence edges with degree caps
- Saves and reloads to verify round-trip
"""

# === PHASE4: GRAPH BUILD TINY ===

import os
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch

# Add src to path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root / "src"))

from graph.build import (
    l2_normalize,
    build_semantic_edges,
    build_cooccurrence_edges,
    assemble_hetero_graph,
    GRAPH_DEFAULTS,
)
from graph.store import save_graph, load_graph


def load_or_generate_embeddings(
    embeddings_path: Path,
    num_items: int,
    dim: int = 512,
    use_random: bool = False,
    seed: int = 42,
) -> np.ndarray:
    """
    Load embeddings from disk or generate random ones for testing.
    
    Args:
        embeddings_path: Path to embeddings .npy file
        num_items: Number of items to load/generate
        dim: Embedding dimension
        use_random: Force random generation even if file exists
        seed: Random seed for reproducibility
        
    Returns:
        L2-normalized embeddings array of shape (num_items, dim)
    """
    if not use_random and embeddings_path.exists():
        print(f"[graph] Loading embeddings from {embeddings_path}")
        embeds = np.load(embeddings_path)
        embeds = embeds[:num_items]  # Take only first num_items
        print(f"  - Loaded {embeds.shape[0]} embeddings of dim {embeds.shape[1]}")
    else:
        print(f"[graph] Generating random embeddings (seed={seed})")
        np.random.seed(seed)
        embeds = np.random.randn(num_items, dim).astype(np.float32)
        print(f"  - Generated {embeds.shape[0]} embeddings of dim {embeds.shape[1]}")
    
    # Ensure L2-normalized
    embeds = l2_normalize(embeds)
    return embeds


def main():
    """Build tiny graph for smoke testing."""
    
    # === Configuration ===
    NUM_IMAGES = 100  # Number of images to process
    K_SEM = GRAPH_DEFAULTS["k_sem"]
    DEGREE_CAP = GRAPH_DEFAULTS["degree_cap"]
    VECTOR_DIM = GRAPH_DEFAULTS["vector_dim"]
    
    # Set seed for reproducibility if FAST_SEED env var is set
    if "FAST_SEED" in os.environ:
        seed = int(os.environ["FAST_SEED"])
        np.random.seed(seed)
        torch.manual_seed(seed)
        print(f"[graph] Using FAST_SEED={seed}")
    else:
        seed = 42
    
    # === Paths ===
    repo_root = Path(__file__).parent.parent
    data_root = Path("/kaggle/input/flickr30k/data")
    
    # Use local data if Kaggle path doesn't exist
    if not data_root.exists():
        data_root = repo_root / "data"
        print(f"[graph] Kaggle data not found, using local: {data_root}")
    
    captions_csv = data_root / "results.csv"
    embeddings_dir = data_root / "embeddings"
    output_dir = repo_root / "data" / "graph_tiny"
    
    print(f"\n=== Phase 4 Graph Builder (Tiny) ===")
    print(f"Processing {NUM_IMAGES} images")
    print(f"k_sem={K_SEM}, degree_cap={DEGREE_CAP}")
    print(f"Data root: {data_root}")
    print(f"Output: {output_dir}")
    
    # === Load captions ===
    t0 = time.time()
    print(f"\n[1/7] Loading captions from {captions_csv}")
    
    if not captions_csv.exists():
        print(f"ERROR: Captions file not found: {captions_csv}")
        print("Please ensure Flickr30K data is available or use local data/results.csv")
        return 1
    
    df = pd.read_csv(captions_csv, sep='|', engine='python')
    df.columns = df.columns.str.strip()
    df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
    df.columns = ['image_name', 'comment_number', 'caption']
    
    # Take only first NUM_IMAGES
    unique_images = df['image_name'].unique()[:NUM_IMAGES]
    df_tiny = df[df['image_name'].isin(unique_images)].copy()
    
    num_images = len(unique_images)
    num_captions = len(df_tiny)
    
    print(f"  ✓ Loaded {num_captions} captions for {num_images} images")
    print(f"    Time: {(time.time() - t0)*1000:.1f} ms")
    
    # === Build node maps ===
    t0 = time.time()
    print(f"\n[2/7] Building node ID maps")
    
    # Image nodes
    image_ids = unique_images.tolist()
    nid_map_image = {img_id: i for i, img_id in enumerate(image_ids)}
    id_map_image = {i: img_id for img_id, i in nid_map_image.items()}
    
    # Caption nodes (use image_name + comment_number as caption_id)
    df_tiny['caption_id'] = df_tiny['image_name'] + '#' + df_tiny['comment_number'].astype(str)
    caption_ids = df_tiny['caption_id'].tolist()
    nid_map_caption = {cap_id: i for i, cap_id in enumerate(caption_ids)}
    id_map_caption = {i: cap_id for cap_id, i in nid_map_caption.items()}
    
    # Build lookup dicts
    image_ids_for_caption = dict(zip(df_tiny['caption_id'], df_tiny['image_name']))
    captions_by_image = df_tiny.groupby('image_name')['caption_id'].apply(list).to_dict()
    
    nid_maps = {'image': nid_map_image, 'caption': nid_map_caption}
    id_maps = {'image': id_map_image, 'caption': id_map_caption}
    
    print(f"  ✓ Built maps: {num_images} images, {num_captions} captions")
    print(f"    Time: {(time.time() - t0)*1000:.1f} ms")
    
    # === Load or generate embeddings ===
    t0 = time.time()
    print(f"\n[3/7] Loading embeddings")
    
    # Try to load precomputed embeddings
    img_emb_path = embeddings_dir / "image_embeddings.npy"
    txt_emb_path = embeddings_dir / "text_embeddings.npy"
    
    use_random = not (img_emb_path.exists() and txt_emb_path.exists())
    if use_random:
        print("  ⚠ Precomputed embeddings not found, using random vectors for smoke test")
    
    x_image = load_or_generate_embeddings(
        img_emb_path, num_images, VECTOR_DIM, use_random, seed
    )
    
    x_caption = load_or_generate_embeddings(
        txt_emb_path, num_captions, VECTOR_DIM, use_random, seed + 1
    )
    
    print(f"  ✓ Embeddings ready: image {x_image.shape}, caption {x_caption.shape}")
    print(f"    Time: {(time.time() - t0)*1000:.1f} ms")
    
    # === Build semantic edges (image ↔ image) ===
    t0 = time.time()
    print(f"\n[4/7] Building semantic edges (image ↔ image)")
    
    eidx_ii, w_ii = build_semantic_edges(
        node_type="image",
        X=x_image,
        k_sem=K_SEM,
        degree_cap=DEGREE_CAP,
    )
    
    print(f"  ✓ Built {eidx_ii.shape[1]} image-image edges")
    print(f"    Time: {(time.time() - t0)*1000:.1f} ms")
    
    # === Build semantic edges (caption ↔ caption) ===
    t0 = time.time()
    print(f"\n[5/7] Building semantic edges (caption ↔ caption)")
    
    eidx_cc, w_cc = build_semantic_edges(
        node_type="caption",
        X=x_caption,
        k_sem=K_SEM,
        degree_cap=DEGREE_CAP,
    )
    
    print(f"  ✓ Built {eidx_cc.shape[1]} caption-caption semantic edges")
    print(f"    Time: {(time.time() - t0)*1000:.1f} ms")
    
    # === Build co-occurrence edges ===
    t0 = time.time()
    print(f"\n[6/7] Building co-occurrence edges")
    
    # First build the actual paired edges with correct image indices
    src_ci, tgt_ci = [], []
    for cap_id in caption_ids:
        img_id = image_ids_for_caption[cap_id]
        src_ci.append(nid_map_caption[cap_id])
        tgt_ci.append(nid_map_image[img_id])
    
    eidx_ci = torch.tensor([src_ci, tgt_ci], dtype=torch.long)
    
    # Build caption-caption cooccur edges
    cooccur_edges = build_cooccurrence_edges(
        caption_ids=caption_ids,
        image_ids_for_caption=image_ids_for_caption,
        captions_by_image=captions_by_image,
        use_caption_caption=True,
    )
    
    eidx_cc_co = cooccur_edges.get('edge_index_cc', None)
    
    print(f"  ✓ Built {eidx_ci.shape[1]} caption-image paired edges")
    if eidx_cc_co is not None:
        print(f"  ✓ Built {eidx_cc_co.shape[1]} caption-caption cooccur edges")
    print(f"    Time: {(time.time() - t0)*1000:.1f} ms")
    
    # === Assemble HeteroData ===
    t0 = time.time()
    print(f"\n[7/7] Assembling HeteroData graph")
    
    graph = assemble_hetero_graph(
        x_image=x_image,
        x_caption=x_caption,
        eidx_ii=eidx_ii,
        w_ii=w_ii,
        eidx_cc=eidx_cc,
        w_cc=w_cc,
        eidx_ci=eidx_ci,
        eidx_cc_co=eidx_cc_co,
        nid_maps=nid_maps,
        id_maps=id_maps,
    )

    # Attach raw caption texts for downstream enrichment (Phase 4 search)
    graph["caption"].text = df_tiny["caption"].tolist()
    
    print(f"  ✓ Graph assembled")
    print(f"    Time: {(time.time() - t0)*1000:.1f} ms")
    
    # === Save graph ===
    t0 = time.time()
    print(f"\n=== Saving graph to {output_dir} ===")
    
    save_graph(graph, str(output_dir))
    print(f"  Time: {(time.time() - t0)*1000:.1f} ms")
    
    # === Reload and validate ===
    t0 = time.time()
    print(f"\n=== Reloading graph for validation ===")
    
    graph_reloaded = load_graph(str(output_dir))
    print(f"  Time: {(time.time() - t0)*1000:.1f} ms")
    
    # === Validate degree caps ===
    print(f"\n=== Validating degree caps ===")
    
    # Check image-image semantic edges
    eidx_ii_check = graph_reloaded['image', 'sem_sim', 'image'].edge_index
    src_ii = eidx_ii_check[0].numpy()
    degrees_ii = np.bincount(src_ii, minlength=num_images)
    max_deg_ii = degrees_ii.max()
    
    print(f"  Image-image semantic edges:")
    print(f"    Max out-degree: {max_deg_ii} (cap={DEGREE_CAP})")
    
    if max_deg_ii > DEGREE_CAP:
        print(f"    ✗ FAILED: Max degree {max_deg_ii} exceeds cap {DEGREE_CAP}")
        return 1
    else:
        print(f"    ✓ PASSED: All degrees ≤ {DEGREE_CAP}")
    
    # Check caption-caption semantic edges
    eidx_cc_check = graph_reloaded['caption', 'sem_sim', 'caption'].edge_index
    src_cc = eidx_cc_check[0].numpy()
    degrees_cc = np.bincount(src_cc, minlength=num_captions)
    max_deg_cc = degrees_cc.max()
    
    print(f"  Caption-caption semantic edges:")
    print(f"    Max out-degree: {max_deg_cc} (cap={DEGREE_CAP})")
    
    if max_deg_cc > DEGREE_CAP:
        print(f"    ✗ FAILED: Max degree {max_deg_cc} exceeds cap {DEGREE_CAP}")
        return 1
    else:
        print(f"    ✓ PASSED: All degrees ≤ {DEGREE_CAP}")
    
    # === Summary ===
    print(f"\n=== SUCCESS ===")
    print(f"Tiny graph built, saved, reloaded, and validated.")
    print(f"All semantic edges satisfy out-degree ≤ {DEGREE_CAP}.")
    print(f"Graph ready at: {output_dir}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
