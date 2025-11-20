"""
Phase 4 Graph Builder
Build multimodal knowledge graph with semantic and co-occurrence edges.
"""

# === PHASE4: BUILD HEADER (do not remove) ===
from typing import Dict, Tuple, Optional, List, Literal
import numpy as np
import torch
from torch import Tensor
from torch_geometric.data import HeteroData
from tqdm.auto import tqdm

# Try to import FAISS
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("[WARNING] FAISS not available. Falling back to numpy-based k-NN (slower).")

# === PHASE4: GRAPH DEFAULTS ===
GRAPH_DEFAULTS = {
    "k_sem": 16,
    "degree_cap": 16,
    "edge_dtype": "fp16",
    "vector_dim": 512
}


# === PHASE4: L2NORM ===
def l2_normalize(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Return row-wise L2-normalized float32 copy of X with shape (N, D).
    
    Args:
        X: Input array of shape (N, D)
        eps: Small epsilon to avoid division by zero
        
    Returns:
        L2-normalized array of shape (N, D) as float32
    """
    X = X.astype(np.float32, copy=False)
    norms = np.sqrt((X * X).sum(axis=1, keepdims=True))
    norms = np.maximum(norms, eps)
    return X / norms


# === PHASE4: SEMANTIC_EDGES ===
def build_semantic_edges(
    node_type: Literal["image", "caption"],
    X: np.ndarray,
    k_sem: int,
    degree_cap: int,
    chunk_size: int = 8192,
) -> Tuple[Tensor, Tensor]:
    """
    Compute chunked k-NN over **L2-normalized** CLIP features using FAISS IndexFlatIP.
    
    Args:
        node_type: Type of nodes ("image" or "caption")
        X: Node features array of shape (N, D), assumed L2-normalized
        k_sem: Number of semantic neighbors to find per node
        degree_cap: Maximum out-degree per source node (hard cap)
        chunk_size: Chunk size for memory-efficient computation
        
    Returns:
        edge_index: LongTensor [2, E] with (source, target) edges
        edge_weight: HalfTensor [E] with cosine sims in fp16
        
    Note:
        - Uses FAISS IndexFlatIP for efficient inner-product search (cosine on normalized data)
        - Excludes self-loops (similarity > 0.9999)
        - Enforces out-degree ≤ degree_cap per source node
        - Uses chunked computation to handle large graphs
        - Shows progress bar for chunked processing
    """
    N, D = X.shape
    
    # Ensure L2-normalized and float32 (required by FAISS)
    X_norm = l2_normalize(X).astype(np.float32)
    
    # Storage for edges
    all_src, all_tgt, all_weights = [], [], []
    
    if FAISS_AVAILABLE:
        # Use FAISS for efficient k-NN search
        # IndexFlatIP computes inner product (cosine similarity on normalized vectors)
        index = faiss.IndexFlatIP(D)
        index.add(X_norm)
        
        # Search in chunks with progress bar
        k_search = min(k_sem + 1, N)  # +1 to include self, then filter
        num_chunks = (N + chunk_size - 1) // chunk_size
        
        desc = f"Building {node_type} semantic edges"
        for chunk_start in tqdm(range(0, N, chunk_size), desc=desc, total=num_chunks):
            chunk_end = min(chunk_start + chunk_size, N)
            query = X_norm[chunk_start:chunk_end]
            
            # Search k_search nearest neighbors
            distances, indices = index.search(query, k_search)
            
            # Process results for this chunk
            for i in range(len(query)):
                global_i = chunk_start + i
                
                # Filter out self-loops (distance very close to 1.0 for normalized vectors)
                mask = indices[i] != global_i  # Exclude exact self-match
                mask &= distances[i] < 0.9999  # Also exclude near-duplicates
                
                valid_neighbors = indices[i][mask]
                valid_distances = distances[i][mask]
                
                # Apply degree cap: keep top degree_cap by distance
                if len(valid_neighbors) > degree_cap:
                    top_k_idx = np.argsort(-valid_distances)[:degree_cap]
                    valid_neighbors = valid_neighbors[top_k_idx]
                    valid_distances = valid_distances[top_k_idx]
                
                # Add edges
                if len(valid_neighbors) > 0:
                    all_src.extend([global_i] * len(valid_neighbors))
                    all_tgt.extend(valid_neighbors.tolist())
                    all_weights.extend(valid_distances.astype(np.float16).tolist())
    
    else:
        # Fallback to numpy-based approach (no FAISS)
        print(f"[{node_type}] Using numpy fallback for k-NN (slower)")
        
        num_chunks = (N + chunk_size - 1) // chunk_size
        desc = f"Building {node_type} semantic edges"
        
        for start in tqdm(range(0, N, chunk_size), desc=desc, total=num_chunks):
            stop = min(start + chunk_size, N)
            Q = X_norm[start:stop]  # (q, D)
            
            # Compute cosine similarity via dot product
            S = Q @ X_norm.T  # (q, N)
            
            # Exclude self-loops
            chunk_indices = np.arange(start, stop)
            S[np.arange(stop - start), chunk_indices] = -np.inf
            
            # Find top-k_sem neighbors
            k_actual = min(k_sem, N - 1)
            
            if k_actual > 0:
                # Get top-k indices using argpartition
                idx = np.argpartition(-S, min(k_actual - 1, S.shape[1] - 1), axis=1)[:, :k_actual]
                part = np.take_along_axis(S, idx, axis=1)
                
                # Apply degree cap
                keep_k = min(k_actual, degree_cap)
                
                # Sort to get actual top-k by score
                topk_idx = np.argsort(-part, axis=1)[:, :keep_k]
                nbrs = np.take_along_axis(idx, topk_idx, axis=1)
                sims = np.take_along_axis(part, topk_idx, axis=1)
                
                # Build edges for this chunk
                rows = np.repeat(np.arange(start, stop)[:, None], keep_k, axis=1).ravel()
                cols = nbrs.ravel()
                weights = sims.ravel()
                
                all_src.extend(rows.tolist())
                all_tgt.extend(cols.tolist())
                all_weights.extend(weights.astype(np.float16).tolist())
    
    # Convert to tensors
    if len(all_src) == 0:
        # Empty graph
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_weight = torch.zeros((0,), dtype=torch.float16)
    else:
        edge_index = torch.tensor(
            [all_src, all_tgt],
            dtype=torch.long
        )
        edge_weight = torch.tensor(all_weights, dtype=torch.float16)
    
    return edge_index, edge_weight


# === PHASE4: COOCCUR_EDGES ===
def build_cooccurrence_edges(
    caption_ids: List[str],
    image_ids_for_caption: Dict[str, str],
    captions_by_image: Dict[str, List[str]],
    image_id_to_idx: Optional[Dict[str, int]] = None,
    use_caption_caption: bool = True,
) -> Dict[str, Tensor]:
    """
    Construct co-occurrence edges with weight=1.0.
    
    Args:
        caption_ids: Ordered list of caption IDs (defines node indices)
        image_ids_for_caption: Map from caption_id to image_id
        captions_by_image: Map from image_id to list of caption_ids
        image_id_to_idx: Optional map from image_id to image node index. If None, skips building paired edges.
        use_caption_caption: Whether to include caption↔caption edges
        
    Returns:
        Dictionary with keys:
            'edge_index_ci': caption->image edges, LongTensor[2, E_ci] (only if image_id_to_idx provided)
            'edge_index_cc': caption->caption edges, LongTensor[2, E_cc] (only if use_caption_caption=True)
    
    Note:
        - All edges have implicit weight=1.0
        - Returns proper index tensors with dtype=torch.long
        - Skips captions whose image_id is not in image_id_to_idx (when provided)
        - Caption-caption edges form cliques for captions sharing the same image
    """
    # Build caption_id to index map
    cap_to_idx = {cap_id: i for i, cap_id in enumerate(caption_ids)}
    
    result = {}
    
    # Build caption->image edges (paired_with relation) only if image_id_to_idx provided
    if image_id_to_idx is not None:
        src_ci, tgt_ci = [], []
        skipped_count = 0
        
        print("[Co-occurrence] Building caption-image paired edges...")
        for cap_id in tqdm(caption_ids, desc="Caption-image pairs"):
            if cap_id in image_ids_for_caption:
                img_id = image_ids_for_caption[cap_id]
                if img_id in image_id_to_idx:
                    src_ci.append(cap_to_idx[cap_id])
                    tgt_ci.append(image_id_to_idx[img_id])
                else:
                    skipped_count += 1
        
        if skipped_count > 0:
            print(f"  Warning: skipped {skipped_count} captions with missing image_id")
        
        # Create edge_index_ci tensor
        if src_ci:
            edge_index_ci = torch.tensor([src_ci, tgt_ci], dtype=torch.long)
        else:
            edge_index_ci = torch.zeros((2, 0), dtype=torch.long)
        
        result['edge_index_ci'] = edge_index_ci
        print(f"  Built {edge_index_ci.shape[1]} caption-image edges")
    
    # Build caption->caption edges (cooccur relation)
    if use_caption_caption:
        src_cc, tgt_cc = [], []
        
        print("[Co-occurrence] Building caption-caption cooccur edges (clique expansion)...")
        for img_id, cap_list in tqdm(captions_by_image.items(), desc="Caption-caption cooccur"):
            # Connect all caption pairs within same image (clique expansion)
            valid_caps = [c for c in cap_list if c in cap_to_idx]
            
            for i in range(len(valid_caps)):
                for j in range(len(valid_caps)):
                    if i != j:  # Exclude self-loops
                        src_cc.append(cap_to_idx[valid_caps[i]])
                        tgt_cc.append(cap_to_idx[valid_caps[j]])
        
        # Only add edge_index_cc if there are pairs
        if src_cc:
            edge_index_cc = torch.tensor([src_cc, tgt_cc], dtype=torch.long)
            result['edge_index_cc'] = edge_index_cc
            print(f"  Built {edge_index_cc.shape[1]} caption-caption cooccur edges")
        else:
            print("  No caption-caption cooccur edges (images have single captions)")
    
    return result


# === PHASE4: ASSEMBLE_HETERO ===
def assemble_hetero_graph(
    x_image: np.ndarray,
    x_caption: np.ndarray,
    eidx_ii: Tensor,
    w_ii: Tensor,
    eidx_cc: Tensor,
    w_cc: Tensor,
    eidx_ci: Tensor,
    eidx_cc_co: Optional[Tensor],
    nid_maps: Dict[str, Dict[str, int]],
    id_maps: Dict[str, Dict[int, str]],
    *,
    enforce_normalized: bool = False,
) -> HeteroData:
    """
    Create a PyG HeteroData object with x_dict and edge_index_dict/edge_weight per type.
    
    Args:
        x_image: Image node features (N_img, D), **must be L2-normalized** float32
        x_caption: Caption node features (N_cap, D), **must be L2-normalized** float32
        eidx_ii: Image-image semantic edges [2, E_ii]
        w_ii: Image-image edge weights [E_ii] fp16
        eidx_cc: Caption-caption semantic edges [2, E_cc]
        w_cc: Caption-caption edge weights [E_cc] fp16
        eidx_ci: Caption-image paired edges [2, E_ci]
        eidx_cc_co: Optional caption-caption cooccur edges [2, E_cc_co]
        nid_maps: Node ID maps {"image": {image_id: idx}, "caption": {caption_id: idx}}
        id_maps: Reverse maps {"image": {idx: image_id}, "caption": {idx: caption_id}}
        enforce_normalized: If True, L2-normalize x_image and x_caption in-place (default: False)
        
    Returns:
        HeteroData object with all node features and edges
        
    Note:
        Input features x_image and x_caption must already be L2-normalized float32 arrays.
        Set enforce_normalized=True to apply normalization if needed (adds overhead).
    """
    # Validate shapes
    if x_image.ndim != 2:
        raise ValueError(f"x_image must be 2D, got shape {x_image.shape}")
    if x_caption.ndim != 2:
        raise ValueError(f"x_caption must be 2D, got shape {x_caption.shape}")
    
    if x_image.shape[1] != 512:
        raise ValueError(f"x_image must have 512 features, got {x_image.shape[1]}")
    if x_caption.shape[1] != 512:
        raise ValueError(f"x_caption must have 512 features, got {x_caption.shape[1]}")
    
    # Validate semantic edge dtypes and shapes
    if eidx_ii.dtype != torch.long:
        raise ValueError(f"eidx_ii must be torch.long, got {eidx_ii.dtype}")
    if eidx_cc.dtype != torch.long:
        raise ValueError(f"eidx_cc must be torch.long, got {eidx_cc.dtype}")
    
    if w_ii.dtype != torch.float16:
        raise ValueError(f"w_ii must be torch.float16, got {w_ii.dtype}")
    if w_cc.dtype != torch.float16:
        raise ValueError(f"w_cc must be torch.float16, got {w_cc.dtype}")
    
    if w_ii.numel() != eidx_ii.shape[1]:
        raise ValueError(f"w_ii length ({w_ii.numel()}) must match eidx_ii edge count ({eidx_ii.shape[1]})")
    if w_cc.numel() != eidx_cc.shape[1]:
        raise ValueError(f"w_cc length ({w_cc.numel()}) must match eidx_cc edge count ({eidx_cc.shape[1]})")
    
    # Validate paired edge dtype
    if eidx_ci.dtype != torch.long:
        raise ValueError(f"eidx_ci must be torch.long, got {eidx_ci.dtype}")
    
    # Validate optional cooccur edge dtype
    if eidx_cc_co is not None and eidx_cc_co.dtype != torch.long:
        raise ValueError(f"eidx_cc_co must be torch.long, got {eidx_cc_co.dtype}")
    
    # Optionally enforce normalization
    if enforce_normalized:
        x_image = l2_normalize(x_image)
        x_caption = l2_normalize(x_caption)
    
    # Initialize HeteroData
    data = HeteroData()
    
    # Store node features (ensure float32)
    data['image'].x = torch.from_numpy(x_image).float()
    data['caption'].x = torch.from_numpy(x_caption).float()
    
    # Store semantic edges (image ↔ image)
    data['image', 'sem_sim', 'image'].edge_index = eidx_ii
    data['image', 'sem_sim', 'image'].edge_weight = w_ii
    
    # Store semantic edges (caption ↔ caption)
    data['caption', 'sem_sim', 'caption'].edge_index = eidx_cc
    data['caption', 'sem_sim', 'caption'].edge_weight = w_cc
    
    # Store paired edges (caption → image)
    data['caption', 'paired_with', 'image'].edge_index = eidx_ci
    # Weight is implicitly 1.0, can be added if needed
    
    # Store cooccur edges (caption ↔ caption) if provided
    if eidx_cc_co is not None and eidx_cc_co.shape[1] > 0:
        data['caption', 'cooccur', 'caption'].edge_index = eidx_cc_co
        # Weight is implicitly 1.0
    
    # Store maps as metadata (not in PyG tensors, but accessible)
    data['_nid_maps'] = nid_maps
    data['_id_maps'] = id_maps
    
    return data
