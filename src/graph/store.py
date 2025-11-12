"""
Phase 4 Graph Storage
Serialization and deserialization utilities for multimodal graphs.
"""

# === PHASE4: STORE HEADER (do not remove) ===
from typing import Dict
import json
import os
import tempfile
from pathlib import Path
import torch
from torch_geometric.data import HeteroData


# === PHASE4: SAVE_GRAPH ===
def save_graph(g: HeteroData, out_dir: str) -> None:
    """
    Save tensors to out_dir/{x_*.pt, eidx_*.pt, w_*.pt, maps.json}. Overwrite atomically.
    
    Args:
        g: HeteroData graph object to save
        out_dir: Output directory path
        
    Saves:
        - x_image.pt: Image node features
        - x_caption.pt: Caption node features
        - eidx_image_sem.pt: Image-image semantic edges
        - w_image_sem.pt: Image-image edge weights
        - eidx_caption_sem.pt: Caption-caption semantic edges
        - w_caption_sem.pt: Caption-caption edge weights
        - eidx_caption_image_paired.pt: Caption-image paired edges
        - eidx_caption_cooccur.pt: Caption-caption cooccur edges (if present)
        - w_caption_cooccur.pt: Caption-caption cooccur edge weights (if present)
        - maps.json: Node ID mappings
    """
    # Create output directory if needed
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    def _atomic_save_tensor(tensor: torch.Tensor, target_path: Path) -> None:
        """Save tensor atomically via temporary file."""
        with tempfile.NamedTemporaryFile(
            mode='wb',
            dir=target_path.parent,
            delete=False,
            suffix='.tmp'
        ) as tmp_file:
            tmp_path = Path(tmp_file.name)
            try:
                torch.save(tensor, tmp_path)
                tmp_path.replace(target_path)
            except Exception:
                tmp_path.unlink(missing_ok=True)
                raise
    
    # Save node features atomically
    _atomic_save_tensor(g['image'].x, out_path / 'x_image.pt')
    _atomic_save_tensor(g['caption'].x, out_path / 'x_caption.pt')
    
    # Save semantic edges (image ↔ image) atomically
    _atomic_save_tensor(
        g['image', 'sem_sim', 'image'].edge_index,
        out_path / 'eidx_image_sem.pt'
    )
    _atomic_save_tensor(
        g['image', 'sem_sim', 'image'].edge_weight,
        out_path / 'w_image_sem.pt'
    )
    
    # Save semantic edges (caption ↔ caption) atomically
    _atomic_save_tensor(
        g['caption', 'sem_sim', 'caption'].edge_index,
        out_path / 'eidx_caption_sem.pt'
    )
    _atomic_save_tensor(
        g['caption', 'sem_sim', 'caption'].edge_weight,
        out_path / 'w_caption_sem.pt'
    )
    
    # Validate and save paired edges (caption → image) atomically
    eidx_ci = g['caption', 'paired_with', 'image'].edge_index
    if eidx_ci.dtype != torch.long:
        raise ValueError(f"eidx_caption_image_paired must be torch.long, got {eidx_ci.dtype}")
    _atomic_save_tensor(eidx_ci, out_path / 'eidx_caption_image_paired.pt')
    
    # Save cooccur edges (caption ↔ caption) if present and non-empty
    cooccur_key = ('caption', 'cooccur', 'caption')
    if cooccur_key in g.edge_types:
        cooccur_store = g[cooccur_key]
        if hasattr(cooccur_store, 'edge_index') and cooccur_store.edge_index is not None:
            eidx_cc_co = cooccur_store.edge_index
            if eidx_cc_co.shape[1] > 0:
                # Validate dtype
                if eidx_cc_co.dtype != torch.long:
                    raise ValueError(f"eidx_caption_cooccur must be torch.long, got {eidx_cc_co.dtype}")
                _atomic_save_tensor(eidx_cc_co, out_path / 'eidx_caption_cooccur.pt')
                
                # Save cooccur edge weights if present
                if hasattr(cooccur_store, 'edge_weight') and cooccur_store.edge_weight is not None:
                    _atomic_save_tensor(cooccur_store.edge_weight, out_path / 'w_caption_cooccur.pt')
    
    # Retrieve maps using attribute-safe access
    nid_maps_raw = getattr(g, '_nid_maps', {})
    id_maps_raw = getattr(g, '_id_maps', {})
    
    # Normalize nid_maps to JSON-safe format: {node_type: {str(external_id): int_index}}
    nid_maps_json = {}
    for node_type, mapping in nid_maps_raw.items():
        nid_maps_json[node_type] = {str(ext_id): int(idx) for ext_id, idx in mapping.items()}
    
    # Normalize id_maps to JSON-safe format: {node_type: {str(int_index): external_id}}
    id_maps_json = {}
    for node_type, mapping in id_maps_raw.items():
        id_maps_json[node_type] = {str(int(idx)): str(ext_id) for idx, ext_id in mapping.items()}
    
    maps_data_serializable = {
        'nid_maps': nid_maps_json,
        'id_maps': id_maps_json
    }
    
    # Atomic write: write to temp, then rename
    with tempfile.NamedTemporaryFile(
        mode='w',
        dir=out_path,
        delete=False,
        suffix='.tmp',
        encoding='utf-8'
    ) as tmp_file:
        tmp_path = Path(tmp_file.name)
        try:
            json.dump(maps_data_serializable, tmp_file, indent=2)
            tmp_file.flush()
            os.fsync(tmp_file.fileno())
            tmp_path.replace(out_path / 'maps.json')
        except Exception:
            tmp_path.unlink(missing_ok=True)
            raise
    
    print(f"[graph] Saved graph to {out_dir}")
    print(f"  - Image nodes: {g['image'].x.shape[0]}")
    print(f"  - Caption nodes: {g['caption'].x.shape[0]}")
    print(f"  - Image-image edges: {g['image', 'sem_sim', 'image'].edge_index.shape[1]}")
    print(f"  - Caption-caption semantic edges: {g['caption', 'sem_sim', 'caption'].edge_index.shape[1]}")
    print(f"  - Caption-image paired edges: {g['caption', 'paired_with', 'image'].edge_index.shape[1]}")
    if cooccur_key in g.edge_types and hasattr(g[cooccur_key], 'edge_index') and g[cooccur_key].edge_index is not None and g[cooccur_key].edge_index.shape[1] > 0:
        print(f"  - Caption-caption cooccur edges: {g[cooccur_key].edge_index.shape[1]}")


# === PHASE4: LOAD_GRAPH ===
def load_graph(out_dir: str) -> HeteroData:
    """
    Load tensors & maps from out_dir and validate shapes/dtypes. Return HeteroData.
    
    Args:
        out_dir: Directory containing saved graph files
        
    Returns:
        HeteroData object with loaded graph
        
    Raises:
        FileNotFoundError: If required files are missing
        ValueError: If shapes or dtypes are incompatible
    """
    out_path = Path(out_dir)
    
    # Check required files exist
    required_files = [
        'x_image.pt',
        'x_caption.pt',
        'eidx_image_sem.pt',
        'w_image_sem.pt',
        'eidx_caption_sem.pt',
        'w_caption_sem.pt',
        'eidx_caption_image_paired.pt',
        'maps.json'
    ]
    
    missing = [f for f in required_files if not (out_path / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing required graph files in {out_dir}: {missing}"
        )
    
    # Load node features
    x_image = torch.load(out_path / 'x_image.pt')
    x_caption = torch.load(out_path / 'x_caption.pt')
    
    # Validate node feature shapes and dtypes
    if x_image.dtype != torch.float32:
        raise ValueError(
            f"x_image must be float32, got {x_image.dtype}"
        )
    if x_caption.dtype != torch.float32:
        raise ValueError(
            f"x_caption must be float32, got {x_caption.dtype}"
        )
    
    if x_image.ndim != 2 or x_caption.ndim != 2:
        raise ValueError(
            f"Node features must be 2D, got x_image: {x_image.shape}, x_caption: {x_caption.shape}"
        )
    
    if x_image.shape[1] != x_caption.shape[1]:
        raise ValueError(
            f"Feature dimensions must match, got x_image: {x_image.shape[1]}, x_caption: {x_caption.shape[1]}"
        )
    
    if x_image.shape[1] != 512:
        raise ValueError(
            f"Feature dimension must be 512, got x_image: {x_image.shape[1]}, x_caption: {x_caption.shape[1]}"
        )
    
    # Load semantic edges (image ↔ image)
    eidx_ii = torch.load(out_path / 'eidx_image_sem.pt')
    w_ii = torch.load(out_path / 'w_image_sem.pt')
    
    # Validate image-image edges
    if eidx_ii.dtype != torch.long:
        raise ValueError(f"Edge indices must be long, got {eidx_ii.dtype}")
    if w_ii.dtype != torch.float16:
        raise ValueError(f"Edge weights must be float16, got {w_ii.dtype}")
    if eidx_ii.shape[0] != 2:
        raise ValueError(f"Edge index must be [2, E], got {eidx_ii.shape}")
    if eidx_ii.shape[1] != w_ii.shape[0]:
        raise ValueError(
            f"Edge index and weight sizes must match, got {eidx_ii.shape[1]} vs {w_ii.shape[0]}"
        )
    
    # Load semantic edges (caption ↔ caption)
    eidx_cc = torch.load(out_path / 'eidx_caption_sem.pt')
    w_cc = torch.load(out_path / 'w_caption_sem.pt')
    
    # Validate caption-caption edges
    if eidx_cc.dtype != torch.long:
        raise ValueError(f"Edge indices must be long, got {eidx_cc.dtype}")
    if w_cc.dtype != torch.float16:
        raise ValueError(f"Edge weights must be float16, got {w_cc.dtype}")
    if eidx_cc.shape[0] != 2:
        raise ValueError(f"Edge index must be [2, E], got {eidx_cc.shape}")
    if eidx_cc.shape[1] != w_cc.shape[0]:
        raise ValueError(
            f"Edge index and weight sizes must match, got {eidx_cc.shape[1]} vs {w_cc.shape[0]}"
        )
    
    # Load paired edges (caption → image)
    eidx_ci = torch.load(out_path / 'eidx_caption_image_paired.pt')
    
    # Validate paired edges
    if eidx_ci.dtype != torch.long:
        raise ValueError(f"Edge indices must be long, got {eidx_ci.dtype}")
    if eidx_ci.shape[0] != 2:
        raise ValueError(f"Edge index must be [2, E], got {eidx_ci.shape}")
    
    # Load cooccur edges if present (optional)
    eidx_cc_co = None
    w_cc_co = None
    cooccur_file = out_path / 'eidx_caption_cooccur.pt'
    if cooccur_file.exists():
        eidx_cc_co = torch.load(cooccur_file)
        if eidx_cc_co.dtype != torch.long:
            raise ValueError(f"Edge indices must be long, got {eidx_cc_co.dtype}")
        if eidx_cc_co.shape[0] != 2:
            raise ValueError(f"Edge index must be [2, E], got {eidx_cc_co.shape}")
        
        # Load cooccur weights if present
        cooccur_weight_file = out_path / 'w_caption_cooccur.pt'
        if cooccur_weight_file.exists():
            w_cc_co = torch.load(cooccur_weight_file)
            if w_cc_co.dtype != torch.float16:
                raise ValueError(f"Edge weights must be float16, got {w_cc_co.dtype}")
            if w_cc_co.shape[0] != eidx_cc_co.shape[1]:
                raise ValueError(
                    f"Edge index and weight sizes must match, got {eidx_cc_co.shape[1]} vs {w_cc_co.shape[0]}"
                )
    
    # Load maps
    with open(out_path / 'maps.json', 'r', encoding='utf-8') as f:
        maps_data = json.load(f)
    
    # Convert string keys back to integers for id_maps
    id_maps = {
        node_type: {int(k): v for k, v in id_map.items()}
        for node_type, id_map in maps_data['id_maps'].items()
    }
    
    # Create HeteroData object
    data = HeteroData()
    
    # Store node features
    data['image'].x = x_image
    data['caption'].x = x_caption
    
    # Store semantic edges (image ↔ image)
    data['image', 'sem_sim', 'image'].edge_index = eidx_ii
    data['image', 'sem_sim', 'image'].edge_weight = w_ii
    
    # Store semantic edges (caption ↔ caption)
    data['caption', 'sem_sim', 'caption'].edge_index = eidx_cc
    data['caption', 'sem_sim', 'caption'].edge_weight = w_cc
    
    # Store paired edges (caption → image)
    data['caption', 'paired_with', 'image'].edge_index = eidx_ci
    
    # Store cooccur edges if present
    if eidx_cc_co is not None and eidx_cc_co.shape[1] > 0:
        data['caption', 'cooccur', 'caption'].edge_index = eidx_cc_co
        if w_cc_co is not None:
            data['caption', 'cooccur', 'caption'].edge_weight = w_cc_co
    
    # Store maps
    data['_nid_maps'] = maps_data['nid_maps']
    data['_id_maps'] = id_maps
    
    print(f"[graph] Loaded graph from {out_dir}")
    print(f"  - Image nodes: {data['image'].x.shape[0]}")
    print(f"  - Caption nodes: {data['caption'].x.shape[0]}")
    
    return data
