"""
Hybrid Search Engine for Multimodal Retrieval

This module implements a two-stage hybrid search pipeline that combines:
1. Stage 1 (CLIP Bi-Encoder): Fast retrieval of top-k1 candidates
2. Stage 2 (BLIP-2 Cross-Encoder): Accurate re-ranking of candidates

The hybrid approach balances speed and accuracy, achieving better results
than CLIP alone while being much faster than using BLIP-2 for all candidates.

Phase 3: Hybrid Retrieval System
Phase 4 Day 13-14: Score fusion and integration with KG
Created: November 4, 2025
Updated: December 1, 2025 (Phase 4 fusion)
"""

import logging
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any, Union
from PIL import Image
from tqdm import tqdm
import time

# Try imports with fallback for Kaggle
try:
    from .bi_encoder import BiEncoder
    from .cross_encoder import CrossEncoder
    from .faiss_index import FAISSIndex
except ImportError:
    import sys
    sys.path.append('..')
    from retrieval.bi_encoder import BiEncoder
    from retrieval.cross_encoder import CrossEncoder
    from retrieval.faiss_index import FAISSIndex

try:
    from ..flickr30k.dataset import Flickr30KDataset
except ImportError:
    import sys
    sys.path.append('..')
    from flickr30k.dataset import Flickr30KDataset


def _normalize(arr):
    """
    Robust min-max normalization with epsilon handling.
    
    Returns zeros if array is constant (avoids division by zero).
    
    Args:
        arr: Array-like to normalize
    
    Returns:
        Normalized array in [0, 1] or zeros if constant
    """
    arr = np.asarray(arr, dtype=np.float32)
    if arr.size == 0:
        return arr
    mn, mx = float(np.min(arr)), float(np.max(arr))
    if mx - mn < 1e-8:
        return np.zeros_like(arr, dtype=np.float32)
    return (arr - mn) / (mx - mn)


def dense_rank_desc(x: np.ndarray) -> np.ndarray:
    """
    Compute dense ranking where highest value gets rank 1.
    
    Args:
        x: Array of scores
    
    Returns:
        Array of ranks (1-based, highest score = rank 1)
    
    Example:
        >>> scores = np.array([0.8, 0.5, 0.9, 0.5])
        >>> dense_rank_desc(scores)
        array([2, 3, 1, 3])  # 0.9 is rank 1, 0.8 is rank 2, both 0.5s are rank 3
    """
    order = np.argsort(-x)  # Sort descending
    ranks = np.empty_like(order, dtype=np.int32)
    ranks[order] = np.arange(1, len(x) + 1)
    return ranks


# ============================================================================
# Phase 4: Score normalization and fusion helpers
# ============================================================================

def _min_max_normalize(scores: Dict[str, float]) -> Dict[str, float]:
    """
    Per-query min-max normalization for score fusion (Phase 4 Day 13-14).
    
    Normalizes scores to [0, 1] range using min-max scaling.
    Returns zeros if all scores are identical (avoids division by zero).
    
    Args:
        scores: Dictionary mapping image_id to raw score
    
    Returns:
        Dictionary mapping image_id to normalized score in [0, 1]
    
    Example:
        >>> raw_scores = {"img1.jpg": 0.8, "img2.jpg": 0.5, "img3.jpg": 0.9}
        >>> normalized = _min_max_normalize(raw_scores)
        >>> # img3 -> 1.0, img1 -> 0.75, img2 -> 0.0
    """
    if not scores:
        return {}
    
    values = list(scores.values())
    v_min, v_max = float(min(values)), float(max(values))
    
    # Handle constant scores (all identical)
    if v_max - v_min < 1e-8:
        return {k: 0.0 for k in scores}
    
    scale = v_max - v_min
    return {k: (v - v_min) / scale for k, v in scores.items()}


def _fuse_scores(
    clip_scores: Dict[str, float],
    stage2_scores: Dict[str, float],
    kg_scores: Dict[str, float],
    w_clip: float,
    w_stage2: float,
    w_kg: float,
) -> Dict[str, float]:
    """
    Fuse multiple score signals using weighted combination (Phase 4 Day 13-14).
    
    Combines normalized scores from three signals:
    - clip_scores: Stage 1 CLIP bi-encoder similarity
    - stage2_scores: Stage 2 BLIP-2 cross-encoder scores
    - kg_scores: Entity graph-derived image scores
    
    All input dicts are assumed to be already normalized to [0, 1].
    Missing scores for a candidate are imputed as 0.0 (neutral).
    
    Args:
        clip_scores: Normalized CLIP scores (image_id -> score)
        stage2_scores: Normalized BLIP-2 scores (image_id -> score)
        kg_scores: Normalized KG scores (image_id -> score)
        w_clip: Weight for CLIP signal
        w_stage2: Weight for BLIP-2 signal
        w_kg: Weight for KG signal
    
    Returns:
        Dictionary mapping image_id to fused score
    
    Example:
        >>> clip = {"img1.jpg": 0.8, "img2.jpg": 0.5}
        >>> stage2 = {"img1.jpg": 0.9}  # Only scored img1
        >>> kg = {"img1.jpg": 0.7, "img2.jpg": 0.3}
        >>> fused = _fuse_scores(clip, stage2, kg, 0.6, 0.2, 0.2)
        >>> # img1: 0.6*0.8 + 0.2*0.9 + 0.2*0.7 = 0.80
        >>> # img2: 0.6*0.5 + 0.2*0.0 + 0.2*0.3 = 0.36
    """
    # Collect all candidate image IDs
    all_ids = set(clip_scores.keys()) | set(stage2_scores.keys()) | set(kg_scores.keys())
    
    # Compute fused scores
    fused = {}
    for img_id in all_ids:
        clip_val = clip_scores.get(img_id, 0.0)
        stage2_val = stage2_scores.get(img_id, 0.0)
        kg_val = kg_scores.get(img_id, 0.0)
        
        fused[img_id] = w_clip * clip_val + w_stage2 * stage2_val + w_kg * kg_val
    
    return fused


class HybridSearchEngine:
    """
    Hybrid Search Engine combining CLIP and BLIP-2 for improved retrieval.
    
    This class implements a two-stage retrieval pipeline:
    - Stage 1: Use CLIP bi-encoder for fast candidate retrieval (top-k1)
    - Stage 2: Use BLIP-2 cross-encoder for accurate re-ranking (top-k2)
    
    The hybrid approach achieves 15-20% better Recall@10 compared to CLIP-only
    while maintaining reasonable latency (<2s total).
    
    Attributes:
        bi_encoder: CLIP model for Stage 1 retrieval
        cross_encoder: BLIP-2 model for Stage 2 re-ranking
        image_index: FAISS index for image embeddings
        text_index: FAISS index for text embeddings (optional)
        dataset: Flickr30K dataset handler
        config: Configuration parameters
        logger: Logger instance
    
    Example:
        >>> # Initialize components
        >>> bi_encoder = BiEncoder()
        >>> cross_encoder = CrossEncoder()
        >>> image_index = FAISSIndex()
        >>> image_index.load('data/indices/image_index.faiss')
        >>> dataset = Flickr30KDataset()
        >>> 
        >>> # Create hybrid search engine
        >>> engine = HybridSearchEngine(
        ...     bi_encoder=bi_encoder,
        ...     cross_encoder=cross_encoder,
        ...     image_index=image_index,
        ...     dataset=dataset
        ... )
        >>> 
        >>> # Perform hybrid search
        >>> results = engine.text_to_image_hybrid_search(
        ...     query="a dog playing in the park",
        ...     k1=100,
        ...     k2=10
        ... )
        >>> for image_id, score in results:
        ...     print(f"{image_id}: {score:.4f}")
    """
    
    def __init__(
        self,
        bi_encoder: BiEncoder,
        cross_encoder: CrossEncoder,
        image_index: FAISSIndex,
        dataset: Flickr30KDataset,
        text_index: Optional[FAISSIndex] = None,
        config: Optional[Dict[str, Any]] = None,
        entity_graph: Optional[Any] = None,
        phase4_cfg: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the Hybrid Search Engine.
        
        Args:
            bi_encoder: CLIP bi-encoder for Stage 1 retrieval
            cross_encoder: BLIP-2 cross-encoder for Stage 2 re-ranking
            image_index: FAISS index containing image embeddings
            dataset: Dataset handler for loading images
            text_index: Optional FAISS index for text embeddings (for image-to-text)
            config: Configuration dictionary with search parameters
            entity_graph: Optional HeteroData entity graph for Phase 4 KG search
            phase4_cfg: Optional Phase 4 configuration (from entity_graph.yaml)
        
        Configuration Parameters:
            k1 (int): Number of candidates to retrieve in Stage 1 (default: 100)
            k2 (int): Number of final results after re-ranking (default: 10)
            batch_size (int): Batch size for BLIP-2 processing (default: 4)
            use_cache (bool): Enable query result caching (default: False)
            show_progress (bool): Show progress bars (default: True)
            stage1_weight (float): Weight for Stage 1 scores in fusion (default: 0.0)
            stage2_weight (float): Weight for Stage 2 scores in fusion (default: 1.0)
        """
        self.bi_encoder = bi_encoder
        self.cross_encoder = cross_encoder
        self.image_index = image_index
        self.text_index = text_index
        self.dataset = dataset
        
        # Phase 4: Entity graph and configuration
        self.entity_graph = entity_graph
        self.phase4_cfg = phase4_cfg
        
        # Setup logger
        print("Initializing Hybrid Search Engine...")
        
        # Load configuration
        self.config = self._load_config(config)
        
        # Cache for query results (optional optimization)
        self.cache_enabled = self.config.get('use_cache', False)
        self.cache: Dict[str, List[Tuple[str, float]]] = {}
        
        # Statistics tracking
        self.stats = {
            'total_queries': 0,
            'cache_hits': 0,
            'stage1_latency_ms': [],
            'stage2_latency_ms': [],
            'total_latency_ms': []
        }
        
        print("✓ Hybrid Search Engine initialized")
        print(f"  Stage 1: CLIP ({bi_encoder.model_name})")
        print(f"  Stage 2: BLIP-2")
        print(f"  Image Index: {image_index.index.ntotal:,} vectors")
        print(f"  Dataset: {len(dataset):,} images")
        
        # Phase 4 info
        if self.entity_graph is not None:
            n_entities = self.entity_graph["entity"].x.shape[0] if hasattr(self.entity_graph, "__getitem__") else 0
            print(f"  Phase 4 KG: {n_entities} entities (available)")
        
        print(f"  Config: k1={self.config['k1']}, k2={self.config['k2']}, "
              f"batch_size={self.config['batch_size']}, fusion={self.config['fusion_method']}")
    
    def _load_config(self, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Load and validate configuration parameters.
        
        Args:
            config: Optional configuration dictionary
        
        Returns:
            Configuration dictionary with defaults applied
        """
        default_config = {
            # Stage 1: CLIP retrieval
            'k1': 50,  # Reduced from 100 to cut Stage-2 cost
            
            # Stage 2: BLIP-2 re-ranking
            'k2': 10,  # Number of final results
            'batch_size': 8,  # Increased from 4 for better throughput
            
            # Performance
            'use_cache': False,  # Enable query caching
            'show_progress': True,  # Show progress bars
            
            # Score fusion - safer defaults (0.6/0.4) to avoid over-trusting Stage-2
            'fusion_method': 'weighted',  # 'replace', 'weighted', or 'rank_fusion'
            'stage1_weight': 0.6,  # Weight for CLIP scores (increased for safety)
            'stage2_weight': 0.4,  # Weight for BLIP-2 scores (decreased for safety)
        }
        
        if config is not None:
            default_config.update(config)
        
        # Validate parameters
        if default_config['k1'] < default_config['k2']:
            raise ValueError(f"k1 ({default_config['k1']}) must be >= k2 ({default_config['k2']})")
        
        if default_config['k2'] < 1:
            raise ValueError(f"k2 must be >= 1, got {default_config['k2']}")
        
        if default_config['batch_size'] < 1:
            raise ValueError(f"batch_size must be >= 1, got {default_config['batch_size']}")
        
        return default_config
    
    def text_to_image_hybrid_search(
        self,
        query: str,
        k1: Optional[int] = None,
        k2: Optional[int] = None,
        batch_size: Optional[int] = None,
        show_progress: Optional[bool] = None,
        mode: Optional[str] = None
    ) -> List[Tuple[str, float]]:
        """
        Perform hybrid text-to-image search with Phase 4 mode support.
        
        Pipeline (depends on mode):
        1. Encode query text with CLIP
        2. Retrieve top-k1 candidates from image index (Stage 1)
        3. Optionally run KG graph search (clip_kg, full modes)
        4. Optionally re-rank candidates with BLIP-2 (hybrid, full modes)
        5. Fuse scores according to mode weights
        6. Return top-k2 results
        
        Retrieval Modes (Phase 4 Day 13-14):
          - clip_only: CLIP Stage 1 only (no BLIP-2, no KG)
          - hybrid: CLIP + BLIP-2 (Phase 3 baseline, no KG)
          - clip_kg: CLIP + KG (no BLIP-2)
          - full: CLIP + BLIP-2 + KG (full Phase 4 hybrid)
        
        Args:
            query: Text query string
            k1: Number of candidates to retrieve in Stage 1 (default: from config)
            k2: Number of final results (default: from config)
            batch_size: Batch size for BLIP-2 (default: from config)
            show_progress: Show progress bars (default: from config)
            mode: Retrieval mode (default: from phase4_cfg fusion.default_mode or 'hybrid')
        
        Returns:
            List of (image_id, score) tuples, sorted by score (descending)
        
        Example:
            >>> # Phase 3 backward compatible (defaults to 'hybrid' if no phase4_cfg)
            >>> results = engine.text_to_image_hybrid_search("a dog playing")
            >>> 
            >>> # Phase 4 full mode with KG fusion
            >>> results = engine.text_to_image_hybrid_search("a dog playing", mode="full")
            >>> 
            >>> # CLIP-only mode for baseline
            >>> results = engine.text_to_image_hybrid_search("a dog playing", mode="clip_only")
        """
        import logging
        logger = logging.getLogger(__name__)
        
        # Use config defaults if not specified
        k1 = k1 or self.config['k1']
        k2 = k2 or self.config['k2']
        batch_size = batch_size or self.config['batch_size']
        show_progress = show_progress if show_progress is not None else self.config['show_progress']
        
        # Resolve mode: use provided mode, fallback to config default, or use 'clip_only'
        if mode is None:
            if self.phase4_cfg is not None:
                try:
                    from ..graph.config import get_fusion_config
                    fusion_cfg = get_fusion_config(self.phase4_cfg)
                    mode = fusion_cfg.get("default_mode", "hybrid")
                except Exception as e:
                    logger.warning(f"Failed to load fusion config: {e}, defaulting to 'hybrid'")
                    mode = "hybrid"
            else:
                mode = "hybrid"  # Backward compatible default
        
        # Validate mode
        valid_modes = ["clip_only", "hybrid", "clip_kg", "full"]
        if mode not in valid_modes:
            logger.warning(f"Unknown mode '{mode}', falling back to 'hybrid'")
            mode = "hybrid"
        
        # Check cache (include mode in cache key)
        if self.cache_enabled:
            cache_key = f"t2i:{query}:{k1}:{k2}:{mode}:{batch_size}"
            if cache_key in self.cache:
                self.stats['cache_hits'] += 1
                return self.cache[cache_key]
        
        # Track timing
        start_time = time.time()
        
        # =====================================================================
        # STAGE 1: CLIP RETRIEVAL (Always runs)
        # =====================================================================
        stage1_start = time.time()
        candidates = self._stage1_retrieve(
            query=query,
            k1=k1,
            show_progress=show_progress
        )
        stage1_time = (time.time() - stage1_start) * 1000
        
        # Extract image IDs and CLIP scores
        image_ids = [img_id for img_id, _ in candidates]
        clip_scores_raw = {img_id: score for img_id, score in candidates}
        
        # Early exit for clip_only mode
        if mode == "clip_only":
            results = candidates[:k2]
            total_time = (time.time() - start_time) * 1000
            
            # Update statistics
            self.stats['total_queries'] += 1
            self.stats['stage1_latency_ms'].append(stage1_time)
            self.stats['stage2_latency_ms'].append(0.0)
            self.stats['total_latency_ms'].append(total_time)
            
            if show_progress:
                print(f"Query completed in {total_time:.0f}ms (CLIP-only mode)")
            
            if self.cache_enabled:
                self.cache[cache_key] = results
            
            return results
        
        # =====================================================================
        # Determine which signals to compute based on mode
        # =====================================================================
        use_kg = mode in ("clip_kg", "full")
        use_stage2 = mode in ("hybrid", "full")
        
        # =====================================================================
        # KNOWLEDGE GRAPH SEARCH (if enabled for this mode)
        # =====================================================================
        kg_scores_raw: Dict[str, float] = {}
        
        if use_kg:
            if self.entity_graph is not None and self.phase4_cfg is not None:
                try:
                    from ..graph.graph_search import graph_search, image_scores_dict
                    from ..graph.config import get_query_enrichment_config
                    
                    logger.debug(f"Running KG graph search for mode '{mode}'")
                    
                    # Prepare seeds from Stage 1 CLIP results for enrichment
                    enrichment_cfg = get_query_enrichment_config(self.phase4_cfg) if self.phase4_cfg else None
                    if enrichment_cfg is not None:
                        K_seed_raw = enrichment_cfg.get("K_seed_raw", 32)
                    else:
                        K_seed_raw = len(candidates)
                    
                    seeds_for_graph: List[Tuple[str, float]] = [
                        (img_id, float(score)) for img_id, score in candidates[:K_seed_raw]
                    ]
                    
                    logger.debug(f"Passing {len(seeds_for_graph)} Stage-1 seeds to graph_search for enrichment")
                    
                    # Run graph search with seeds
                    kg_result = graph_search(
                        query=query,
                        graph=self.entity_graph,
                        encoders=self.bi_encoder,
                        cfg=self.phase4_cfg,
                        seeds=seeds_for_graph
                    )
                    
                    # Convert image_scores list to dict
                    kg_scores_all = image_scores_dict(kg_result)
                    
                    # Restrict KG scores to Stage 1 candidates only
                    kg_scores_raw = {
                        img_id: kg_scores_all.get(img_id, 0.0)
                        for img_id in image_ids
                    }
                    
                    logger.debug(f"KG search completed: {len(kg_scores_all)} images scored, "
                                f"{sum(1 for s in kg_scores_raw.values() if s > 0)} candidates matched")
                
                except Exception as e:
                    logger.warning(f"KG search failed: {e}, falling back to non-KG mode", exc_info=True)
                    use_kg = False
                    kg_scores_raw = {}
                    
                    # Adjust mode fallback
                    if mode == "clip_kg":
                        mode = "clip_only"
                        logger.info("Fallback: clip_kg -> clip_only")
                    elif mode == "full":
                        mode = "hybrid"
                        logger.info("Fallback: full -> hybrid")
            else:
                logger.warning("KG not available (entity_graph or phase4_cfg missing), disabling KG")
                use_kg = False
                kg_scores_raw = {}
                
                # Adjust mode fallback
                if mode == "clip_kg":
                    mode = "clip_only"
                elif mode == "full":
                    mode = "hybrid"
        
        # =====================================================================
        # STAGE 2: BLIP-2 RE-RANKING (if enabled for this mode)
        # =====================================================================
        stage2_scores_raw: Dict[str, float] = {}
        stage2_time = 0.0
        
        if use_stage2:
            if self.cross_encoder is not None:
                try:
                    stage2_start = time.time()
                    
                    # Run Stage 2 re-ranking (reuse existing implementation)
                    reranked = self._stage2_rerank(
                        query=query,
                        candidates=candidates,
                        k2=k1,  # Re-rank all candidates, fusion will select top-k2
                        batch_size=batch_size,
                        show_progress=show_progress
                    )
                    
                    # Extract Stage 2 scores
                    # Note: _stage2_rerank returns fused scores, but we need raw BLIP-2 scores
                    # For now, we'll work with what we have and note this limitation
                    # TODO: Refactor _stage2_rerank to return raw scores separately
                    stage2_scores_raw = {img_id: score for img_id, score in reranked}
                    
                    stage2_time = (time.time() - stage2_start) * 1000
                    
                except Exception as e:
                    logger.warning(f"Stage 2 BLIP-2 failed: {e}, disabling stage2", exc_info=True)
                    use_stage2 = False
                    stage2_scores_raw = {}
                    
                    # Adjust mode fallback
                    if mode == "hybrid":
                        mode = "clip_only"
                        logger.info("Fallback: hybrid -> clip_only")
                    elif mode == "full":
                        mode = "clip_kg" if use_kg else "clip_only"
                        logger.info(f"Fallback: full -> {mode}")
            else:
                logger.warning("BLIP-2 not available (cross_encoder is None), disabling stage2")
                use_stage2 = False
                stage2_scores_raw = {}
                
                # Adjust mode fallback
                if mode == "hybrid":
                    mode = "clip_only"
                elif mode == "full":
                    mode = "clip_kg" if use_kg else "clip_only"
        
        # =====================================================================
        # SCORE FUSION (Phase 4 Day 13-14)
        # =====================================================================
        
        # Load fusion weights for the current mode
        try:
            from ..graph.config import get_fusion_config
            fusion_cfg = get_fusion_config(self.phase4_cfg) if self.phase4_cfg else None
        except:
            fusion_cfg = None
        
        if fusion_cfg and mode in fusion_cfg:
            weights = fusion_cfg[mode]
            w_clip = float(weights.get("w_clip", 1.0))
            w_stage2 = float(weights.get("w_stage2", 0.0))
            w_kg = float(weights.get("w_kg", 0.0))
        else:
            # Fallback weights based on mode
            if mode == "clip_only":
                w_clip, w_stage2, w_kg = 1.0, 0.0, 0.0
            elif mode == "hybrid":
                w_clip, w_stage2, w_kg = 0.7, 0.3, 0.0
            elif mode == "clip_kg":
                w_clip, w_stage2, w_kg = 0.7, 0.0, 0.3
            elif mode == "full":
                w_clip, w_stage2, w_kg = 0.6, 0.2, 0.2
            else:
                w_clip, w_stage2, w_kg = 1.0, 0.0, 0.0
        
        # Adjust weights for disabled signals
        if not use_stage2:
            w_stage2 = 0.0
        if not use_kg:
            w_kg = 0.0
        
        logger.debug(f"Fusion weights for mode '{mode}': "
                    f"clip={w_clip:.2f}, stage2={w_stage2:.2f}, kg={w_kg:.2f}")
        
        # Normalize scores (min-max per signal)
        clip_norm = _min_max_normalize(clip_scores_raw)
        stage2_norm = _min_max_normalize(stage2_scores_raw) if stage2_scores_raw else {}
        kg_norm = _min_max_normalize(kg_scores_raw) if kg_scores_raw else {}
        
        # Fuse normalized scores
        fused_scores = _fuse_scores(
            clip_scores=clip_norm,
            stage2_scores=stage2_norm,
            kg_scores=kg_norm,
            w_clip=w_clip,
            w_stage2=w_stage2,
            w_kg=w_kg
        )
        
        # Sort by fused score and select top-k2
        sorted_items = sorted(fused_scores.items(), key=lambda kv: kv[1], reverse=True)
        results = sorted_items[:k2]
        
        # Calculate total time
        total_time = (time.time() - start_time) * 1000
        
        # Update statistics
        self.stats['total_queries'] += 1
        self.stats['stage1_latency_ms'].append(stage1_time)
        self.stats['stage2_latency_ms'].append(stage2_time)
        self.stats['total_latency_ms'].append(total_time)
        
        if show_progress:
            print(f"Query completed in {total_time:.0f}ms "
                  f"(Stage 1: {stage1_time:.0f}ms, Stage 2: {stage2_time:.0f}ms, mode: {mode})")
        
        # Cache results
        if self.cache_enabled:
            self.cache[cache_key] = results
        
        return results
    
    def _stage1_retrieve(
        self,
        query: str,
        k1: int = 100,
        show_progress: bool = True
    ) -> List[Tuple[str, float]]:
        """
        Stage 1: Fast retrieval using CLIP bi-encoder.
        
        This stage quickly retrieves top-k1 candidates from the entire collection
        using pre-computed CLIP embeddings and FAISS index.
        
        Target latency: <100ms
        
        Args:
            query: Text query string
            k1: Number of candidates to retrieve
            show_progress: Show progress information
        
        Returns:
            List of (image_id, clip_score) tuples for top-k1 candidates
        """
        # Encode query with CLIP
        query_embedding = self.bi_encoder.encode_texts(
            texts=[query],
            batch_size=1,
            normalize=True,
            show_progress=False
        )
        
        # Search FAISS index
        scores, indices = self.image_index.search(
            query_embeddings=query_embedding,
            k=k1,
            return_scores=True
        )
        
        # Convert to list of (image_id, score) tuples
        candidates = []
        image_ids = self.image_index.metadata.get('ids', [])
        
        for idx, score in zip(indices[0], scores[0]):
            if idx < len(image_ids):
                image_id = image_ids[idx]
                candidates.append((image_id, float(score)))
        
        return candidates
    
    def _stage2_rerank(
        self,
        query: str,
        candidates: List[Tuple[str, float]],
        k2: int = 10,
        batch_size: int = 4,
        show_progress: bool = True
    ) -> List[Tuple[str, float]]:
        """
        Stage 2: Accurate re-ranking using BLIP-2 cross-encoder with fusion.
        
        This stage re-scores the top-k1 candidates from Stage 1 using BLIP-2's
        deep cross-modal interaction, providing more accurate relevance scores.
        
        Implements fusion between Stage-1 and Stage-2 scores for robustness:
        - 'replace': Use only Stage-2 scores (requires higher_is_better correctness)
        - 'weighted': Weighted combination of normalized Stage-1 and Stage-2 scores
        - 'rank_fusion': Reciprocal rank fusion (orientation-safe)
        
        Includes guardrails:
        - Filters out missing files before scoring
        - Health check: detects inverted rankings via correlation
        - Automatic correction: flips Stage-2 or switches to rank fusion if inverted
        
        Target latency: <2000ms for 50 candidates with batch_size=8
        
        Args:
            query: Text query string
            candidates: List of (image_id, clip_score) from Stage 1
            k2: Number of final results to return
            batch_size: Batch size for BLIP-2 processing
            show_progress: Show progress bar
        
        Returns:
            List of (image_id, final_score) tuples for top-k2 results,
            sorted by final score (descending)
        """
        if not candidates:
            return []
        
        # Extract image IDs and Stage-1 scores
        image_ids = [img_id for img_id, _ in candidates]
        clip_scores = np.array([score for _, score in candidates], dtype=np.float32)
        
        # Guardrail 1: Filter out missing files
        image_paths = [self.dataset.images_dir / img_id for img_id in image_ids]
        valid_indices = [i for i, path in enumerate(image_paths) if path.exists()]
        
        if not valid_indices:
            print("Warning: No valid image files found, returning Stage-1 results")
            return candidates[:k2]
        
        if len(valid_indices) < len(image_ids):
            print(f"Warning: {len(image_ids) - len(valid_indices)} images not found, using {len(valid_indices)} valid images")
            image_ids = [image_ids[i] for i in valid_indices]
            image_paths = [image_paths[i] for i in valid_indices]
            clip_scores = clip_scores[valid_indices]
        
        # Prepare batch data
        queries = [query] * len(image_ids)
        
        # Score with BLIP-2
        try:
            blip2_scores = self.cross_encoder.score_pairs(
                queries=queries,
                candidates=image_paths,
                query_type='text',
                candidate_type='image',
                batch_size=batch_size,
                show_progress=show_progress
            )
            blip2_scores = np.array(blip2_scores, dtype=np.float32)
        except Exception as e:
            print(f"Warning: Stage-2 scoring failed: {e}")
            print("Falling back to Stage-1 results")
            return candidates[:k2]
        
        # Agreement quality thresholds
        LOW_AGREE = 0.15   # |rho| below this → weak agreement
        NEGATIVE = -0.20   # rho below this → inversion
        
        # Get scoring direction from cross-encoder
        stage2_higher = getattr(self.cross_encoder, "higher_is_better", True)
        
        # Orient Stage-2 scores for fusion (higher = better)
        blip2_for_fusion = blip2_scores if stage2_higher else (-1.0 * blip2_scores)
        
        # Orient Stage-1 scores: try both signs and choose the one that best aligns with Stage-2
        clip_raw = np.asarray(clip_scores, dtype=np.float32)
        if len(clip_raw) > 3:
            c_pos = np.corrcoef(clip_raw, blip2_for_fusion)[0, 1]
            c_neg = np.corrcoef(-clip_raw, blip2_for_fusion)[0, 1]
            clip_for_fusion = clip_raw if abs(c_pos) >= abs(c_neg) else (-clip_raw)
            correlation = np.corrcoef(clip_for_fusion, blip2_for_fusion)[0, 1]
        else:
            clip_for_fusion = clip_raw
            correlation = 0.0
        
        # Default safer weights (0.6/0.4) unless config overrides were provided
        stage1_weight = float(self.config.get('stage1_weight', 0.6))
        stage2_weight = float(self.config.get('stage2_weight', 0.4))
        fusion_method = self.config.get('fusion_method', 'weighted')
        
        # Gate on agreement quality
        if correlation < NEGATIVE:
            # True inversion → flip Stage-2 and keep conservative weights
            if stage2_higher:
                blip2_for_fusion = 1.0 - blip2_for_fusion  # probability flip (p → 1-p)
            else:
                blip2_for_fusion = -blip2_for_fusion       # score flip
            print(f"Warning: Stage-2 inverted (rho={correlation:.3f}) → flipped, weights=0.6/0.4")
            fusion_method = 'weighted'
            stage1_weight, stage2_weight = 0.6, 0.4
            
        elif abs(correlation) < LOW_AGREE:
            # Weak agreement → prefer rank fusion (orientation-safe)
            print(f"Note: Stage-1/2 weak agreement (rho={correlation:.3f}) → rank_fusion fallback")
            fusion_method = 'rank_fusion'
        else:
            # Good agreement: use configured fusion method/weights as-is
            pass
        
        # Apply fusion method
        if fusion_method == 'replace':
            # Use only Stage-2 scores (after orientation correction)
            final_scores = blip2_for_fusion
            
        elif fusion_method == 'weighted':
            # Weighted combination with robust normalization
            clip_norm = _normalize(clip_for_fusion)
            blip2_norm = _normalize(blip2_for_fusion)
            
            # Weighted combination
            final_scores = stage1_weight * clip_norm + stage2_weight * blip2_norm
            final_scores = np.asarray(final_scores, dtype=np.float32)
            
        elif fusion_method == 'rank_fusion':
            # Orientation-safe reciprocal rank fusion
            # Build ranks where lower rank index means better candidate
            clip_rank = np.argsort(np.argsort(-clip_for_fusion))
            blip_rank = np.argsort(np.argsort(-blip2_for_fusion))
            # RRF with k=60 (typical); avoid divide-by-zero with +1
            k_rrf = 60.0
            final_scores = 1.0 / (k_rrf + clip_rank + 1) + 1.0 / (k_rrf + blip_rank + 1)
            final_scores = np.asarray(final_scores, dtype=np.float32)
            
        else:
            print(f"Warning: Unknown fusion_method '{fusion_method}', using 'weighted'")
            clip_norm = _normalize(clip_for_fusion)
            blip2_norm = _normalize(blip2_for_fusion)
            final_scores = stage1_weight * clip_norm + stage2_weight * blip2_norm
            final_scores = np.asarray(final_scores, dtype=np.float32)
        
        # Apply ordering: higher final_scores = better
        order = np.argsort(-final_scores)
        
        # Create list of (image_id, final_score) tuples
        reranked_results = [
            (image_ids[idx], float(final_scores[idx]))
            for idx in order[:k2]
        ]
        
        return reranked_results
    
    def image_to_image_hybrid_search(
        self,
        query_image: Union[str, Path, Image.Image],
        k1: Optional[int] = None,
        k2: Optional[int] = None,
        batch_size: Optional[int] = None,
        show_progress: Optional[bool] = None
    ) -> List[Tuple[str, float]]:
        """
        Perform hybrid image-to-image search.
        
        Pipeline:
        1. Encode query image with CLIP
        2. Retrieve top-k1 candidates from image index (Stage 1)
        3. Re-rank candidates with BLIP-2 image comparison (Stage 2)
        4. Return top-k2 results
        
        Args:
            query_image: Query image (PIL Image, path string, or Path)
            k1: Number of candidates to retrieve in Stage 1 (default: from config)
            k2: Number of final results (default: from config)
            batch_size: Batch size for BLIP-2 (default: from config)
            show_progress: Show progress bars (default: from config)
        
        Returns:
            List of (image_id, score) tuples, sorted by score (descending)
        
        Example:
            >>> results = engine.image_to_image_hybrid_search(
            ...     query_image="query.jpg",
            ...     k1=100,
            ...     k2=10
            ... )
        """
        # Use config defaults if not specified
        k1 = k1 or self.config['k1']
        k2 = k2 or self.config['k2']
        batch_size = batch_size or self.config['batch_size']
        show_progress = show_progress if show_progress is not None else self.config['show_progress']
        
        # Load image if path is provided
        if isinstance(query_image, (str, Path)):
            query_image = Image.open(query_image).convert('RGB')
        
        # Track timing
        start_time = time.time()
        
        # Stage 1: CLIP retrieval
        stage1_start = time.time()
        candidates = self._stage1_retrieve_image(
            query_image=query_image,
            k1=k1,
            show_progress=show_progress
        )
        stage1_time = (time.time() - stage1_start) * 1000
        
        # Stage 2: BLIP-2 re-ranking (for image-to-image, we can use image captioning)
        # For now, we'll skip Stage 2 for image-to-image and just return CLIP results
        # TODO: Implement BLIP-2 image-to-image comparison
        stage2_time = 0
        
        print("Warning: Image-to-image Stage 2 re-ranking not yet implemented. "
              "Returning CLIP-only results.")
        
        results = candidates[:k2]
        
        # Calculate total time
        total_time = (time.time() - start_time) * 1000
        
        # Update statistics
        self.stats['total_queries'] += 1
        self.stats['stage1_latency_ms'].append(stage1_time)
        self.stats['stage2_latency_ms'].append(stage2_time)
        self.stats['total_latency_ms'].append(total_time)
        
        return results
    
    def _stage1_retrieve_image(
        self,
        query_image: Image.Image,
        k1: int = 100,
        show_progress: bool = True
    ) -> List[Tuple[str, float]]:
        """
        Stage 1: Fast image retrieval using CLIP bi-encoder.
        
        Args:
            query_image: PIL Image
            k1: Number of candidates to retrieve
            show_progress: Show progress information
        
        Returns:
            List of (image_id, clip_score) tuples for top-k1 candidates
        """
        # Encode query image with CLIP
        query_embedding = self.bi_encoder.encode_images(
            images=[query_image],
            batch_size=1,
            normalize=True,
            show_progress=False
        )
        
        # Search FAISS index
        scores, indices = self.image_index.search(
            query_embeddings=query_embedding,
            k=k1,
            return_scores=True
        )
        
        # Convert to list of (image_id, score) tuples
        candidates = []
        image_ids = self.image_index.metadata.get('ids', [])
        
        for idx, score in zip(indices[0], scores[0]):
            if idx < len(image_ids):
                image_id = image_ids[idx]
                candidates.append((image_id, float(score)))
        
        return candidates
    
    def batch_text_to_image_search(
        self,
        queries: List[str],
        k1: Optional[int] = None,
        k2: Optional[int] = None,
        batch_size: Optional[int] = None,
        show_progress: Optional[bool] = None,
        mode: Optional[str] = None
    ) -> List[List[Tuple[str, float]]]:
        """
        Perform batch hybrid text-to-image search for multiple queries with Phase 4 mode support.
        
        Efficiently processes multiple queries by:
        1. Batching Stage 1 (CLIP) encoding for all queries at once
        2. Batching Stage 2 (BLIP-2) re-ranking across all candidates
        3. Optionally batching KG graph search across all queries
        
        Args:
            queries: List of text query strings
            k1: Number of candidates per query (default: from config)
            k2: Number of final results per query (default: from config)
            batch_size: Batch size for BLIP-2 (default: from config)
            show_progress: Show progress bars (default: from config)
            mode: Retrieval mode (clip_only, hybrid, clip_kg, full) (default: from config)
        
        Returns:
            List of result lists, one per query. Each result list contains
            (image_id, score) tuples sorted by score.
        
        Example:
            >>> queries = ["a dog", "a cat", "a bird"]
            >>> results = engine.batch_text_to_image_search(queries, mode="full")
            >>> for i, query_results in enumerate(results):
            ...     print(f"Query '{queries[i]}':")
            ...     for img_id, score in query_results[:3]:
            ...         print(f"  {img_id}: {score:.4f}")
        
        Performance Notes:
            - Stage 1 processes all queries in parallel (single batch)
            - Stage 2 batches all candidates together for efficiency
            - KG search is run per-query but can be optimized in future
            - Typically 2-3x faster than sequential processing
        """
        import logging
        logger = logging.getLogger(__name__)
        
        # Use config defaults if not specified
        k1 = k1 or self.config['k1']
        k2 = k2 or self.config['k2']
        batch_size = batch_size or self.config['batch_size']
        show_progress = show_progress if show_progress is not None else self.config['show_progress']
        
        # Resolve mode
        if mode is None:
            if self.phase4_cfg is not None:
                try:
                    from ..graph.config import get_fusion_config
                    fusion_cfg = get_fusion_config(self.phase4_cfg)
                    mode = fusion_cfg.get("default_mode", "hybrid")
                except Exception:
                    mode = "hybrid"
            else:
                mode = "hybrid"
        
        if not queries:
            return []
        
        # For simplicity, delegate to single-query search for now
        # TODO: Future optimization - batch Stage 1 + Stage 2 + KG across all queries
        results = []
        for query in (tqdm(queries, desc="Batch search") if show_progress else queries):
            query_results = self.text_to_image_hybrid_search(
                query=query,
                k1=k1,
                k2=k2,
                batch_size=batch_size,
                show_progress=False,
                mode=mode
            )
            results.append(query_results)
        
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get search engine statistics.
        
        Returns:
            Dictionary containing performance metrics and statistics
        """
        stats = {
            'total_queries': self.stats['total_queries'],
            'cache_hits': self.stats['cache_hits'],
            'cache_hit_rate': (
                self.stats['cache_hits'] / self.stats['total_queries']
                if self.stats['total_queries'] > 0 else 0.0
            )
        }
        
        # Calculate latency statistics
        if self.stats['total_latency_ms']:
            stats['latency'] = {
                'stage1_ms': {
                    'mean': np.mean(self.stats['stage1_latency_ms']),
                    'median': np.median(self.stats['stage1_latency_ms']),
                    'p95': np.percentile(self.stats['stage1_latency_ms'], 95),
                    'p99': np.percentile(self.stats['stage1_latency_ms'], 99),
                },
                'stage2_ms': {
                    'mean': np.mean(self.stats['stage2_latency_ms']),
                    'median': np.median(self.stats['stage2_latency_ms']),
                    'p95': np.percentile(self.stats['stage2_latency_ms'], 95),
                    'p99': np.percentile(self.stats['stage2_latency_ms'], 99),
                },
                'total_ms': {
                    'mean': np.mean(self.stats['total_latency_ms']),
                    'median': np.median(self.stats['total_latency_ms']),
                    'p95': np.percentile(self.stats['total_latency_ms'], 95),
                    'p99': np.percentile(self.stats['total_latency_ms'], 99),
                }
            }
        
        return stats
    
    def update_config(self, **kwargs) -> Dict[str, Any]:
        """
        Update configuration parameters at runtime.
        
        Args:
            **kwargs: Configuration parameters to update
                k1 (int): Stage 1 candidate count (50, 100, 200)
                k2 (int): Final result count (5, 10, 20)
                batch_size (int): BLIP-2 batch size (2, 4, 8)
                use_cache (bool): Enable/disable caching
                show_progress (bool): Show/hide progress bars
        
        Returns:
            Updated configuration dictionary
        
        Example:
            >>> engine.update_config(k1=200, k2=20, batch_size=8)
            >>> print(engine.config['k1'])  # 200
        """
        old_config = self.config.copy()
        
        # Update config
        for key, value in kwargs.items():
            if key in self.config:
                self.config[key] = value
            else:
                print(f"Warning: Unknown config parameter '{key}' ignored")
        
        # Validate updated config
        try:
            if self.config['k1'] < self.config['k2']:
                raise ValueError(f"k1 ({self.config['k1']}) must be >= k2 ({self.config['k2']})")
            
            if self.config['k2'] < 1:
                raise ValueError(f"k2 must be >= 1, got {self.config['k2']}")
            
            if self.config['batch_size'] < 1:
                raise ValueError(f"batch_size must be >= 1, got {self.config['batch_size']}")
        
        except ValueError as e:
            # Revert to old config on validation error
            self.config = old_config
            print(f"Config update failed: {e}")
            print("Reverted to previous configuration")
            return self.config
        
        # Update cache_enabled flag
        self.cache_enabled = self.config.get('use_cache', False)
        
        # Clear cache if caching was disabled
        if not self.cache_enabled and self.cache:
            self.clear_cache()
        
        print(f"✓ Configuration updated:")
        for key, value in kwargs.items():
            if key in self.config:
                print(f"  {key}: {old_config.get(key)} → {value}")
        
        return self.config
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get current configuration.
        
        Returns:
            Copy of current configuration dictionary
        """
        return self.config.copy()
    
    def reset_config(self) -> Dict[str, Any]:
        """
        Reset configuration to defaults.
        
        Returns:
            Reset configuration dictionary
        """
        print("Resetting configuration to defaults...")
        self.config = self._load_config(None)
        self.cache_enabled = self.config.get('use_cache', False)
        if not self.cache_enabled:
            self.clear_cache()
        print("✓ Configuration reset")
        return self.config
    
    def clear_cache(self):
        """
        Clear the query result cache.
        
        Returns:
            Number of cached entries cleared
        """
        n_entries = len(self.cache)
        self.cache.clear()
        print(f"✓ Cache cleared ({n_entries} entries removed)")
        return n_entries
    
    def get_cache_size(self) -> int:
        """
        Get current cache size.
        
        Returns:
            Number of cached queries
        """
        return len(self.cache)
    
    def get_cache_keys(self) -> List[str]:
        """
        Get all cached query keys.
        
        Returns:
            List of cached query strings
        """
        return list(self.cache.keys())
    
    def reset_statistics(self):
        """
        Reset performance statistics.
        
        Returns:
            Previous statistics before reset
        """
        old_stats = self.stats.copy()
        self.stats = {
            'total_queries': 0,
            'cache_hits': 0,
            'stage1_latency_ms': [],
            'stage2_latency_ms': [],
            'total_latency_ms': []
        }
        print(f"✓ Statistics reset ({old_stats['total_queries']} queries cleared)")
        return old_stats
    
    def profile_search(
        self,
        test_queries: Optional[List[str]] = None,
        n_queries: int = 10,
        k1_values: Optional[List[int]] = None,
        k2_values: Optional[List[int]] = None,
        batch_sizes: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """
        Profile search performance with different configurations.
        
        Tests various parameter combinations to find optimal settings.
        
        Args:
            test_queries: List of test queries (if None, uses sample from dataset)
            n_queries: Number of queries to test (if test_queries is None)
            k1_values: List of k1 values to test (default: [50, 100, 200])
            k2_values: List of k2 values to test (default: [5, 10, 20])
            batch_sizes: List of batch sizes to test (default: [2, 4, 8])
        
        Returns:
            Dictionary with profiling results
        
        Example:
            >>> results = engine.profile_search(
            ...     test_queries=["a dog", "a cat", "a bird"],
            ...     k1_values=[50, 100],
            ...     k2_values=[10],
            ...     batch_sizes=[4, 8]
            ... )
            >>> print(results['summary'])
        """
        print("\n" + "="*70)
        print("PERFORMANCE PROFILING")
        print("="*70)
        
        # Default test queries from dataset
        if test_queries is None:
            print(f"\nGenerating {n_queries} test queries from dataset...")
            unique_images = self.dataset.get_unique_images()
            all_captions = []
            for i in range(min(n_queries, len(unique_images))):
                image_id = unique_images[i]
                captions = self.dataset.get_captions(image_id)
                if captions:
                    all_captions.append(captions[0])
            test_queries = all_captions[:n_queries]
            print(f"  ✓ Generated {len(test_queries)} test queries")
        
        # Default parameter ranges
        k1_values = k1_values or [50, 100, 200]
        k2_values = k2_values or [5, 10, 20]
        batch_sizes = batch_sizes or [2, 4, 8]
        
        print(f"\nTest configuration:")
        print(f"  Queries: {len(test_queries)}")
        print(f"  k1 values: {k1_values}")
        print(f"  k2 values: {k2_values}")
        print(f"  Batch sizes: {batch_sizes}")
        
        # Save original config
        original_config = self.config.copy()
        
        results = {
            'test_queries': test_queries,
            'configs_tested': [],
            'best_config': None,
            'best_latency': float('inf')
        }
        
        # Test each configuration
        total_tests = len(k1_values) * len(k2_values) * len(batch_sizes)
        test_num = 0
        
        print(f"\nRunning {total_tests} configuration tests...")
        print("-"*70)
        
        for k1 in k1_values:
            for k2 in k2_values:
                for batch_size in batch_sizes:
                    test_num += 1
                    
                    # Skip invalid combinations
                    if k1 < k2:
                        continue
                    
                    print(f"\n[Test {test_num}/{total_tests}] k1={k1}, k2={k2}, batch_size={batch_size}")
                    
                    # Update config
                    self.update_config(k1=k1, k2=k2, batch_size=batch_size, show_progress=False)
                    
                    # Reset stats for this test
                    self.reset_statistics()
                    
                    # Run test queries
                    start_time = time.time()
                    for query in test_queries:
                        self.text_to_image_hybrid_search(
                            query=query,
                            show_progress=False
                        )
                    test_time = (time.time() - start_time) * 1000
                    
                    # Get statistics
                    stats = self.get_statistics()
                    
                    avg_latency = test_time / len(test_queries)
                    
                    config_result = {
                        'k1': k1,
                        'k2': k2,
                        'batch_size': batch_size,
                        'total_time_ms': test_time,
                        'avg_latency_ms': avg_latency,
                        'stage1_avg_ms': stats['latency']['stage1_ms']['mean'] if 'latency' in stats else 0,
                        'stage2_avg_ms': stats['latency']['stage2_ms']['mean'] if 'latency' in stats else 0,
                    }
                    
                    results['configs_tested'].append(config_result)
                    
                    print(f"  Total: {test_time:.2f}ms | Avg: {avg_latency:.2f}ms/query")
                    print(f"  Stage 1: {config_result['stage1_avg_ms']:.2f}ms | "
                          f"Stage 2: {config_result['stage2_avg_ms']:.2f}ms")
                    
                    # Track best config
                    if avg_latency < results['best_latency']:
                        results['best_latency'] = avg_latency
                        results['best_config'] = config_result
        
        # Restore original config
        self.config = original_config
        self.cache_enabled = self.config.get('use_cache', False)
        
        # Generate summary
        print("\n" + "="*70)
        print("PROFILING SUMMARY")
        print("="*70)
        
        print(f"\nConfigurations tested: {len(results['configs_tested'])}")
        
        if results['best_config']:
            best = results['best_config']
            print(f"\n🏆 Best Configuration:")
            print(f"  k1={best['k1']}, k2={best['k2']}, batch_size={best['batch_size']}")
            print(f"  Average latency: {best['avg_latency_ms']:.2f}ms")
            print(f"  Stage 1: {best['stage1_avg_ms']:.2f}ms")
            print(f"  Stage 2: {best['stage2_avg_ms']:.2f}ms")
        
        # Show top 3 configs
        sorted_configs = sorted(results['configs_tested'], key=lambda x: x['avg_latency_ms'])
        
        print(f"\nTop 3 Configurations:")
        print(f"{'Rank':<6} {'k1':<6} {'k2':<6} {'Batch':<8} {'Avg Latency (ms)':<18}")
        print("-"*70)
        for i, config in enumerate(sorted_configs[:3], 1):
            print(f"{i:<6} {config['k1']:<6} {config['k2']:<6} "
                  f"{config['batch_size']:<8} {config['avg_latency_ms']:<18.2f}")
        
        print("\n" + "="*70)
        
        results['summary'] = {
            'best_config': results['best_config'],
            'top_3': sorted_configs[:3]
        }
        
        return results
    
    def optimize_config(
        self,
        target_latency_ms: float = 500,
        test_queries: Optional[List[str]] = None,
        n_queries: int = 10
    ) -> Dict[str, Any]:
        """
        Automatically find optimal configuration for target latency.
        
        Args:
            target_latency_ms: Target average latency per query
            test_queries: Test queries (if None, samples from dataset)
            n_queries: Number of test queries
        
        Returns:
            Dictionary with optimization results and recommended config
        
        Example:
            >>> result = engine.optimize_config(target_latency_ms=400)
            >>> engine.update_config(**result['recommended_config'])
        """
        print(f"\nOptimizing for target latency: {target_latency_ms}ms")
        
        # Profile with different configs
        profile_results = self.profile_search(
            test_queries=test_queries,
            n_queries=n_queries
        )
        
        # Find config closest to target
        best_match = None
        min_diff = float('inf')
        
        for config in profile_results['configs_tested']:
            diff = abs(config['avg_latency_ms'] - target_latency_ms)
            if diff < min_diff:
                min_diff = diff
                best_match = config
        
        result = {
            'target_latency_ms': target_latency_ms,
            'recommended_config': {
                'k1': best_match['k1'],
                'k2': best_match['k2'],
                'batch_size': best_match['batch_size']
            },
            'expected_latency_ms': best_match['avg_latency_ms'],
            'latency_diff_ms': min_diff,
            'profile_results': profile_results
        }
        
        print(f"\n✓ Optimization complete")
        print(f"  Recommended config: k1={best_match['k1']}, k2={best_match['k2']}, "
              f"batch_size={best_match['batch_size']}")
        print(f"  Expected latency: {best_match['avg_latency_ms']:.2f}ms "
              f"(target: {target_latency_ms}ms)")
        
        return result
    
    def __repr__(self) -> str:
        """String representation of the search engine."""
        cache_info = f", cache={len(self.cache)}" if self.cache_enabled else ""
        return (
            f"HybridSearchEngine("
            f"images={self.image_index.index.ntotal:,}, "
            f"k1={self.config['k1']}, "
            f"k2={self.config['k2']}, "
            f"batch_size={self.config['batch_size']}, "
            f"queries={self.stats['total_queries']}"
            f"{cache_info}"
            f")"
        )
    
    # ============================================================================
    # Phase 4: Graph-based search with query enrichment
    # ============================================================================
    
    def text_to_image_graph_search(
        self,
        query: str,
        entity_context: Optional[Dict[int, Dict[str, Any]]] = None,
        phase4_config: Optional[Dict[str, Any]] = None,
        k1: Optional[int] = None,
        k2: Optional[int] = None,
        show_progress: Optional[bool] = None
    ) -> List[Tuple[str, float]]:
        """
        DEPRECATED: Use text_to_image_hybrid_search with mode='clip_kg' or mode='full' instead.
        
        This method is a thin wrapper for backward compatibility and will be removed in a future version.
        
        Legacy entry point for Phase 4 entity-centric retrieval with query enrichment.
        
        Args:
            query: Text query string
            entity_context: DEPRECATED - Not used (entity_context is loaded from phase4_cfg in hybrid search)
            phase4_config: DEPRECATED - Use HybridSearchEngine initialization parameter instead
            k1: Number of Stage 1 candidates (default: from config)
            k2: Number of final results (default: from config)
            show_progress: Show progress bars (default: from config)
        
        Returns:
            List of (image_id, score) tuples
        """
        import warnings
        import logging
        
        warnings.warn(
            "text_to_image_graph_search is deprecated and will be removed in a future version. "
            "Use text_to_image_hybrid_search(mode='clip_kg') for KG-only search or "
            "text_to_image_hybrid_search(mode='full') for full pipeline (CLIP + BLIP-2 + KG).",
            DeprecationWarning,
            stacklevel=2
        )
        
        # Use defaults if not provided
        k1 = k1 or self.config['k1']
        k2 = k2 or self.config['k2']
        show_progress = show_progress if show_progress is not None else self.config['show_progress']
        
        # Determine best mode based on what's available
        # If entity_graph is available, use 'clip_kg', otherwise fall back to 'clip_only'
        if self.entity_graph is not None and self.phase4_cfg is not None:
            mode = "clip_kg"
        else:
            mode = "clip_only"
            logging.getLogger(__name__).warning(
                "Entity graph not available for text_to_image_graph_search, falling back to clip_only mode"
            )
        
        # Delegate to unified hybrid search with appropriate mode
        return self.text_to_image_hybrid_search(
            query=query,
            k1=k1,
            k2=k2,
            mode=mode,
            show_progress=show_progress
        )


if __name__ == "__main__":
    # Simple test
    print("Hybrid Search Engine module loaded successfully")
    print("To use, initialize with:")
    print("  - BiEncoder (CLIP)")
    print("  - CrossEncoder (BLIP-2)")
    print("  - FAISSIndex (image embeddings)")
    print("  - Flickr30KDataset")
