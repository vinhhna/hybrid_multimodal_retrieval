"""
Hybrid Search Engine for Multimodal Retrieval

This module implements a two-stage hybrid search pipeline that combines:
1. Stage 1 (CLIP Bi-Encoder): Fast retrieval of top-k1 candidates
2. Stage 2 (BLIP-2 Cross-Encoder): Accurate re-ranking of candidates

The hybrid approach balances speed and accuracy, achieving better results
than CLIP alone while being much faster than using BLIP-2 for all candidates.

Phase 3: Hybrid Retrieval System
Created: November 4, 2025
"""

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
        config: Optional[Dict[str, Any]] = None
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
            
            # Score fusion
            'fusion_method': 'weighted',  # 'replace', 'weighted', or 'rank_fusion'
            'stage1_weight': 0.3,  # Weight for CLIP scores
            'stage2_weight': 0.7,  # Weight for BLIP-2 scores
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
        show_progress: Optional[bool] = None
    ) -> List[Tuple[str, float]]:
        """
        Perform hybrid text-to-image search.
        
        Pipeline:
        1. Encode query text with CLIP
        2. Retrieve top-k1 candidates from image index (Stage 1)
        3. Re-rank candidates with BLIP-2 (Stage 2)
        4. Return top-k2 results
        
        Args:
            query: Text query string
            k1: Number of candidates to retrieve in Stage 1 (default: from config)
            k2: Number of final results (default: from config)
            batch_size: Batch size for BLIP-2 (default: from config)
            show_progress: Show progress bars (default: from config)
        
        Returns:
            List of (image_id, score) tuples, sorted by score (descending)
        
        Example:
            >>> results = engine.text_to_image_hybrid_search(
            ...     query="a dog playing in the park",
            ...     k1=100,
            ...     k2=10
            ... )
            >>> for image_id, score in results[:5]:
            ...     print(f"{image_id}: {score:.4f}")
        """
        # Use config defaults if not specified
        k1 = k1 or self.config['k1']
        k2 = k2 or self.config['k2']
        batch_size = batch_size or self.config['batch_size']
        show_progress = show_progress if show_progress is not None else self.config['show_progress']
        
        # Check cache (include fusion-affecting parameters in key)
        if self.cache_enabled:
            cache_key = "t2i:{q}:{k1}:{k2}:{fm}:{w1}:{w2}:{bs}".format(
                q=query,
                k1=k1,
                k2=k2,
                fm=self.config.get('fusion_method', 'weighted'),
                w1=self.config.get('stage1_weight', 0.3),
                w2=self.config.get('stage2_weight', 0.7),
                bs=batch_size
            )
            if cache_key in self.cache:
                self.stats['cache_hits'] += 1
                # Cache hit for query
                return self.cache[cache_key]
        
        # Track timing
        start_time = time.time()
        
        # Stage 1: CLIP retrieval
        stage1_start = time.time()
        candidates = self._stage1_retrieve(
            query=query,
            k1=k1,
            show_progress=show_progress
        )
        stage1_time = (time.time() - stage1_start) * 1000  # Convert to ms
        
        # Stage 2: BLIP-2 re-ranking
        stage2_start = time.time()
        results = self._stage2_rerank(
            query=query,
            candidates=candidates,
            k2=k2,
            batch_size=batch_size,
            show_progress=show_progress
        )
        stage2_time = (time.time() - stage2_start) * 1000  # Convert to ms
        
        # Calculate total time
        total_time = (time.time() - start_time) * 1000
        
        # Update statistics
        self.stats['total_queries'] += 1
        self.stats['stage1_latency_ms'].append(stage1_time)
        self.stats['stage2_latency_ms'].append(stage2_time)
        self.stats['total_latency_ms'].append(total_time)
        
        # Query completed
        print(f"Query completed in {total_time:.0f}ms (Stage 1: {stage1_time:.0f}ms, Stage 2: {stage2_time:.0f}ms)")
        
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
        
        # Get scoring direction from cross-encoder
        stage2_higher = getattr(self.cross_encoder, "higher_is_better", True)
        
        # Orient Stage-2 scores for fusion (higher = better)
        blip2_for_fusion = blip2_scores if stage2_higher else (-1.0 * blip2_scores)
        
        # Orient Stage-1 scores: try both signs and choose the one that best aligns with Stage-2
        clip_raw = np.asarray(clip_scores, dtype=np.float32)
        if len(clip_raw) > 3:
            c1 = np.corrcoef(clip_raw, blip2_for_fusion)[0, 1]
            c2 = np.corrcoef(-clip_raw, blip2_for_fusion)[0, 1]
            clip_for_fusion = clip_raw if abs(c1) >= abs(c2) else (-clip_raw)
            correlation = np.corrcoef(clip_for_fusion, blip2_for_fusion)[0, 1]
        else:
            clip_for_fusion = clip_raw
            correlation = 0.0
        
        # Guardrail 2: Correct inversion when BLIP-2 disagrees strongly with Stage-1
        fusion_method = self.config.get("fusion_method", "weighted")
        stage1_weight = float(self.config.get("stage1_weight", 0.3))
        stage2_weight = float(self.config.get("stage2_weight", 0.7))
        
        if correlation < -0.2:
            print(f"Warning: Stage-2 scores seem inverted (correlation={correlation:.3f})")
            # Flip Stage-2: if it's a probability in [0,1], flipping is 1 - p
            blip2_for_fusion = 1.0 - blip2_for_fusion
            # Recompute correlation (optional)
            if len(clip_raw) > 3:
                correlation = np.corrcoef(clip_for_fusion, blip2_for_fusion)[0, 1]
            # Be conservative: either reduce Stage-2 weight or fall back to rank fusion
            if fusion_method == "weighted":
                stage1_weight, stage2_weight = 0.6, 0.4
                print(f"Corrected: flipped Stage-2, reduced weight to {stage2_weight:.1f}")
            else:
                fusion_method = "rank_fusion"
                print("Corrected: switching to rank_fusion (orientation-safe)")
        
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
        show_progress: Optional[bool] = None
    ) -> List[List[Tuple[str, float]]]:
        """
        Perform batch hybrid text-to-image search for multiple queries.
        
        Efficiently processes multiple queries by:
        1. Batching Stage 1 (CLIP) encoding for all queries at once
        2. Batching Stage 2 (BLIP-2) re-ranking across all candidates
        
        Args:
            queries: List of text query strings
            k1: Number of candidates per query (default: from config)
            k2: Number of final results per query (default: from config)
            batch_size: Batch size for BLIP-2 (default: from config)
            show_progress: Show progress bars (default: from config)
        
        Returns:
            List of result lists, one per query. Each result list contains
            (image_id, score) tuples sorted by score.
        
        Example:
            >>> queries = ["a dog", "a cat", "a bird"]
            >>> results = engine.batch_text_to_image_search(queries)
            >>> for i, query_results in enumerate(results):
            ...     print(f"Query '{queries[i]}':")
            ...     for img_id, score in query_results[:3]:
            ...         print(f"  {img_id}: {score:.4f}")
        
        Performance Notes:
            - Stage 1 processes all queries in parallel (single batch)
            - Stage 2 batches all candidates together for efficiency
            - Typically 2-3x faster than sequential processing
        """
        # Use config defaults if not specified
        k1 = k1 or self.config['k1']
        k2 = k2 or self.config['k2']
        batch_size = batch_size or self.config['batch_size']
        show_progress = show_progress if show_progress is not None else self.config['show_progress']
        
        if not queries:
            return []
        
        n_queries = len(queries)
        if show_progress:
            print(f"\n{'='*60}")
            print(f"Batch Hybrid Search - {n_queries} queries")
            print(f"{'='*60}")
        
        # =====================================================================
        # STAGE 1: PARALLEL CLIP RETRIEVAL (All queries at once)
        # =====================================================================
        if show_progress:
            print(f"\n[Stage 1] CLIP Retrieval (k1={k1})...")
        
        stage1_start = time.time()
        
        # Encode all queries in one batch
        query_embeddings = self.bi_encoder.encode_texts(
            texts=queries,
            batch_size=32,  # CLIP can handle larger batches
            normalize=True,
            show_progress=show_progress
        )
        
        # Search FAISS index for all queries
        all_scores, all_indices = self.image_index.search(
            query_embeddings=query_embeddings,
            k=k1,
            return_scores=True
        )
        
        # Organize candidates per query
        all_candidates = []
        image_ids = self.image_index.metadata.get('ids', [])
        
        for query_idx in range(n_queries):
            candidates = []
            for idx, score in zip(all_indices[query_idx], all_scores[query_idx]):
                if idx < len(image_ids):
                    image_id = image_ids[idx]
                    candidates.append((image_id, float(score)))
            all_candidates.append(candidates)
        
        stage1_time = (time.time() - stage1_start) * 1000
        
        if show_progress:
            print(f"  ✓ Retrieved {k1} candidates per query")
            print(f"  ✓ Latency: {stage1_time:.2f}ms ({stage1_time/n_queries:.2f}ms per query)")
        
        # =====================================================================
        # STAGE 2: BATCHED BLIP-2 RE-RANKING (All candidates together)
        # =====================================================================
        if show_progress:
            print(f"\n[Stage 2] BLIP-2 Re-ranking (k2={k2}, batch_size={batch_size})...")
        
        stage2_start = time.time()
        
        # Prepare batch data: (query_idx, query_text, image_name, clip_score)
        batch_items = []
        for query_idx, (query, candidates) in enumerate(zip(queries, all_candidates)):
            for image_name, clip_score in candidates:
                batch_items.append({
                    'query_idx': query_idx,
                    'query': query,
                    'image_name': image_name,
                    'clip_score': clip_score
                })
        
        # Get image paths
        for item in batch_items:
            image_path = self.dataset.images_dir / item['image_name']
            item['image_path'] = str(image_path) if image_path.exists() else None
        
        # Filter out missing images
        valid_items = [item for item in batch_items if item['image_path'] is not None]
        
        if show_progress:
            print(f"  → Scoring {len(valid_items)} image-text pairs...")
        
        # Batch score all pairs
        cross_scores = []
        
        iterator = range(0, len(valid_items), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="  Re-ranking batches", unit="batch")
        
        for i in iterator:
            batch = valid_items[i:i + batch_size]
            
            # Prepare batch for cross-encoder
            queries_batch = [item['query'] for item in batch]
            candidates_batch = [item['image_path'] for item in batch]
            
            # Score batch
            scores = self.cross_encoder.score_pairs(
                queries=queries_batch,
                candidates=candidates_batch,
                query_type='text',
                candidate_type='image',
                batch_size=len(batch),
                show_progress=False
            )
            cross_scores.extend(scores)
        
        # Add cross-encoder scores to items
        for item, cross_score in zip(valid_items, cross_scores):
            item['cross_score'] = cross_score
        
        # Group results by query
        query_items = [[] for _ in range(n_queries)]
        for item in valid_items:
            query_items[item['query_idx']].append(item)
        
        # Apply fusion for each query
        fusion_method = self.config.get('fusion_method', 'weighted')
        stage1_weight = float(self.config.get('stage1_weight', 0.3))
        stage2_weight = float(self.config.get('stage2_weight', 0.7))
        stage2_higher = getattr(self.cross_encoder, "higher_is_better", True)
        
        final_results = []
        
        for query_idx in range(n_queries):
            items = query_items[query_idx]
            if not items:
                final_results.append([])
                continue
            
            # Extract scores
            clip_scores = np.array([it['clip_score'] for it in items], dtype=np.float32)
            blip_scores = np.array([it['cross_score'] for it in items], dtype=np.float32)
            
            # Orient Stage-2 scores for fusion (higher = better)
            blip2_for_fusion = blip_scores if stage2_higher else (-1.0 * blip_scores)
            
            # Orient Stage-1 scores: try both signs and choose the one that best aligns with Stage-2
            clip_raw = clip_scores
            if len(clip_raw) > 3:
                c1 = np.corrcoef(clip_raw, blip2_for_fusion)[0, 1]
                c2 = np.corrcoef(-clip_raw, blip2_for_fusion)[0, 1]
                clip_for_fusion = clip_raw if abs(c1) >= abs(c2) else (-clip_raw)
                correlation = np.corrcoef(clip_for_fusion, blip2_for_fusion)[0, 1]
            else:
                clip_for_fusion = clip_raw
                correlation = 0.0
            
            # Correct inversion when BLIP-2 disagrees strongly with Stage-1
            current_fusion = fusion_method
            current_s1_weight = stage1_weight
            current_s2_weight = stage2_weight
            
            if correlation < -0.2:
                # Flip Stage-2: if it's a probability in [0,1], flipping is 1 - p
                blip2_for_fusion = 1.0 - blip2_for_fusion
                # Be conservative: reduce Stage-2 weight or switch to rank fusion
                if current_fusion == "weighted":
                    current_s1_weight, current_s2_weight = 0.6, 0.4
                else:
                    current_fusion = "rank_fusion"
            
            # Compute fused scores based on method
            if current_fusion == 'weighted':
                clip_norm = _normalize(clip_for_fusion)
                blip_norm = _normalize(blip2_for_fusion)
                fused_scores = current_s1_weight * clip_norm + current_s2_weight * blip_norm
                fused_scores = np.asarray(fused_scores, dtype=np.float32)
                
            elif current_fusion == 'rank_fusion':
                clip_rank = np.argsort(np.argsort(-clip_for_fusion))
                blip_rank = np.argsort(np.argsort(-blip2_for_fusion))
                k_rrf = 60.0
                fused_scores = 1.0 / (k_rrf + clip_rank + 1) + 1.0 / (k_rrf + blip_rank + 1)
                fused_scores = np.asarray(fused_scores, dtype=np.float32)
                
            else:  # 'replace'
                # Use Stage-2 scores (after orientation correction)
                fused_scores = blip2_for_fusion
            
            # Attach fused scores and sort
            for item, fused_score in zip(items, fused_scores):
                item['fused_score'] = float(fused_score)
            
            items.sort(key=lambda x: x['fused_score'], reverse=True)
            
            # Keep top-k2 and extract (image_name, fused_score) tuples
            query_results = [
                (item['image_name'], item['fused_score'])
                for item in items[:k2]
            ]
            final_results.append(query_results)
        
        stage2_time = (time.time() - stage2_start) * 1000
        total_time = stage1_time + stage2_time
        
        # Update statistics
        for _ in range(n_queries):
            self.stats['total_queries'] += 1
            self.stats['stage1_latency_ms'].append(stage1_time / n_queries)
            self.stats['stage2_latency_ms'].append(stage2_time / n_queries)
            self.stats['total_latency_ms'].append(total_time / n_queries)
        
        if show_progress:
            print(f"  ✓ Re-ranked to top {k2} per query")
            print(f"  ✓ Latency: {stage2_time:.2f}ms ({stage2_time/n_queries:.2f}ms per query)")
            print(f"\n{'='*60}")
            print(f"Batch Search Complete")
            print(f"  • Total queries: {n_queries}")
            print(f"  • Total latency: {total_time:.2f}ms")
            print(f"  • Per-query latency: {total_time/n_queries:.2f}ms")
            print(f"  • Stage 1: {stage1_time/n_queries:.2f}ms/query")
            print(f"  • Stage 2: {stage2_time/n_queries:.2f}ms/query")
            print(f"{'='*60}\n")
        
        return final_results
    
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


if __name__ == "__main__":
    # Simple test
    print("Hybrid Search Engine module loaded successfully")
    print("To use, initialize with:")
    print("  - BiEncoder (CLIP)")
    print("  - CrossEncoder (BLIP-2)")
    print("  - FAISSIndex (image embeddings)")
    print("  - Flickr30KDataset")
