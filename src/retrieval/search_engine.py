"""
Multimodal Search Engine combining BiEncoder and FAISS.

This module provides a high-level interface for multimodal search,
supporting text-to-image, image-to-text, and image-to-image search.
"""

from pathlib import Path
from typing import List, Union, Tuple, Dict, Optional
from PIL import Image
import time

from .bi_encoder import BiEncoder
from .faiss_index import FAISSIndex


class SearchResult:
    """Container for search results."""
    
    def __init__(self, ids: List[str], scores: List[float], metadata: Optional[Dict] = None):
        """
        Initialize search result.
        
        Args:
            ids: List of result IDs (image names or caption indices)
            scores: Similarity scores
            metadata: Optional metadata dictionary
        """
        self.ids = ids
        self.scores = scores
        self.metadata = metadata or {}
    
    def __repr__(self) -> str:
        return f"SearchResult(n={len(self.ids)}, top_score={self.scores[0]:.4f})"
    
    def __len__(self) -> int:
        return len(self.ids)
    
    def __getitem__(self, idx: int) -> Tuple[str, float]:
        """Get result at index."""
        return self.ids[idx], self.scores[idx]


class MultimodalSearchEngine:
    """
    Multimodal search engine for Flickr30K dataset.
    
    Supports:
    - Text-to-Image search
    - Image-to-Text search  
    - Image-to-Image search
    - Batch search for all modes
    
    Attributes:
        encoder: BiEncoder for embedding generation
        image_index: FAISS index for images
        text_index: FAISS index for text captions
        dataset: Flickr30K dataset (optional)
    """
    
    def __init__(
        self,
        encoder: BiEncoder,
        image_index: FAISSIndex,
        text_index: FAISSIndex,
        dataset = None
    ):
        """
        Initialize search engine.
        
        Args:
            encoder: BiEncoder instance
            image_index: FAISS index for images
            text_index: FAISS index for text captions
            dataset: Optional Flickr30K dataset for metadata
        """
        self.encoder = encoder
        self.image_index = image_index
        self.text_index = text_index
        self.dataset = dataset
        
        # Performance tracking
        self._last_search_time = 0.0
        self._last_encode_time = 0.0
    
    def text_to_image_search(
        self,
        query_text: Union[str, List[str]],
        k: int = 10,
        return_metadata: bool = True
    ) -> Union[SearchResult, List[SearchResult]]:
        """
        Search images using text query.
        
        Args:
            query_text: Text query or list of queries
            k: Number of results to return
            return_metadata: Whether to include metadata
        
        Returns:
            SearchResult or list of SearchResults
        """
        # Handle single vs batch
        is_single = isinstance(query_text, str)
        queries = [query_text] if is_single else query_text
        
        # Encode queries
        start_time = time.time()
        query_embeddings = self.encoder.encode_texts(queries, show_progress=False)
        self._last_encode_time = time.time() - start_time
        
        # Search
        start_time = time.time()
        scores, indices = self.image_index.search(query_embeddings, k=k)
        self._last_search_time = time.time() - start_time
        
        # Build results
        results = []
        for i in range(len(queries)):
            # Get image names
            image_ids = [self.image_index.metadata['ids'][idx] for idx in indices[i]]
            result_scores = scores[i].tolist()
            
            # Add metadata
            metadata = {
                'query': queries[i],
                'search_type': 'text_to_image',
                'k': k,
                'encode_time_ms': self._last_encode_time * 1000,
                'search_time_ms': self._last_search_time * 1000
            } if return_metadata else {}
            
            results.append(SearchResult(image_ids, result_scores, metadata))
        
        return results[0] if is_single else results
    
    def image_to_text_search(
        self,
        query_image: Union[str, Path, Image.Image, List],
        k: int = 10,
        return_metadata: bool = True
    ) -> Union[SearchResult, List[SearchResult]]:
        """
        Search captions using image query.
        
        Args:
            query_image: Image path, PIL Image, or list of images
            k: Number of results to return
            return_metadata: Whether to include metadata
        
        Returns:
            SearchResult or list of SearchResults
        """
        # Handle single vs batch
        is_single = not isinstance(query_image, list)
        queries = [query_image] if is_single else query_image
        
        # Load images if paths
        images = []
        for q in queries:
            if isinstance(q, (str, Path)):
                images.append(Image.open(q))
            else:
                images.append(q)
        
        # Encode queries
        start_time = time.time()
        query_embeddings = self.encoder.encode_images(images, show_progress=False)
        self._last_encode_time = time.time() - start_time
        
        # Search
        start_time = time.time()
        scores, indices = self.text_index.search(query_embeddings, k=k)
        self._last_search_time = time.time() - start_time
        
        # Build results
        results = []
        for i in range(len(queries)):
            # Caption indices are just the FAISS indices
            caption_ids = [str(idx) for idx in indices[i]]
            result_scores = scores[i].tolist()
            
            # Get actual captions if dataset available
            if self.dataset is not None:
                captions = [self.dataset.df.iloc[idx]['caption'] for idx in indices[i]]
                caption_ids = captions
            
            # Add metadata
            query_name = queries[i] if isinstance(queries[i], (str, Path)) else "PIL_Image"
            metadata = {
                'query': str(query_name),
                'search_type': 'image_to_text',
                'k': k,
                'encode_time_ms': self._last_encode_time * 1000,
                'search_time_ms': self._last_search_time * 1000
            } if return_metadata else {}
            
            results.append(SearchResult(caption_ids, result_scores, metadata))
        
        return results[0] if is_single else results
    
    def image_to_image_search(
        self,
        query_image: Union[str, Path, Image.Image, List],
        k: int = 10,
        return_metadata: bool = True
    ) -> Union[SearchResult, List[SearchResult]]:
        """
        Search similar images using image query.
        
        Args:
            query_image: Image path, PIL Image, or list of images
            k: Number of results to return
            return_metadata: Whether to include metadata
        
        Returns:
            SearchResult or list of SearchResults
        """
        # Handle single vs batch
        is_single = not isinstance(query_image, list)
        queries = [query_image] if is_single else query_image
        
        # Load images if paths
        images = []
        for q in queries:
            if isinstance(q, (str, Path)):
                images.append(Image.open(q))
            else:
                images.append(q)
        
        # Encode queries
        start_time = time.time()
        query_embeddings = self.encoder.encode_images(images, show_progress=False)
        self._last_encode_time = time.time() - start_time
        
        # Search
        start_time = time.time()
        scores, indices = self.image_index.search(query_embeddings, k=k)
        self._last_search_time = time.time() - start_time
        
        # Build results
        results = []
        for i in range(len(queries)):
            # Get image names
            image_ids = [self.image_index.metadata['ids'][idx] for idx in indices[i]]
            result_scores = scores[i].tolist()
            
            # Add metadata
            query_name = queries[i] if isinstance(queries[i], (str, Path)) else "PIL_Image"
            metadata = {
                'query': str(query_name),
                'search_type': 'image_to_image',
                'k': k,
                'encode_time_ms': self._last_encode_time * 1000,
                'search_time_ms': self._last_search_time * 1000
            } if return_metadata else {}
            
            results.append(SearchResult(image_ids, result_scores, metadata))
        
        return results[0] if is_single else results
    
    def batch_search(
        self,
        queries: List,
        search_type: str = 'text_to_image',
        k: int = 10,
        return_metadata: bool = True
    ) -> List[SearchResult]:
        """
        Perform batch search for multiple queries.
        
        Args:
            queries: List of text queries or images
            search_type: 'text_to_image', 'image_to_text', or 'image_to_image'
            k: Number of results per query
            return_metadata: Whether to include metadata
        
        Returns:
            List of SearchResults
        """
        if search_type == 'text_to_image':
            return self.text_to_image_search(queries, k, return_metadata)
        elif search_type == 'image_to_text':
            return self.image_to_text_search(queries, k, return_metadata)
        elif search_type == 'image_to_image':
            return self.image_to_image_search(queries, k, return_metadata)
        else:
            raise ValueError(f"Unknown search type: {search_type}")
    
    def get_performance_stats(self) -> Dict:
        """Get performance statistics from last search."""
        return {
            'encode_time_ms': self._last_encode_time * 1000,
            'search_time_ms': self._last_search_time * 1000,
            'total_time_ms': (self._last_encode_time + self._last_search_time) * 1000,
            'qps': 1.0 / (self._last_encode_time + self._last_search_time) if (self._last_encode_time + self._last_search_time) > 0 else 0
        }
    
    def __repr__(self) -> str:
        return (
            f"MultimodalSearchEngine(\n"
            f"  encoder={self.encoder},\n"
            f"  image_index={self.image_index},\n"
            f"  text_index={self.text_index}\n"
            f")"
        )
