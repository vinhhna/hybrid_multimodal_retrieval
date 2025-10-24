"""
FAISS Index module for fast similarity search.

This module provides a wrapper around FAISS indices for efficient
similarity search on embeddings.
"""

import numpy as np
import faiss
import json
from pathlib import Path
from typing import Tuple, List, Optional, Dict


class FAISSIndex:
    """
    FAISS Index wrapper for similarity search.
    
    Supports multiple index types:
    - 'flat': Exact search (IndexFlatIP)
    - 'ivf': Approximate search (IndexIVFFlat)
    - 'hnsw': Graph-based search (IndexHNSWFlat)
    
    Attributes:
        dimension (int): Embedding dimension
        index_type (str): Type of FAISS index
        index: The FAISS index object
        metadata (dict): Metadata about indexed items
    """
    
    def __init__(
        self,
        dimension: int = 512,
        index_type: str = 'flat',
        metric: str = 'cosine',
        **kwargs
    ):
        """
        Initialize FAISS index.
        
        Args:
            dimension: Dimension of embeddings (default: 512 for CLIP)
            index_type: Type of index ('flat', 'ivf', 'hnsw')
            metric: Distance metric ('cosine' or 'euclidean')
            **kwargs: Additional parameters for specific index types
        """
        self.dimension = dimension
        self.index_type = index_type
        self.metric = metric
        self.index = None
        self.metadata = {}
        self.is_trained = False
        
        # Create index based on type
        self._create_index(**kwargs)
    
    def _create_index(self, **kwargs):
        """Create FAISS index based on specified type."""
        if self.metric == 'cosine':
            # For cosine similarity, use Inner Product after normalization
            if self.index_type == 'flat':
                self.index = faiss.IndexFlatIP(self.dimension)
                self.is_trained = True
            
            elif self.index_type == 'ivf':
                nlist = kwargs.get('nlist', 100)  # Number of clusters
                quantizer = faiss.IndexFlatIP(self.dimension)
                self.index = faiss.IndexIVFFlat(
                    quantizer, self.dimension, nlist, faiss.METRIC_INNER_PRODUCT
                )
            
            elif self.index_type == 'hnsw':
                m = kwargs.get('m', 32)  # Number of connections
                self.index = faiss.IndexHNSWFlat(self.dimension, m, faiss.METRIC_INNER_PRODUCT)
                self.is_trained = True
            
            else:
                raise ValueError(f"Unknown index type: {self.index_type}")
        
        elif self.metric == 'euclidean':
            # For Euclidean distance
            if self.index_type == 'flat':
                self.index = faiss.IndexFlatL2(self.dimension)
                self.is_trained = True
            
            elif self.index_type == 'ivf':
                nlist = kwargs.get('nlist', 100)
                quantizer = faiss.IndexFlatL2(self.dimension)
                self.index = faiss.IndexIVFFlat(
                    quantizer, self.dimension, nlist, faiss.METRIC_L2
                )
            
            else:
                raise ValueError(f"Index type {self.index_type} not supported for euclidean")
        
        else:
            raise ValueError(f"Unknown metric: {self.metric}")
    
    def train(self, embeddings: np.ndarray):
        """
        Train the index (required for IVF indices).
        
        Args:
            embeddings: Training embeddings (n_samples, dimension)
        """
        if self.is_trained:
            print("Index already trained or doesn't require training")
            return
        
        print(f"Training index on {len(embeddings):,} vectors...")
        self.index.train(embeddings)
        self.is_trained = True
        print("✓ Index trained")
    
    def add(
        self,
        embeddings: np.ndarray,
        ids: Optional[List[str]] = None,
        metadata: Optional[Dict] = None
    ):
        """
        Add embeddings to the index.
        
        Args:
            embeddings: Embeddings to add (n_samples, dimension)
            ids: Optional IDs for each embedding
            metadata: Optional metadata dictionary
        """
        if not self.is_trained:
            raise ValueError("Index must be trained before adding vectors")
        
        # Ensure embeddings are normalized for cosine similarity
        if self.metric == 'cosine':
            faiss.normalize_L2(embeddings)
        
        # Add to index
        self.index.add(embeddings)
        
        # Store metadata
        if ids is not None:
            self.metadata['ids'] = ids
        if metadata is not None:
            self.metadata.update(metadata)
        
        print(f"✓ Added {len(embeddings):,} vectors to index")
        print(f"  Total vectors in index: {self.index.ntotal:,}")
    
    def search(
        self,
        query_embeddings: np.ndarray,
        k: int = 10,
        return_scores: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for k nearest neighbors.
        
        Args:
            query_embeddings: Query vectors (n_queries, dimension)
            k: Number of neighbors to return
            return_scores: Whether to return similarity scores
        
        Returns:
            Tuple of (scores, indices) if return_scores=True
            Otherwise just indices
        """
        if self.index.ntotal == 0:
            raise ValueError("Index is empty. Add embeddings first.")
        
        # Normalize query for cosine similarity
        if self.metric == 'cosine':
            query_embeddings = query_embeddings.copy()
            faiss.normalize_L2(query_embeddings)
        
        # Search
        scores, indices = self.index.search(query_embeddings, k)
        
        if return_scores:
            return scores, indices
        return indices
    
    def save(self, index_path: str, metadata_path: Optional[str] = None):
        """
        Save index and metadata to disk.
        
        Args:
            index_path: Path to save FAISS index
            metadata_path: Path to save metadata (auto-generated if None)
        """
        index_path = Path(index_path)
        index_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, str(index_path))
        print(f"✓ Index saved to: {index_path}")
        
        # Save metadata
        if metadata_path is None:
            metadata_path = index_path.with_suffix('.json')
        
        metadata_to_save = {
            'dimension': self.dimension,
            'index_type': self.index_type,
            'metric': self.metric,
            'ntotal': self.index.ntotal,
            'is_trained': self.is_trained,
            **self.metadata
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata_to_save, f, indent=2)
        print(f"✓ Metadata saved to: {metadata_path}")
    
    def load(self, index_path: str, metadata_path: Optional[str] = None):
        """
        Load index and metadata from disk.
        
        Args:
            index_path: Path to FAISS index file
            metadata_path: Path to metadata file (auto-generated if None)
        """
        index_path = Path(index_path)
        
        # Load FAISS index
        self.index = faiss.read_index(str(index_path))
        self.is_trained = True
        print(f"✓ Index loaded from: {index_path}")
        print(f"  Vectors in index: {self.index.ntotal:,}")
        
        # Load metadata
        if metadata_path is None:
            metadata_path = index_path.with_suffix('.json')
        
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                loaded_metadata = json.load(f)
            
            self.dimension = loaded_metadata.get('dimension', self.dimension)
            self.index_type = loaded_metadata.get('index_type', self.index_type)
            self.metric = loaded_metadata.get('metric', self.metric)
            
            # Store remaining metadata
            for key in ['dimension', 'index_type', 'metric', 'ntotal', 'is_trained']:
                loaded_metadata.pop(key, None)
            self.metadata = loaded_metadata
            
            print(f"✓ Metadata loaded from: {metadata_path}")
    
    def get_stats(self) -> Dict:
        """Get index statistics."""
        return {
            'dimension': self.dimension,
            'index_type': self.index_type,
            'metric': self.metric,
            'ntotal': self.index.ntotal,
            'is_trained': self.is_trained,
            'metadata_keys': list(self.metadata.keys())
        }
    
    def __repr__(self) -> str:
        return (
            f"FAISSIndex(type={self.index_type}, "
            f"metric={self.metric}, "
            f"dim={self.dimension}, "
            f"n={self.index.ntotal:,})"
        )
