"""FAISS index for fast similarity search."""

import numpy as np
import faiss
import json
from pathlib import Path


class FAISSIndex:
    """FAISS index wrapper."""
    
    def __init__(self, dimension=512):
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        self.ids = []
    
    def add(self, embeddings, ids=None):
        """Add embeddings to index."""
        faiss.normalize_L2(embeddings)  # Normalize for cosine similarity
        self.index.add(embeddings)
        if ids:
            self.ids = ids
        print(f"Added {len(embeddings)} vectors. Total: {self.index.ntotal}")
    
    def search(self, query_embeddings, k=10):
        """Search for k nearest neighbors."""
        query_embeddings = query_embeddings.copy()
        faiss.normalize_L2(query_embeddings)
        scores, indices = self.index.search(query_embeddings, k)
        return scores, indices
    
    def save(self, path):
        """Save index to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        faiss.write_index(self.index, str(path))
        
        # Save metadata
        metadata = {'dimension': self.dimension, 'ids': self.ids}
        with open(path.with_suffix('.json'), 'w') as f:
            json.dump(metadata, f)
        print(f"Index saved to {path}")
    
    def load(self, path):
        """Load index from disk."""
        self.index = faiss.read_index(str(path))
        
        # Load metadata
        metadata_path = Path(path).with_suffix('.json')
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            self.dimension = metadata.get('dimension', self.dimension)
            self.ids = metadata.get('ids', [])
        print(f"Index loaded: {self.index.ntotal} vectors")
