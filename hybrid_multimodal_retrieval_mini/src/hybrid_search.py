"""Hybrid search combining CLIP and BLIP-2."""

from pathlib import Path


class HybridSearchEngine:
    """Two-stage search: fast CLIP retrieval + accurate BLIP-2 re-ranking."""
    
    def __init__(self, encoder, reranker, image_index, dataset):
        self.encoder = encoder
        self.reranker = reranker
        self.image_index = image_index
        self.dataset = dataset
    
    def search(self, query_text, k1=100, k2=10):
        """
        Hybrid search pipeline:
        1. Use CLIP to get top k1 candidates (fast)
        2. Use BLIP-2 to re-rank to top k2 (accurate)
        """
        print(f"\n[Stage 1] CLIP retrieval (k1={k1})...")
        
        # Stage 1: Fast CLIP search
        query_emb = self.encoder.encode_texts([query_text])
        scores, indices = self.image_index.search(query_emb, k=k1)
        
        # Get candidate images
        candidates = []
        for idx, score in zip(indices[0], scores[0]):
            image_name = self.image_index.ids[idx]
            candidates.append((image_name, float(score)))
        
        print(f"  Retrieved {len(candidates)} candidates")
        
        # Stage 2: BLIP-2 re-ranking
        print(f"\n[Stage 2] BLIP-2 re-ranking (k2={k2})...")
        
        # Prepare data for re-ranking
        image_names = [name for name, _ in candidates]
        image_paths = [str(self.dataset.images_dir / name) for name in image_names]
        texts = [query_text] * len(candidates)
        
        # Score with BLIP-2
        blip2_scores = self.reranker.score_pairs(texts, image_paths, batch_size=4)
        
        # Create re-ranked results
        reranked = [(name, score) for name, score in zip(image_names, blip2_scores)]
        reranked.sort(key=lambda x: x[1], reverse=True)
        
        print(f"  Re-ranked to top {k2} results")
        
        return reranked[:k2]
