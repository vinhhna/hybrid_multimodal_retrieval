"""
Demo of multimodal and hybrid search.
Run this after setting up the data and building indices.
"""

from src.dataset import Flickr30KDataset
from src.encoder import CLIPEncoder
from src.index import FAISSIndex
from src.search import SearchEngine
from src.reranker import BLIP2Reranker
from src.hybrid_search import HybridSearchEngine


def main():
    print("=== Multimodal Search Demo ===\n")
    
    # Load dataset
    print("Loading dataset...")
    dataset = Flickr30KDataset('/kaggle/input/flickr30k/data/images', '/kaggle/input/flickr30k/data/results.csv')
    
    # Load encoder
    print("\nLoading CLIP encoder...")
    encoder = CLIPEncoder()
    
    # Load indices
    print("\nLoading FAISS indices...")
    image_index = FAISSIndex()
    image_index.load('/kaggle/input/flickr30k/data/indices/image_index.faiss')
    
    text_index = FAISSIndex()
    text_index.load('/kaggle/input/flickr30k/data/indices/text_index.faiss')
    
    # Create basic search engine
    print("\nInitializing search engine...")
    engine = SearchEngine(encoder, image_index, text_index, dataset)
    
    # Text-to-Image search
    print("\n=== Text-to-Image Search ===")
    query = "a dog playing in the park"
    print(f"Query: '{query}'\n")
    
    results = engine.text_to_image(query, k=5)
    for i, (img_name, score) in enumerate(results, 1):
        print(f"{i}. {img_name} (score: {score:.4f})")
    
    # Hybrid search
    print("\n\n=== Hybrid Search (CLIP + BLIP-2) ===")
    print("Loading BLIP-2 re-ranker...")
    reranker = BLIP2Reranker()
    
    hybrid_engine = HybridSearchEngine(encoder, reranker, image_index, dataset)
    
    print(f"\nQuery: '{query}'")
    hybrid_results = hybrid_engine.search(query, k1=50, k2=5)
    
    print("\nTop 5 results after re-ranking:")
    for i, (img_name, score) in enumerate(hybrid_results, 1):
        print(f"{i}. {img_name} (score: {score:.4f})")
    
    print("\n=== Demo Complete! ===")


if __name__ == "__main__":
    main()
