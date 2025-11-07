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
    # Load dataset
    dataset = Flickr30KDataset('/kaggle/input/flickr30k/data/images', '/kaggle/input/flickr30k/data/results.csv')
    
    # Load encoder
    encoder = CLIPEncoder()
    
    # Load indices
    image_index = FAISSIndex()
    image_index.load('/kaggle/input/flickr30k/data/indices/image_index.faiss')
    
    text_index = FAISSIndex()
    text_index.load('/kaggle/input/flickr30k/data/indices/text_index.faiss')
    
    # Create basic search engine
    engine = SearchEngine(encoder, image_index, text_index, dataset)
    
    # Text-to-Image search
    query = "a dog playing in the park"
    print(f"Query: '{query}'\n")
    
    results = engine.text_to_image(query, k=5)
    for i, (img_name, score) in enumerate(results, 1):
        print(f"{i}. {img_name} (score: {score:.4f})")
    
    # Hybrid search
    reranker = BLIP2Reranker()
    
    hybrid_engine = HybridSearchEngine(encoder, reranker, image_index, dataset)
    
    print(f"\nQuery: '{query}'")
    hybrid_results = hybrid_engine.search(query, k1=50, k2=5)
    
    print("\nTop 5 results after re-ranking:")
    for i, (img_name, score) in enumerate(hybrid_results, 1):
        print(f"{i}. {img_name} (score: {score:.4f})")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
