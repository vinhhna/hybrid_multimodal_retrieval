"""
Simple demo of the multimodal search engine.
Run this after setting up the data and building indices.
"""

from src.dataset import Flickr30KDataset
from src.encoder import CLIPEncoder
from src.index import FAISSIndex
from src.search import SearchEngine


def main():
    print("=== Multimodal Search Demo ===\n")
    
    # Load dataset
    print("Loading dataset...")
    dataset = Flickr30KDataset('data/images', 'data/results.csv')
    
    # Load encoder
    print("\nLoading CLIP encoder...")
    encoder = CLIPEncoder()
    
    # Load indices
    print("\nLoading FAISS indices...")
    image_index = FAISSIndex()
    image_index.load('data/image_index.faiss')
    
    text_index = FAISSIndex()
    text_index.load('data/text_index.faiss')
    
    # Create search engine
    print("\nInitializing search engine...")
    engine = SearchEngine(encoder, image_index, text_index, dataset)
    
    # Text-to-Image search
    print("\n=== Text-to-Image Search ===")
    query = "a dog playing in the park"
    print(f"Query: '{query}'\n")
    
    results = engine.text_to_image(query, k=5)
    for i, (img_name, score) in enumerate(results, 1):
        print(f"{i}. {img_name} (score: {score:.4f})")
    
    # Image-to-Text search
    print("\n=== Image-to-Text Search ===")
    test_image = results[0][0]  # Use first result from above
    print(f"Query image: {test_image}\n")
    
    captions = engine.image_to_text(f'data/images/{test_image}', k=3)
    for i, (caption, score) in enumerate(captions, 1):
        print(f"{i}. {caption} (score: {score:.4f})")
    
    # Image-to-Image search
    print("\n=== Image-to-Image Search ===")
    print(f"Query image: {test_image}\n")
    
    similar = engine.image_to_image(f'data/images/{test_image}', k=5)
    for i, (img_name, score) in enumerate(similar, 1):
        print(f"{i}. {img_name} (score: {score:.4f})")
    
    print("\n=== Demo Complete! ===")


if __name__ == "__main__":
    main()
