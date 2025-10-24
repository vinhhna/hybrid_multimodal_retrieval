"""
Quick validation script for the MultimodalSearchEngine.

Tests all search modes with sample queries.
"""

import warnings
warnings.filterwarnings('ignore', category=UserWarning)

import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

from retrieval import BiEncoder, FAISSIndex, MultimodalSearchEngine
from flickr30k import Flickr30KDataset
from flickr30k.utils import load_config


def main():
    print("=" * 60)
    print("MULTIMODAL SEARCH ENGINE VALIDATION")
    print("=" * 60)
    
    # Load components
    print("\n📂 Loading components...")
    
    encoder = BiEncoder(model_name='ViT-B-32', pretrained='openai')
    
    faiss_config = load_config('configs/faiss_config.yaml')
    image_index = FAISSIndex()
    image_index.load(faiss_config['paths']['image_index'])
    
    text_index = FAISSIndex()
    text_index.load(faiss_config['paths']['text_index'])
    
    dataset = Flickr30KDataset(
        images_dir='data/images',
        captions_file='data/results.csv'
    )
    
    # Initialize search engine
    search_engine = MultimodalSearchEngine(
        encoder=encoder,
        image_index=image_index,
        text_index=text_index,
        dataset=dataset
    )
    
    print("✓ Search engine initialized")
    
    # Test 1: Text-to-Image
    print("\n" + "=" * 60)
    print("TEST 1: TEXT-TO-IMAGE SEARCH")
    print("=" * 60)
    
    test_queries = [
        "A dog playing in the park",
        "Children at the beach",
        "Person riding a bicycle"
    ]
    
    for query in test_queries:
        result = search_engine.text_to_image_search(query, k=5)
        print(f"\nQuery: '{query}'")
        print(f"  Top 3 results:")
        for i, (img_name, score) in enumerate(list(result)[:3], 1):
            print(f"    {i}. {img_name} ({score:.4f})")
        
        stats = search_engine.get_performance_stats()
        print(f"  Performance: {stats['total_time_ms']:.2f}ms")
    
    # Test 2: Image-to-Text
    print("\n" + "=" * 60)
    print("TEST 2: IMAGE-TO-TEXT SEARCH")
    print("=" * 60)
    
    sample_images = dataset.get_unique_images()[:3]
    
    for img_name in sample_images:
        img_path = Path('data/images') / img_name
        result = search_engine.image_to_text_search(img_path, k=5)
        
        print(f"\nQuery image: {img_name}")
        print(f"  Top 3 captions:")
        for i, (caption, score) in enumerate(list(result)[:3], 1):
            print(f"    {i}. [{score:.4f}] {caption[:60]}...")
    
    # Test 3: Image-to-Image
    print("\n" + "=" * 60)
    print("TEST 3: IMAGE-TO-IMAGE SEARCH")
    print("=" * 60)
    
    for img_name in sample_images[:2]:
        img_path = Path('data/images') / img_name
        result = search_engine.image_to_image_search(img_path, k=5)
        
        print(f"\nQuery image: {img_name}")
        print(f"  Top 3 similar images (excluding self):")
        for i, (similar_img, score) in enumerate(list(result)[1:4], 1):
            print(f"    {i}. {similar_img} ({score:.4f})")
    
    # Test 4: Batch Search
    print("\n" + "=" * 60)
    print("TEST 4: BATCH SEARCH")
    print("=" * 60)
    
    batch_queries = [
        "Mountain landscape",
        "City street",
        "Ocean waves"
    ]
    
    results = search_engine.batch_search(
        queries=batch_queries,
        search_type='text_to_image',
        k=3
    )
    
    print(f"\nBatch of {len(batch_queries)} queries:")
    for query, result in zip(batch_queries, results):
        top_img = result.ids[0]
        top_score = result.scores[0]
        print(f"  '{query}' -> {top_img} ({top_score:.4f})")
    
    # Summary
    print("\n" + "=" * 60)
    print("✅ ALL TESTS PASSED!")
    print("=" * 60)
    print("\nSearch Engine Features:")
    print("  ✓ Text-to-Image search")
    print("  ✓ Image-to-Text search")
    print("  ✓ Image-to-Image search")
    print("  ✓ Batch search")
    print("  ✓ Performance tracking")
    print("\n🎉 Week 3 Implementation Complete!")


if __name__ == '__main__':
    main()
