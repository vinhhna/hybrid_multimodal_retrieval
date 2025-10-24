"""
Script to build FAISS indices from generated embeddings.

This script loads the pre-computed embeddings and builds FAISS indices
for fast similarity search.

Usage:
    python scripts/build_faiss_indices.py
"""

import warnings
warnings.filterwarnings('ignore', category=UserWarning)

import sys
from pathlib import Path
import yaml
import numpy as np

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

from retrieval import BiEncoder, FAISSIndex


def load_config(config_path: str = 'configs/faiss_config.yaml') -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def build_image_index(config: dict):
    """Build FAISS index for image embeddings."""
    print("\n" + "=" * 60)
    print("BUILDING IMAGE INDEX")
    print("=" * 60)
    
    # Load embeddings
    print("\n📂 Loading image embeddings...")
    encoder = BiEncoder()
    embeddings, metadata = encoder.load_embeddings(
        'data/embeddings/image_embeddings.npy'
    )
    
    print(f"✓ Loaded {embeddings.shape[0]:,} image embeddings")
    print(f"  Dimension: {embeddings.shape[1]}")
    
    # Create index
    print(f"\n🔨 Creating FAISS index (type={config['index']['type']})...")
    index = FAISSIndex(
        dimension=config['index']['dimension'],
        index_type=config['index']['type'],
        metric=config['index']['metric'],
        **config.get(config['index']['type'], {})
    )
    
    # Train if needed
    if config['index']['type'] == 'ivf':
        index.train(embeddings)
    
    # Add embeddings
    print(f"\n➕ Adding embeddings to index...")
    index.add(
        embeddings,
        ids=metadata.get('image_names'),
        metadata={
            'num_images': metadata.get('num_images'),
            'model': metadata.get('model'),
            'embedding_dim': metadata.get('embedding_dim')
        }
    )
    
    # Save index
    print(f"\n💾 Saving index...")
    index.save(
        config['paths']['image_index'],
        config['paths']['image_index_metadata']
    )
    
    # Print stats
    print(f"\n📊 Index Statistics:")
    stats = index.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    return index


def build_text_index(config: dict):
    """Build FAISS index for text embeddings."""
    print("\n" + "=" * 60)
    print("BUILDING TEXT INDEX")
    print("=" * 60)
    
    # Load embeddings
    print("\n📂 Loading text embeddings...")
    encoder = BiEncoder()
    embeddings, metadata = encoder.load_embeddings(
        'data/embeddings/text_embeddings.npy'
    )
    
    print(f"✓ Loaded {embeddings.shape[0]:,} text embeddings")
    print(f"  Dimension: {embeddings.shape[1]}")
    
    # Create index
    print(f"\n🔨 Creating FAISS index (type={config['index']['type']})...")
    index = FAISSIndex(
        dimension=config['index']['dimension'],
        index_type=config['index']['type'],
        metric=config['index']['metric'],
        **config.get(config['index']['type'], {})
    )
    
    # Train if needed
    if config['index']['type'] == 'ivf':
        # Use subset for training if too large
        training_size = min(len(embeddings), 100000)
        training_embeddings = embeddings[:training_size]
        print(f"  Using {training_size:,} samples for training")
        index.train(training_embeddings)
    
    # Add embeddings
    print(f"\n➕ Adding embeddings to index...")
    index.add(
        embeddings,
        metadata={
            'num_captions': metadata.get('num_captions'),
            'num_images': metadata.get('num_images'),
            'model': metadata.get('model'),
            'embedding_dim': metadata.get('embedding_dim')
        }
    )
    
    # Save index
    print(f"\n💾 Saving index...")
    index.save(
        config['paths']['text_index'],
        config['paths']['text_index_metadata']
    )
    
    # Print stats
    print(f"\n📊 Index Statistics:")
    stats = index.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    return index


def test_indices(config: dict):
    """Quick test of built indices."""
    print("\n" + "=" * 60)
    print("TESTING INDICES")
    print("=" * 60)
    
    # Load indices
    print("\n📂 Loading indices...")
    image_index = FAISSIndex()
    image_index.load(config['paths']['image_index'])
    
    text_index = FAISSIndex()
    text_index.load(config['paths']['text_index'])
    
    # Test search with random query
    print("\n🔍 Testing search...")
    
    # Random image query
    random_query = np.random.randn(1, 512).astype('float32')
    scores, indices = image_index.search(random_query, k=5)
    print(f"\n✓ Image index search successful")
    print(f"  Top 5 scores: {scores[0]}")
    
    # Random text query
    random_query = np.random.randn(1, 512).astype('float32')
    scores, indices = text_index.search(random_query, k=5)
    print(f"\n✓ Text index search successful")
    print(f"  Top 5 scores: {scores[0]}")


def main():
    """Main function to build all indices."""
    print("=" * 60)
    print("FAISS INDEX BUILDER")
    print("=" * 60)
    
    # Load configuration
    config = load_config()
    print(f"\n✓ Configuration loaded")
    print(f"  Index type: {config['index']['type']}")
    print(f"  Metric: {config['index']['metric']}")
    
    # Build indices
    image_index = build_image_index(config)
    text_index = build_text_index(config)
    
    # Test indices
    test_indices(config)
    
    print("\n" + "=" * 60)
    print("✅ ALL INDICES BUILT SUCCESSFULLY!")
    print("=" * 60)
    print(f"\nImage index: {config['paths']['image_index']}")
    print(f"Text index: {config['paths']['text_index']}")


if __name__ == '__main__':
    main()
