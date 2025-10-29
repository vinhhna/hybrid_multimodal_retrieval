"""
Script to generate embeddings for all Flickr30K images.

This script processes all images in the Flickr30K dataset and generates
CLIP embeddings, saving them to disk for later use.

Usage:
    python scripts/generate_image_embeddings.py
"""

import sys
from pathlib import Path
import yaml
import numpy as np
from tqdm import tqdm

# Import from src
try:
    from src.retrieval import BiEncoder
    from src.flickr30k import Flickr30KDataset
except ImportError:
    # If import fails, add project root to path
    project_root = Path(__file__).resolve().parent.parent
    if project_root not in sys.path:
        sys.path.insert(0, str(project_root))
    from src.retrieval import BiEncoder
    from src.flickr30k import Flickr30KDataset


def load_config(config_path: str = 'configs/clip_config.yaml') -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    """Main function to generate image embeddings."""
    print("=" * 60)
    print("FLICKR30K IMAGE EMBEDDING GENERATION")
    print("=" * 60)
    
    # Load configuration
    config = load_config()
    print(f"\n✓ Configuration loaded")
    print(f"  Model: {config['model']['name']}")
    print(f"  Batch size: {config['encoding']['image_batch_size']}")
    
    # Initialize dataset
    print(f"\n📂 Loading dataset...")
    dataset = Flickr30KDataset(
        images_dir=config['dataset']['images_dir'],
        captions_file=config['dataset']['captions_file']
    )
    
    image_names = dataset.get_unique_images()
    print(f"✓ Found {len(image_names):,} images")
    
    # Initialize encoder
    print(f"\n🤖 Loading CLIP model...")
    encoder = BiEncoder(
        model_name=config['model']['name'],
        pretrained=config['model']['pretrained']
    )
    
    # Generate embeddings
    print(f"\n🔄 Generating embeddings...")
    
    # Load images in batches
    all_embeddings = []
    batch_size = config['encoding']['image_batch_size']
    num_batches = (len(image_names) + batch_size - 1) // batch_size
    
    for i in tqdm(range(num_batches), desc="Processing batches"):
        batch_names = image_names[i * batch_size:(i + 1) * batch_size]
        batch_paths = [
            Path(config['dataset']['images_dir']) / name 
            for name in batch_names
        ]
        
        # Encode batch
        batch_embeddings = encoder.encode_images(
            batch_paths,
            batch_size=batch_size,
            normalize=config['encoding']['normalize'],
            show_progress=False
        )
        
        all_embeddings.append(batch_embeddings)
    
    # Combine all embeddings
    embeddings = np.vstack(all_embeddings)
    print(f"\n✓ Generated embeddings shape: {embeddings.shape}")
    
    # Save embeddings
    print(f"\n💾 Saving embeddings...")
    metadata = {
        'num_images': len(image_names),
        'image_names': image_names,
        'model': config['model']['name'],
        'pretrained': config['model']['pretrained'],
        'embedding_dim': embeddings.shape[1],
        'normalized': config['encoding']['normalize']
    }
    
    encoder.save_embeddings(
        embeddings,
        config['paths']['image_embeddings'],
        metadata=metadata
    )
    
    print(f"\n{'=' * 60}")
    print(f"✅ COMPLETED!")
    print(f"{'=' * 60}")
    print(f"Total images processed: {len(image_names):,}")
    print(f"Embedding dimension: {embeddings.shape[1]}")
    print(f"Output file: {config['paths']['image_embeddings']}")
    print(f"Metadata file: {config['paths']['image_metadata']}")


if __name__ == '__main__':
    main()
