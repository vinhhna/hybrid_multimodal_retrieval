"""
Script to generate embeddings for all Flickr30K captions.

This script processes all captions in the Flickr30K dataset and generates
CLIP text embeddings, saving them to disk for later use.

Usage:
    python scripts/generate_text_embeddings.py
"""

import sys
from pathlib import Path
import yaml
import numpy as np
from tqdm import tqdm

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

from retrieval import BiEncoder
from flickr30k import Flickr30KDataset


def load_config(config_path: str = 'configs/clip_config.yaml') -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    """Main function to generate text embeddings."""
    print("=" * 60)
    print("FLICKR30K TEXT EMBEDDING GENERATION")
    print("=" * 60)
    
    # Load configuration
    config = load_config()
    print(f"\n✓ Configuration loaded")
    print(f"  Model: {config['model']['name']}")
    print(f"  Batch size: {config['encoding']['text_batch_size']}")
    
    # Initialize dataset
    print(f"\n📂 Loading dataset...")
    dataset = Flickr30KDataset(
        images_dir=config['dataset']['images_dir'],
        captions_file=config['dataset']['captions_file']
    )
    
    # Get all captions and filter out None/NaN values
    all_captions = dataset.df['caption'].tolist()
    print(f"✓ Found {len(all_captions):,} captions")
    
    # Filter out None, NaN, and empty captions
    original_count = len(all_captions)
    all_captions = [str(cap) for cap in all_captions if cap is not None and str(cap).strip() and str(cap) != 'nan']
    filtered_count = original_count - len(all_captions)
    
    if filtered_count > 0:
        print(f"⚠️  Filtered out {filtered_count} invalid captions")
    print(f"✓ Processing {len(all_captions):,} valid captions")
    
    # Initialize encoder
    print(f"\n🤖 Loading CLIP model...")
    encoder = BiEncoder(
        model_name=config['model']['name'],
        pretrained=config['model']['pretrained']
    )
    
    # Generate embeddings
    print(f"\n🔄 Generating embeddings...")
    
    embeddings = encoder.encode_texts(
        all_captions,
        batch_size=config['encoding']['text_batch_size'],
        normalize=config['encoding']['normalize'],
        show_progress=True
    )
    
    print(f"\n✓ Generated embeddings shape: {embeddings.shape}")
    
    # Create metadata
    print(f"\n💾 Saving embeddings...")
    metadata = {
        'num_captions': len(all_captions),
        'num_images': dataset.num_images,
        'model': config['model']['name'],
        'pretrained': config['model']['pretrained'],
        'embedding_dim': embeddings.shape[1],
        'normalized': config['encoding']['normalize'],
        'caption_to_image_mapping': dataset.df[['image_name', 'caption']].to_dict('records')
    }
    
    encoder.save_embeddings(
        embeddings,
        config['paths']['text_embeddings'],
        metadata=metadata
    )
    
    print(f"\n{'=' * 60}")
    print(f"✅ COMPLETED!")
    print(f"{'=' * 60}")
    print(f"Total captions processed: {len(all_captions):,}")
    print(f"Embedding dimension: {embeddings.shape[1]}")
    print(f"Output file: {config['paths']['text_embeddings']}")
    print(f"Metadata file: {config['paths']['text_metadata']}")


if __name__ == '__main__':
    main()
