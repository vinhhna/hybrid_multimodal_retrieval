"""
Build FAISS indices from Flickr30K dataset.
Run this once to create the search indices.
"""

from src.dataset import Flickr30KDataset
from src.encoder import CLIPEncoder
from src.index import FAISSIndex


def main():
    # Load dataset
    dataset = Flickr30KDataset('data/images', 'data/results.csv')
    
    # Initialize encoder
    encoder = CLIPEncoder()
    
    # Generate image embeddings
    image_names = dataset.get_all_images()
    images = [dataset.get_image(name) for name in image_names]
    image_embeddings = encoder.encode_images(images, batch_size=32)
    
    # Build image index
    image_index = FAISSIndex(dimension=512)
    image_index.add(image_embeddings, ids=image_names)
    image_index.save('data/image_index.faiss')
    
    # Generate text embeddings
    captions = dataset.df['caption'].tolist()
    text_embeddings = encoder.encode_texts(captions, batch_size=64)
    
    # Build text index
    text_index = FAISSIndex(dimension=512)
    text_index.add(text_embeddings)
    text_index.save('data/text_index.faiss')


if __name__ == "__main__":
    main()
