"""Multimodal search engine."""

from PIL import Image


class SearchEngine:
    """Simple multimodal search engine."""
    
    def __init__(self, encoder, image_index, text_index, dataset):
        self.encoder = encoder
        self.image_index = image_index
        self.text_index = text_index
        self.dataset = dataset
    
    def text_to_image(self, query_text, k=10):
        """Search images using text."""
        # Encode query
        query_emb = self.encoder.encode_texts([query_text])
        
        # Search
        scores, indices = self.image_index.search(query_emb, k=k)
        
        # Get image names
        results = []
        for idx, score in zip(indices[0], scores[0]):
            image_name = self.image_index.ids[idx]
            results.append((image_name, float(score)))
        
        return results
    
    def image_to_text(self, image_path, k=10):
        """Search captions using image."""
        # Load and encode image
        img = Image.open(image_path).convert('RGB')
        query_emb = self.encoder.encode_images([img])
        
        # Search
        scores, indices = self.text_index.search(query_emb, k=k)
        
        # Get captions
        results = []
        for idx, score in zip(indices[0], scores[0]):
            caption = self.dataset.df.iloc[idx]['caption']
            results.append((caption, float(score)))
        
        return results
    
    def image_to_image(self, image_path, k=10):
        """Search similar images."""
        # Load and encode image
        img = Image.open(image_path).convert('RGB')
        query_emb = self.encoder.encode_images([img])
        
        # Search
        scores, indices = self.image_index.search(query_emb, k=k)
        
        # Get image names
        results = []
        for idx, score in zip(indices[0], scores[0]):
            image_name = self.image_index.ids[idx]
            results.append((image_name, float(score)))
        
        return results
