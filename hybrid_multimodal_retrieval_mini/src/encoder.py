"""CLIP encoder for image and text embeddings."""

import torch
import numpy as np
from PIL import Image
import open_clip
from tqdm import tqdm


class CLIPEncoder:
    """Encode images and text using CLIP."""
    
    def __init__(self, model_name='ViT-B-32', device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Loading CLIP model on {self.device}...")
        
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained='openai'
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model = self.model.to(self.device)
        self.model.eval()
        print("Model loaded!")
    
    def encode_images(self, images, batch_size=32):
        """Encode images to embeddings."""
        # Load images if they are paths
        if isinstance(images[0], str):
            images = [Image.open(img).convert('RGB') for img in images]
        
        embeddings = []
        with torch.no_grad():
            for i in tqdm(range(0, len(images), batch_size), desc="Encoding images"):
                batch = images[i:i + batch_size]
                batch_tensors = torch.stack([self.preprocess(img) for img in batch])
                batch_tensors = batch_tensors.to(self.device)
                
                batch_emb = self.model.encode_image(batch_tensors)
                batch_emb = torch.nn.functional.normalize(batch_emb, dim=-1)
                embeddings.append(batch_emb.cpu().numpy())
        
        return np.vstack(embeddings)
    
    def encode_texts(self, texts, batch_size=64):
        """Encode texts to embeddings."""
        embeddings = []
        with torch.no_grad():
            for i in tqdm(range(0, len(texts), batch_size), desc="Encoding texts"):
                batch = texts[i:i + batch_size]
                batch_tokens = self.tokenizer(batch).to(self.device)
                
                batch_emb = self.model.encode_text(batch_tokens)
                batch_emb = torch.nn.functional.normalize(batch_emb, dim=-1)
                embeddings.append(batch_emb.cpu().numpy())
        
        return np.vstack(embeddings)
