"""
Bi-Encoder module using CLIP for multimodal embeddings.

This module provides the BiEncoder class that uses CLIP (ViT-B/32) to generate
embeddings for both images and text, enabling cross-modal retrieval.
"""

import torch
import numpy as np
from pathlib import Path
from typing import List, Union, Optional
from PIL import Image
import open_clip
from tqdm import tqdm
import json


class BiEncoder:
    """
    Bi-Encoder using CLIP for generating image and text embeddings.
    
    This class wraps OpenCLIP models to provide a unified interface for:
    - Encoding images into dense vectors
    - Encoding text into dense vectors
    - Saving and loading embeddings
    
    Attributes:
        model: The CLIP model
        preprocess: Image preprocessing function
        tokenizer: Text tokenizer
        device: Device to run the model on (cuda/cpu)
        model_name: Name of the CLIP model variant
    """
    
    def __init__(
        self,
        model_name: str = 'ViT-B-32',
        pretrained: str = 'openai',
        device: Optional[str] = None
    ):
        """
        Initialize the BiEncoder with a CLIP model.
        
        Args:
            model_name: CLIP model variant (default: 'ViT-B-32')
            pretrained: Pretrained weights to use (default: 'openai')
            device: Device to use ('cuda' or 'cpu'). Auto-detects if None.
        """
        # Canonicalize model name: convert '/' to '-' (e.g., 'ViT-B/32' → 'ViT-B-32')
        model_name = model_name.replace('/', '-')
        self.model_name = model_name
        
        # Preserve explicit device strings (e.g., 'cuda:1')
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # Keep device string verbatim - do NOT coerce 'cuda:1' -> 'cuda'
        self.device = device
        
        print(f"Loading CLIP model: {model_name} ({pretrained})")
        print(f"Using device: {self.device}")
        
        # Load CLIP model
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, 
            pretrained=pretrained
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)
        
        # Move model to device and set to eval mode
        self.model = self.model.to(self.device)
        self.model.eval()
        
        print(f"✓ CLIP model loaded successfully")
    
    def encode_images(
        self,
        images: Union[List[Image.Image], List[str], List[Path]],
        batch_size: int = 32,
        normalize: bool = True,
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Encode a list of images into embeddings.
        
        Args:
            images: List of PIL Images or paths to image files
            batch_size: Number of images to process at once
            normalize: Whether to L2-normalize the embeddings
            show_progress: Whether to show progress bar
            
        Returns:
            numpy array of shape (N, embedding_dim) containing image embeddings
        """
        # Load images if paths are provided
        if isinstance(images[0], (str, Path)):
            images = [Image.open(img).convert('RGB') for img in images]
        
        embeddings = []
        num_batches = (len(images) + batch_size - 1) // batch_size
        
        iterator = range(num_batches)
        if show_progress:
            iterator = tqdm(iterator, desc="Encoding images")
        
        with torch.no_grad():
            for i in iterator:
                batch = images[i * batch_size:(i + 1) * batch_size]
                
                # Preprocess images
                batch_tensors = torch.stack([self.preprocess(img) for img in batch])
                batch_tensors = batch_tensors.to(self.device)
                
                # Generate embeddings
                batch_embeddings = self.model.encode_image(batch_tensors)
                
                # Normalize if requested
                if normalize:
                    batch_embeddings = torch.nn.functional.normalize(batch_embeddings, dim=-1)
                
                embeddings.append(batch_embeddings.cpu().numpy())
        
        return np.vstack(embeddings)
    
    def encode_texts(
        self,
        texts: List[str],
        batch_size: int = 64,
        normalize: bool = True,
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Encode a list of texts into embeddings.
        
        Args:
            texts: List of text strings
            batch_size: Number of texts to process at once
            normalize: Whether to L2-normalize the embeddings
            show_progress: Whether to show progress bar
            
        Returns:
            numpy array of shape (N, embedding_dim) containing text embeddings
        """
        embeddings = []
        num_batches = (len(texts) + batch_size - 1) // batch_size
        
        iterator = range(num_batches)
        if show_progress:
            iterator = tqdm(iterator, desc="Encoding texts")
        
        with torch.no_grad():
            for i in iterator:
                batch = texts[i * batch_size:(i + 1) * batch_size]
                
                # Tokenize texts
                batch_tokens = self.tokenizer(batch)
                batch_tokens = batch_tokens.to(self.device)
                
                # Generate embeddings
                batch_embeddings = self.model.encode_text(batch_tokens)
                
                # Normalize if requested
                if normalize:
                    batch_embeddings = torch.nn.functional.normalize(batch_embeddings, dim=-1)
                
                embeddings.append(batch_embeddings.cpu().numpy())
        
        return np.vstack(embeddings)
    
    def save_embeddings(
        self,
        embeddings: np.ndarray,
        save_path: Union[str, Path],
        metadata: Optional[dict] = None
    ):
        """
        Save embeddings to disk in .npy format.
        
        Args:
            embeddings: Numpy array of embeddings
            save_path: Path to save the embeddings
            metadata: Optional metadata to save alongside embeddings
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save embeddings
        np.save(save_path, embeddings)
        print(f"✓ Saved embeddings to {save_path}")
        
        # Save metadata if provided
        if metadata is not None:
            metadata_path = save_path.with_suffix('.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            print(f"✓ Saved metadata to {metadata_path}")
    
    def load_embeddings(
        self,
        load_path: Union[str, Path]
    ) -> tuple[np.ndarray, Optional[dict]]:
        """
        Load embeddings from disk.
        
        Args:
            load_path: Path to the saved embeddings
            
        Returns:
            Tuple of (embeddings array, metadata dict or None)
        """
        load_path = Path(load_path)
        
        # Load embeddings
        embeddings = np.load(load_path)
        print(f"✓ Loaded embeddings from {load_path}")
        print(f"  Shape: {embeddings.shape}")
        
        # Load metadata if exists
        metadata_path = load_path.with_suffix('.json')
        metadata = None
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            print(f"✓ Loaded metadata from {metadata_path}")
        
        return embeddings, metadata
    
    def get_embedding_dim(self) -> int:
        """
        Get the dimensionality of embeddings produced by this model.
        
        Returns:
            Embedding dimension
        """
        # Use the visual projection output dim
        return self.model.visual.output_dim
    
    def __repr__(self) -> str:
        """String representation of the BiEncoder."""
        return f"BiEncoder(model='{self.model_name}', device='{self.device}')"
