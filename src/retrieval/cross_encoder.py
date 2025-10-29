"""
BLIP-2 Cross-Encoder for Multimodal Reranking

This module implements a cross-encoder using BLIP-2 for accurate reranking
of retrieval results. Unlike bi-encoders (CLIP) that encode queries and 
candidates separately, cross-encoders jointly process pairs for deeper
interaction and better accuracy.

Phase 3: Cross-Encoder Reranking
Created: October 28, 2025
Updated: October 28, 2025 (Switched to Hugging Face transformers)
"""

import torch
import numpy as np
from pathlib import Path
from typing import Union, List, Tuple, Optional, Dict, Any
from PIL import Image
from tqdm import tqdm
import yaml
import logging

try:
    from transformers import Blip2Processor, Blip2ForConditionalGeneration
except ImportError:
    raise ImportError(
        "transformers not found. Install with: pip install transformers"
    )


class CrossEncoder:
    """
    BLIP-2 based cross-encoder for reranking query-candidate pairs.
    
    This class provides functionality to score image-text pairs with deep
    interaction, enabling accurate reranking of retrieval results.
    
    Uses Hugging Face transformers implementation of BLIP-2.
    
    Attributes:
        model: BLIP-2 model instance (Hugging Face)
        processor: BLIP-2 processor for image/text preprocessing
        device: torch.device for computation
        config: Configuration dictionary
        batch_size: Default batch size for processing
        use_fp16: Whether to use mixed precision
    
    Example:
        >>> encoder = CrossEncoder()
        >>> score = encoder.score_pair("A dog playing", image_path)
        >>> scores = encoder.score_pairs(queries, images, batch_size=8)
    """
    
    def __init__(
        self,
        model_name: str = 'Salesforce/blip2-flan-t5-xl',
        config_path: Optional[Union[str, Path]] = None,
        device: Optional[Union[str, torch.device]] = None,
        use_fp16: bool = True
    ):
        """
        Initialize BLIP-2 cross-encoder.
        
        Args:
            model_name: Hugging Face model ID (default: 'Salesforce/blip2-flan-t5-xl')
                       Using BLIP-2 FlanT5-XL model (~15GB, fits P100 16GB GPU)
            config_path: Path to YAML config file
            device: Device to use ('cuda', 'cpu', or torch.device)
            use_fp16: Use mixed precision (FP16) for efficiency
        """
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing BLIP-2 Cross-Encoder (Hugging Face)...")
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Set device
        if device is None:
            device = self.config['model'].get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.device = torch.device(device)
        self.logger.info(f"Using device: {self.device}")
        
        # Load BLIP-2 model from Hugging Face
        self.logger.info(f"Loading BLIP-2 model from Hugging Face: {model_name}")
        self.processor = Blip2Processor.from_pretrained(model_name)
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if use_fp16 and self.device.type == 'cuda' else torch.float32
        )
        self.model.to(self.device)
        
        # Model configuration
        self.use_fp16 = use_fp16 and self.device.type == 'cuda'
        if self.use_fp16:
            self.logger.info("Using FP16 mixed precision")
        
        self.model.eval()
        
        # Processing configuration
        self.batch_size = self.config['scoring'].get('batch_size', 8)
        self.max_text_length = self.config['scoring'].get('max_text_length', 77)
        
        # Memory management
        self.max_batch_size = self.config['memory'].get('max_batch_size', 16)
        self.fallback_batch_size = self.config['memory'].get('fallback_batch_size', 4)
        self.clear_cache = self.config['memory'].get('clear_cache_after_batch', True)
        
        self.logger.info("✓ BLIP-2 Cross-Encoder initialized successfully")
    
    def _load_config(self, config_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if config_path is None:
            # Use default config path
            config_path = Path(__file__).parent.parent.parent / 'configs' / 'blip2_config.yaml'
        
        config_path = Path(config_path)
        
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            self.logger.info(f"Loaded config from {config_path}")
        else:
            # Default configuration
            config = {
                'model': {'device': 'cuda'},
                'scoring': {'batch_size': 8, 'max_text_length': 77},
                'memory': {
                    'max_batch_size': 16,
                    'fallback_batch_size': 4,
                    'clear_cache_after_batch': True
                },
                'optimization': {'use_fp16': True}
            }
            self.logger.warning(f"Config not found at {config_path}, using defaults")
        
        return config
    
    def score_pair(
        self,
        query: Union[str, Image.Image, Path],
        candidate: Union[str, Image.Image, Path],
        query_type: str = 'text',
        candidate_type: str = 'image'
    ) -> float:
        """
        Score a single query-candidate pair.
        
        Args:
            query: Query (text string, PIL Image, or path)
            candidate: Candidate (text string, PIL Image, or path)
            query_type: Type of query ('text' or 'image')
            candidate_type: Type of candidate ('text' or 'image')
        
        Returns:
            Relevance score (higher = more relevant)
        
        Example:
            >>> score = encoder.score_pair("A dog", "dog.jpg")
            >>> print(f"Score: {score:.4f}")
        """
        scores = self.score_pairs(
            [query], [candidate],
            query_type=query_type,
            candidate_type=candidate_type,
            show_progress=False
        )
        return scores[0]
    
    def score_pairs(
        self,
        queries: List[Union[str, Image.Image, Path]],
        candidates: List[Union[str, Image.Image, Path]],
        query_type: str = 'text',
        candidate_type: str = 'image',
        batch_size: Optional[int] = None,
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Score multiple query-candidate pairs with batch processing.
        
        Args:
            queries: List of queries (texts or images)
            candidates: List of candidates (texts or images)
            query_type: Type of queries ('text' or 'image')
            candidate_type: Type of candidates ('text' or 'image')
            batch_size: Batch size for processing (None = use default)
            show_progress: Show progress bar
        
        Returns:
            Array of scores, shape (n_pairs,)
        
        Example:
            >>> queries = ["dog", "cat", "bird"]
            >>> images = ["img1.jpg", "img2.jpg", "img3.jpg"]
            >>> scores = encoder.score_pairs(queries, images, batch_size=8)
        """
        if len(queries) != len(candidates):
            raise ValueError(f"Queries ({len(queries)}) and candidates ({len(candidates)}) must have same length")
        
        if batch_size is None:
            batch_size = self.batch_size
        
        n_pairs = len(queries)
        all_scores = []
        
        # Process in batches
        pbar = tqdm(total=n_pairs, desc="Scoring pairs", disable=not show_progress)
        
        for i in range(0, n_pairs, batch_size):
            batch_queries = queries[i:i + batch_size]
            batch_candidates = candidates[i:i + batch_size]
            
            try:
                # Process batch
                batch_scores = self._process_batch(
                    batch_queries, batch_candidates,
                    query_type, candidate_type
                )
                all_scores.extend(batch_scores)
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    self.logger.warning(f"OOM at batch size {batch_size}, using fallback")
                    # Handle OOM
                    batch_scores = self._handle_oom(
                        batch_queries, batch_candidates,
                        query_type, candidate_type,
                        batch_size
                    )
                    all_scores.extend(batch_scores)
                else:
                    raise e
            
            pbar.update(len(batch_queries))
            
            # Clear CUDA cache if enabled
            if self.clear_cache and self.device.type == 'cuda':
                torch.cuda.empty_cache()
        
        pbar.close()
        
        return np.array(all_scores)
    
    def _process_batch(
        self,
        batch_queries: List[Union[str, Image.Image, Path]],
        batch_candidates: List[Union[str, Image.Image, Path]],
        query_type: str,
        candidate_type: str
    ) -> List[float]:
        """
        Process a single batch of query-candidate pairs.
        
        Args:
            batch_queries: Batch of queries
            batch_candidates: Batch of candidates
            query_type: Type of queries
            candidate_type: Type of candidates
        
        Returns:
            List of scores for the batch
        """
        # Currently supporting text-to-image (most common case)
        if query_type == 'text' and candidate_type == 'image':
            return self._score_text_image_batch(batch_queries, batch_candidates)
        elif query_type == 'image' and candidate_type == 'text':
            return self._score_image_text_batch(batch_queries, batch_candidates)
        else:
            raise NotImplementedError(f"Scoring {query_type}-to-{candidate_type} not yet implemented")
    
    def _score_text_image_batch(
        self,
        texts: List[str],
        images: List[Union[str, Path, Image.Image]]
    ) -> List[float]:
        """
        Score text-image pairs using BLIP-2 (Hugging Face).
        
        Uses question-answering format to get interpretable similarity scores
        based on yes/no probability from BLIP-2's language model.
        
        Args:
            texts: List of text queries
            images: List of images (paths or PIL Images)
        
        Returns:
            List of relevance scores in [0, 1] range
        """
        # Load images
        pil_images = []
        for img in images:
            if isinstance(img, (str, Path)):
                img = Image.open(img).convert('RGB')
            elif not isinstance(img, Image.Image):
                raise TypeError(f"Expected PIL Image, str, or Path, got {type(img)}")
            pil_images.append(img)
        
        # Score each pair individually for better control
        scores = []
        for text, image in zip(texts, pil_images):
            score = self._score_single_pair(text, image)
            scores.append(score)
        
        return scores
    
    def _score_image_text_batch(
        self,
        images: List[Union[str, Path, Image.Image]],
        texts: List[str]
    ) -> List[float]:
        """Score image-text pairs (swap of text-image)."""
        return self._score_text_image_batch(texts, images)
    
    def _score_single_pair(
        self,
        text: str,
        image: Image.Image
    ) -> float:
        """
        Score a single text-image pair using yes/no probability.
        
        Uses BLIP-2's question-answering capability to ask:
        "Question: Does this image show {text}? Answer:"
        and computes P(yes) / (P(yes) + P(no)) as the similarity score.
        
        Args:
            text: Text query
            image: PIL Image
        
        Returns:
            Similarity score in [0, 1] range
        """
        try:
            # Format as yes/no question
            prompt = f"Question: Does this image show {text.lower()}? Answer:"
            
            # Process inputs
            inputs = self.processor(
                images=image,
                text=prompt,
                return_tensors="pt"
            ).to(self.device)
            
            if self.use_fp16:
                inputs = {k: v.half() if v.dtype == torch.float32 else v for k, v in inputs.items()}
            
            # Generate with forced yes/no answer
            with torch.no_grad():
                # Get model outputs without generation
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=1,
                    output_scores=True,
                    return_dict_in_generate=True,
                    do_sample=False
                )
                
                # Get the logits for the first generated token
                logits = outputs.scores[0][0]  # Shape: (vocab_size,)
                
                # Get token IDs for "yes" and "no"
                yes_token_id = self.processor.tokenizer.encode(" yes", add_special_tokens=False)[0]
                no_token_id = self.processor.tokenizer.encode(" no", add_special_tokens=False)[0]
                
                # Get probabilities
                probs = torch.nn.functional.softmax(logits, dim=-1)
                prob_yes = probs[yes_token_id].item()
                prob_no = probs[no_token_id].item()
                
                # Normalize to get score in [0, 1]
                total = prob_yes + prob_no
                score = prob_yes / total if total > 0 else 0.5
                
                return score
                
        except Exception as e:
            self.logger.warning(f"Scoring failed for text '{text[:50]}...': {e}")
            # Fallback: use neutral score
            return 0.5
    
    def _handle_oom(
        self,
        batch_queries: List,
        batch_candidates: List,
        query_type: str,
        candidate_type: str,
        current_batch_size: int
    ) -> List[float]:
        """
        Handle out-of-memory errors by reducing batch size.
        
        Args:
            batch_queries: Queries that caused OOM
            batch_candidates: Candidates that caused OOM
            query_type: Type of queries
            candidate_type: Type of candidates
            current_batch_size: Batch size that caused OOM
        
        Returns:
            Scores computed with smaller batch size
        """
        # Clear CUDA cache
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        
        # Use smaller batch size
        new_batch_size = max(self.fallback_batch_size, current_batch_size // 2)
        self.logger.info(f"Retrying with batch size {new_batch_size}")
        
        all_scores = []
        for i in range(0, len(batch_queries), new_batch_size):
            sub_queries = batch_queries[i:i + new_batch_size]
            sub_candidates = batch_candidates[i:i + new_batch_size]
            
            scores = self._process_batch(sub_queries, sub_candidates, query_type, candidate_type)
            all_scores.extend(scores)
        
        return all_scores
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary with model information
        """
        return {
            'device': str(self.device),
            'use_fp16': self.use_fp16,
            'batch_size': self.batch_size,
            'max_batch_size': self.max_batch_size,
            'model_type': type(self.model).__name__,
            'memory_allocated': torch.cuda.memory_allocated(self.device) if self.device.type == 'cuda' else 0,
            'memory_reserved': torch.cuda.memory_reserved(self.device) if self.device.type == 'cuda' else 0
        }
    
    def __repr__(self) -> str:
        return f"CrossEncoder(device={self.device}, fp16={self.use_fp16}, batch_size={self.batch_size})"


if __name__ == "__main__":
    # Simple test
    logging.basicConfig(level=logging.INFO)
    
    print("Testing BLIP-2 Cross-Encoder...")
    encoder = CrossEncoder()
    print(encoder.get_model_info())
    print("✓ Cross-Encoder initialized successfully")
