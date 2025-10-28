"""
BLIP-2 Cross-Encoder for Multimodal Reranking

This module implements a cross-encoder using BLIP-2 for accurate reranking
of retrieval results. Unlike bi-encoders (CLIP) that encode queries and 
candidates separately, cross-encoders jointly process pairs for deeper
interaction and better accuracy.

Phase 3: Cross-Encoder Reranking
Created: October 28, 2025
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
    from lavis.models import load_model_and_preprocess
except ImportError:
    raise ImportError(
        "salesforce-lavis not found. Install with: pip install salesforce-lavis"
    )


class CrossEncoder:
    """
    BLIP-2 based cross-encoder for reranking query-candidate pairs.
    
    This class provides functionality to score image-text pairs with deep
    interaction, enabling accurate reranking of retrieval results.
    
    Attributes:
        model: BLIP-2 model instance
        vis_processors: Image preprocessing pipelines
        txt_processors: Text preprocessing pipelines
        device: torch.device for computation
        config: Configuration dictionary
        batch_size: Default batch size for processing
        use_fp16: Whether to use mixed precision
    
    Example:
        >>> encoder = CrossEncoder(model_name='blip2_opt')
        >>> score = encoder.score_pair("A dog playing", image_path)
        >>> scores = encoder.score_pairs(queries, images, batch_size=8)
    """
    
    def __init__(
        self,
        model_name: str = 'blip2_opt',
        model_type: str = 'pretrain_opt2.7b',
        config_path: Optional[Union[str, Path]] = None,
        device: Optional[Union[str, torch.device]] = None,
        use_fp16: bool = True
    ):
        """
        Initialize BLIP-2 cross-encoder.
        
        Args:
            model_name: BLIP-2 model name ('blip2_opt', 'blip2_t5')
            model_type: Model variant (e.g., 'pretrain_opt2.7b')
            config_path: Path to YAML config file
            device: Device to use ('cuda', 'cpu', or torch.device)
            use_fp16: Use mixed precision (FP16) for efficiency
        """
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing BLIP-2 Cross-Encoder...")
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Set device
        if device is None:
            device = self.config['model'].get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.device = torch.device(device)
        self.logger.info(f"Using device: {self.device}")
        
        # Load BLIP-2 model
        self.logger.info(f"Loading BLIP-2 model: {model_name} ({model_type})")
        self.model, self.vis_processors, self.txt_processors = load_model_and_preprocess(
            name=model_name,
            model_type=model_type,
            is_eval=True,
            device=self.device
        )
        
        # Model configuration
        self.use_fp16 = use_fp16 and self.device.type == 'cuda'
        if self.use_fp16:
            self.model = self.model.half()
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
        Score text-image pairs using BLIP-2.
        
        Args:
            texts: List of text queries
            images: List of images (paths or PIL Images)
        
        Returns:
            List of relevance scores
        """
        # Load and preprocess images
        processed_images = []
        for img in images:
            if isinstance(img, (str, Path)):
                img = Image.open(img).convert('RGB')
            elif not isinstance(img, Image.Image):
                raise TypeError(f"Expected PIL Image, str, or Path, got {type(img)}")
            
            # Preprocess image
            img_tensor = self.vis_processors["eval"](img).unsqueeze(0).to(self.device)
            if self.use_fp16:
                img_tensor = img_tensor.half()
            processed_images.append(img_tensor)
        
        # Stack images
        images_tensor = torch.cat(processed_images, dim=0)
        
        # Process texts
        processed_texts = [self.txt_processors["eval"](text) for text in texts]
        
        # Generate scores using BLIP-2
        with torch.no_grad():
            # Use BLIP-2's image-text matching capability
            # This uses the model's ITM (Image-Text Matching) head
            try:
                # For BLIP-2, we can use the model's forward method
                # Different BLIP-2 variants may have different APIs
                samples = {
                    "image": images_tensor,
                    "text_input": processed_texts
                }
                
                # Get image-text matching scores
                # This is a simplified version - actual implementation may vary
                outputs = self.model(samples, match_head="itm")
                scores = outputs[:, 1].cpu().numpy()  # Probability of match
                
            except Exception as e:
                # Fallback: use caption generation probability as score
                self.logger.warning(f"ITM scoring failed, using generation probability: {e}")
                scores = self._score_via_generation(images_tensor, processed_texts)
        
        return scores.tolist()
    
    def _score_image_text_batch(
        self,
        images: List[Union[str, Path, Image.Image]],
        texts: List[str]
    ) -> List[float]:
        """Score image-text pairs (swap of text-image)."""
        return self._score_text_image_batch(texts, images)
    
    def _score_via_generation(
        self,
        images_tensor: torch.Tensor,
        texts: List[str]
    ) -> np.ndarray:
        """
        Fallback scoring using caption generation probability.
        
        Computes the log-likelihood of each text query given the corresponding image
        by using BLIP-2's language model to score how likely the text is as a caption
        for the image. Higher log-likelihood indicates better match.
        
        Args:
            images_tensor: Preprocessed images tensor, shape (batch_size, C, H, W)
            texts: List of text queries
        
        Returns:
            Array of similarity scores normalized to [0, 1] range
        """
        scores = []
        
        for i, text in enumerate(texts):
            img_tensor = images_tensor[i:i+1]
            
            try:
                # For BLIP-2, we can compute the log-likelihood of the text given the image
                # by using the model in evaluation mode
                
                # Prepare text input (add prompt for better scoring)
                # BLIP-2 works better with question-answering format
                prompt = f"Question: Does this image show {text.lower()}? Answer:"
                
                # Tokenize the text
                text_input = self.txt_processors["eval"](prompt)
                
                # Get image features
                with torch.no_grad():
                    # Extract image features using the vision encoder
                    image_embeds = self.model.ln_vision(self.model.visual_encoder(img_tensor))
                    image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(self.device)
                    
                    # Get query tokens
                    query_tokens = self.model.query_tokens.expand(image_embeds.shape[0], -1, -1)
                    
                    # Get Q-Former output
                    query_output = self.model.Qformer.bert(
                        query_embeds=query_tokens,
                        encoder_hidden_states=image_embeds,
                        encoder_attention_mask=image_atts,
                        return_dict=True,
                    )
                    
                    # Project to language model space
                    inputs_llm = self.model.llm_proj(query_output.last_hidden_state)
                    atts_llm = torch.ones(inputs_llm.size()[:-1], dtype=torch.long).to(self.device)
                    
                    # Tokenize target text for scoring
                    # Use a simple yes/no answer for scoring
                    answer_yes = " yes"
                    answer_no = " no"
                    
                    # Get logits for both answers
                    llm_tokens_yes = self.model.llm_tokenizer(
                        answer_yes, 
                        return_tensors="pt"
                    ).to(self.device)
                    
                    llm_tokens_no = self.model.llm_tokenizer(
                        answer_no,
                        return_tensors="pt"
                    ).to(self.device)
                    
                    # Prepare prompt tokens
                    prompt_tokens = self.model.llm_tokenizer(
                        prompt,
                        return_tensors="pt"
                    ).to(self.device)
                    
                    # Concatenate image embeddings with prompt
                    inputs_embeds = self.model.llm_model.get_input_embeddings()(prompt_tokens.input_ids)
                    inputs_embeds = torch.cat([inputs_llm, inputs_embeds], dim=1)
                    attention_mask = torch.cat([atts_llm, prompt_tokens.attention_mask], dim=1)
                    
                    # Get model outputs
                    outputs = self.model.llm_model(
                        inputs_embeds=inputs_embeds,
                        attention_mask=attention_mask,
                        return_dict=True,
                    )
                    
                    # Get logits for next token prediction
                    logits = outputs.logits[:, -1, :]  # Last token logits
                    
                    # Get probabilities using softmax
                    probs = torch.nn.functional.softmax(logits, dim=-1)
                    
                    # Get token IDs for "yes" and "no"
                    yes_token_id = llm_tokens_yes.input_ids[0, 1]  # Skip BOS token
                    no_token_id = llm_tokens_no.input_ids[0, 1]
                    
                    # Get probabilities for yes and no
                    prob_yes = probs[0, yes_token_id].item()
                    prob_no = probs[0, no_token_id].item()
                    
                    # Score is the probability of "yes" normalized by both probabilities
                    # This gives a score in [0, 1] range
                    score = prob_yes / (prob_yes + prob_no) if (prob_yes + prob_no) > 0 else 0.5
                    
            except Exception as e:
                # If anything fails, fall back to a neutral score
                self.logger.warning(f"Scoring failed for text '{text[:50]}...': {e}")
                score = 0.5
            
            scores.append(score)
        
        return np.array(scores)
    
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
