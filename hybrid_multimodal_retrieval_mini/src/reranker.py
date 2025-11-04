"""BLIP-2 re-ranker for improving search results."""

import torch
from PIL import Image
from tqdm import tqdm
from transformers import Blip2Processor, Blip2ForConditionalGeneration


class BLIP2Reranker:
    """Use BLIP-2 to re-rank search results."""
    
    def __init__(self, model_name='Salesforce/blip2-opt-2.7b', device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Loading BLIP-2 on {self.device}...")
        
        self.processor = Blip2Processor.from_pretrained(model_name)
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32
        )
        self.model.to(self.device)
        self.model.eval()
        print("BLIP-2 loaded!")
    
    def score_pairs(self, texts, images, batch_size=4):
        """Score text-image pairs."""
        scores = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Re-ranking"):
            batch_texts = texts[i:i + batch_size]
            batch_images = images[i:i + batch_size]
            
            # Load images
            pil_images = []
            for img in batch_images:
                if isinstance(img, (str, Image.Image)):
                    if isinstance(img, str):
                        img = Image.open(img).convert('RGB')
                    pil_images.append(img)
            
            # Score each pair
            for text, image in zip(batch_texts, pil_images):
                score = self._score_single(text, image)
                scores.append(score)
        
        return scores
    
    def _score_single(self, text, image):
        """Score one text-image pair using yes/no probability."""
        try:
            # Ask "Does this image show {text}?"
            prompt = f"Question: Does this image show {text.lower()}? Answer:"
            
            inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=1,
                    output_scores=True,
                    return_dict_in_generate=True,
                    do_sample=False
                )
                
                logits = outputs.scores[0][0]
                
                # Get yes/no token IDs
                yes_id = self.processor.tokenizer.encode(" yes", add_special_tokens=False)[0]
                no_id = self.processor.tokenizer.encode(" no", add_special_tokens=False)[0]
                
                # Calculate probability
                probs = torch.nn.functional.softmax(logits, dim=-1)
                prob_yes = probs[yes_id].item()
                prob_no = probs[no_id].item()
                
                score = prob_yes / (prob_yes + prob_no) if (prob_yes + prob_no) > 0 else 0.5
                return score
                
        except Exception as e:
            print(f"Warning: Scoring failed: {e}")
            return 0.5
