# Running Phase 3 on Kaggle

Quick guide to run your BLIP-2 Cross-Encoder exploration on Kaggle.

---

## 📋 Prerequisites

- Kaggle account
- Flickr30K dataset already on Kaggle (you should have this from Phase 2)

---

## 🚀 Quick Setup

### Step 1: Create New Kaggle Notebook

1. Go to https://www.kaggle.com
2. Click **"Code"** → **"New Notebook"**
3. Choose **"Notebook"** (not Script)
4. Enable **GPU**: 
   - Settings (right sidebar) → Accelerator → **GPU T4 x2** (or P100)

### Step 2: Add Flickr30K Dataset

1. In the notebook, click **"+ Add Data"** (right sidebar)
2. Search for **"Flickr30K"** or your uploaded dataset
3. Add it to the notebook
4. Dataset will be at `/kaggle/input/flickr30k/` (or similar path)

### Step 3: Clone Your GitHub Repository

Add this cell at the top of your notebook:

```python
# Clone your GitHub repository
!git clone https://github.com/vinhhna/hybrid_multimodal_retrieval.git
%cd hybrid_multimodal_retrieval

# Verify the clone
!ls -la
```

### Step 4: Install Dependencies

```python
# Install Phase 3 dependencies
!pip install -q transformers accelerate open-clip-torch pyyaml

# Install project in development mode
!pip install -e .
```

---

## 📝 Kaggle Notebook Structure

Here's a simple notebook structure:

### Cell 1: Clone Repository & Install

```python
# Clone your GitHub repository
!git clone https://github.com/vinhhna/hybrid_multimodal_retrieval.git
%cd hybrid_multimodal_retrieval

# Install dependencies
!pip install -q transformers accelerate open-clip-torch pyyaml

# Install project in development mode
!pip install -e .

import sys
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm.auto import tqdm

print(f"✓ PyTorch: {torch.__version__}")
print(f"✓ CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
```

### Cell 2: Setup Paths

```python
# Adjust these paths based on your Kaggle dataset location
IMAGES_DIR = Path('/kaggle/input/flickr30k/flickr30k-images/flickr30k-images')
CAPTIONS_FILE = Path('/kaggle/input/flickr30k/results.csv')

# Verify paths
print(f"Images dir exists: {IMAGES_DIR.exists()}")
print(f"Captions file exists: {CAPTIONS_FILE.exists()}")

# Count images
if IMAGES_DIR.exists():
    num_images = len(list(IMAGES_DIR.glob('*.jpg')))
    print(f"Found {num_images} images")
```

### Cell 3: Import CrossEncoder

```python
# Import from your cloned repository
from src.retrieval.cross_encoder import CrossEncoder

print("✓ CrossEncoder imported successfully!")
```

### Cell 4: Load BLIP-2 Model

```python
from src.retrieval.cross_encoder import CrossEncoder

print("Loading BLIP-2 from Hugging Face...")
encoder = CrossEncoder(
    model_name='Salesforce/blip2-opt-2.7b',  # Hugging Face model
    use_fp16=True  # Important for GPU memory
)
print("✓ Model loaded!")
```

### Cell 5: Test Single Pair

```python
# Get a test image
test_images = list(IMAGES_DIR.glob('*.jpg'))[:5]
test_image = test_images[0]

# Display image
img = Image.open(test_image)
display(img)

# Test queries
queries = [
    "A dog playing in the park",
    "People at a beach",
    "A colorful outdoor scene"
]

print(f"Image: {test_image.name}\n")
for query in queries:
    score = encoder.score_pair(query, test_image)
    print(f"'{query}': {score:.4f}")
```

### Cell 6: Batch Scoring Test

```python
# Test batch processing
n_pairs = 10
test_images = list(IMAGES_DIR.glob('*.jpg'))[:n_pairs]
test_queries = ["A photograph"] * n_pairs

import time
start = time.time()

scores = encoder.score_pairs(
    test_queries,
    test_images,
    batch_size=4,  # Adjust based on GPU memory
    show_progress=True
)

elapsed = time.time() - start
print(f"\n✓ Scored {n_pairs} pairs in {elapsed:.2f}s")
print(f"Average: {elapsed/n_pairs*1000:.1f}ms per pair")
```

### Cell 7: Benchmark (100 pairs)

```python
# Simulate reranking top-100
n_benchmark = 100
benchmark_images = list(IMAGES_DIR.glob('*.jpg'))[:n_benchmark]
benchmark_queries = ["A photograph"] * n_benchmark

print(f"Benchmarking {n_benchmark} pairs...")

start = time.time()
scores = encoder.score_pairs(
    benchmark_queries,
    benchmark_images,
    batch_size=4,
    show_progress=True
)
elapsed = time.time() - start

print(f"\n✓ Results:")
print(f"  Total time: {elapsed:.2f}s")
print(f"  Per pair: {elapsed/n_benchmark*1000:.1f}ms")
print(f"  Target: < 2s for 100 pairs")
print(f"  Status: {'✓ PASS' if elapsed < 2 else '⚠ NEEDS OPTIMIZATION'}")
```

---

## ⚙️ Kaggle-Specific Tips

### Memory Management

Kaggle notebooks have **GPU memory limits** (~16GB for T4). If you get OOM errors:

```python
# Use smaller batch size
encoder.batch_size = 2  # or even 1

# Clear cache frequently
import gc
torch.cuda.empty_cache()
gc.collect()
```

### Save Your Results

```python
# Save scores to Kaggle output
import pandas as pd

results_df = pd.DataFrame({
    'image': [img.name for img in test_images],
    'query': test_queries,
    'score': scores
})

results_df.to_csv('blip2_scores.csv', index=False)
print("✓ Saved to /kaggle/working/blip2_scores.csv")
```

### Download Results

After running, go to:
- **Output** tab (right sidebar)
- Download `blip2_scores.csv` or other saved files

---

## 🔄 Update Your Code on Kaggle

If you make changes to your GitHub repo and want to update in Kaggle:

```python
%cd /kaggle/working
!rm -rf hybrid_multimodal_retrieval
!git clone https://github.com/vinhhna/hybrid_multimodal_retrieval.git
%cd hybrid_multimodal_retrieval
!pip install -e .
```

---

## 🎯 Minimal CrossEncoder Code (If Not Using Git Clone)

If you prefer to paste code directly without cloning, here's a minimal version:

```python
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm.auto import tqdm
from transformers import Blip2Processor, Blip2ForConditionalGeneration

class CrossEncoder:
    def __init__(self, model_name='Salesforce/blip2-opt-2.7b', use_fp16=True):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Loading {model_name}...")
        
        self.processor = Blip2Processor.from_pretrained(model_name)
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if use_fp16 and self.device.type == 'cuda' else torch.float32
        )
        self.model.to(self.device)
        self.model.eval()
        self.use_fp16 = use_fp16 and self.device.type == 'cuda'
        print("✓ Model loaded!")
    
    def score_pair(self, query, image_path):
        scores = self.score_pairs([query], [image_path], show_progress=False)
        return scores[0]
    
    def score_pairs(self, queries, image_paths, batch_size=4, show_progress=True):
        scores = []
        
        for i in tqdm(range(len(queries)), disable=not show_progress):
            query = queries[i]
            image_path = image_paths[i]
            
            # Load image
            image = Image.open(image_path).convert('RGB')
            
            # Format as yes/no question
            prompt = f"Question: Does this image show {query.lower()}? Answer:"
            
            # Process inputs
            inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(self.device)
            if self.use_fp16:
                inputs = {k: v.half() if v.dtype == torch.float32 else v for k, v in inputs.items()}
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=1,
                    output_scores=True,
                    return_dict_in_generate=True,
                    do_sample=False
                )
                
                # Get yes/no probabilities
                logits = outputs.scores[0][0]
                yes_token_id = self.processor.tokenizer.encode(" yes", add_special_tokens=False)[0]
                no_token_id = self.processor.tokenizer.encode(" no", add_special_tokens=False)[0]
                
                probs = torch.nn.functional.softmax(logits, dim=-1)
                prob_yes = probs[yes_token_id].item()
                prob_no = probs[no_token_id].item()
                
                score = prob_yes / (prob_yes + prob_no) if (prob_yes + prob_no) > 0 else 0.5
                scores.append(score)
            
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
        
        return np.array(scores)

# Use it
encoder = CrossEncoder()
```

---

## 📊 Expected Performance on Kaggle

| GPU | Batch Size | Time/100 pairs | Status |
|-----|------------|----------------|---------|
| T4 x2 | 4 | ~2-3s | ✓ Good |
| T4 x2 | 8 | ~1.5-2s | ✓ Better |
| P100 | 8 | ~1-1.5s | ✓ Best |

---

## ❗ Common Issues

### 1. "transformers not found"
```python
!pip install transformers accelerate
```

### 2. "CUDA out of memory"
```python
# Reduce batch size
encoder.score_pairs(queries, images, batch_size=2)  # or 1
```

### 3. "Dataset path not found"
```python
# Check your dataset path
!ls /kaggle/input/
!ls /kaggle/input/flickr30k/
```

### 4. Model download slow
First run will download ~5GB model from Hugging Face. Be patient (~5-10 minutes).

---

## ✅ Quick Checklist

- [ ] Created Kaggle notebook
- [ ] Enabled GPU (T4 or P100)
- [ ] Added Flickr30K dataset
- [ ] Cloned GitHub repository (`git clone https://github.com/vinhhna/hybrid_multimodal_retrieval.git`)
- [ ] Installed dependencies (`transformers`, `accelerate`)
- [ ] Installed project (`pip install -e .`)
- [ ] Adjusted dataset paths in Cell 2
- [ ] Tested single pair scoring
- [ ] Tested batch scoring
- [ ] Ran 100-pair benchmark
- [ ] Saved results

---

## 🎉 You're Done!

Once you complete the benchmark successfully on Kaggle, you've finished **Phase 3 Week 1**!

**Next**: Week 2 - Build the hybrid retrieval pipeline (bi-encoder + cross-encoder)

---

**Tips**:
- Save your notebook frequently (Ctrl+S)
- Kaggle sessions timeout after 12 hours
- Download important results before session ends
- You can commit your notebook to save it permanently
- Your GitHub repo URL: `https://github.com/vinhhna/hybrid_multimodal_retrieval`
- To update code: Delete the folder and re-clone from GitHub
