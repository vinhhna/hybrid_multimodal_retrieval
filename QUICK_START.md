# 🚀 Quick Start Guide

## Installation

### 1. Navigate to Project Directory
```bash
cd "d:\Giáo trình 20251\IT3930E - Project III\hybrid_multimodal_retrieval"
```

### 2. Install the Package
```bash
# Install in development mode
pip install -e .

# Or with development dependencies
pip install -e ".[dev]"
```

### 3. Verify Installation
```bash
python -c "from flickr30k import Flickr30KDataset; print('✓ Installation successful!')"
```

---

## Usage Examples

### In Python Scripts
```python
# Import the package
from flickr30k import Flickr30KDataset
from flickr30k.visualization import display_random_samples

# Load dataset
dataset = Flickr30KDataset(
    images_dir='data/images',
    captions_file='data/results.csv'
)

# Get statistics
stats = dataset.get_statistics()
print(f"Dataset: {stats['num_images']:,} images, {stats['num_captions']:,} captions")

# Search captions
results = dataset.search_captions('dog', max_results=5)
print(f"Found {len(results)} captions containing 'dog'")

# Display random samples
display_random_samples(dataset, n_samples=3)
```

### In Jupyter Notebooks
```python
# Add to first cell if package not installed
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent / 'src'))

# Import and use
from flickr30k import Flickr30KDataset
from flickr30k.visualization import display_image_with_captions

dataset = Flickr30KDataset()
display_image_with_captions('1000092795.jpg', dataset=dataset)
```

---

## Running the Notebooks

### Option 1: Use the Clean Refactored Notebook
```bash
jupyter notebook notebooks/flickr30k_exploration_clean.ipynb
```

This notebook:
- ✅ Uses the modular package
- ✅ Much cleaner and shorter
- ✅ Easy to understand
- ✅ Focused on exploration

### Option 2: Use the Original Notebook
```bash
jupyter notebook notebooks/flickr30k_exploration.ipynb
```

This notebook:
- ✅ Original version with all code inline
- ✅ Still works as before
- ⚠️  Longer and more complex

---

## Common Tasks

### Check Dataset Status
```python
from flickr30k.utils import print_data_status
print_data_status()
```

### Load Configuration
```python
from flickr30k.utils import load_config
config = load_config()
print(config['data'])
```

### Get Random Sample
```python
from flickr30k import Flickr30KDataset

dataset = Flickr30KDataset()
image_name, captions = dataset.get_random_sample(seed=42)

print(f"Image: {image_name}")
for i, caption in enumerate(captions, 1):
    print(f"{i}. {caption}")
```

### Search and Display
```python
from flickr30k import Flickr30KDataset
from flickr30k.visualization import display_search_results

dataset = Flickr30KDataset()
results = dataset.search_captions('children playing', max_results=10)

display_search_results(
    results_df=results,
    keyword='children playing',
    max_display=5,
    show_images=False
)
```

---

## Project Structure Overview

```
hybrid_multimodal_retrieval/
│
├── 📄 README.md              → Full documentation
├── 📄 requirements.txt       → Dependencies
├── 📄 setup.py              → Package installation
├── 📄 RESTRUCTURING_SUMMARY.md → This restructuring guide
│
├── 📁 src/flickr30k/        → Main package (importable)
│   ├── __init__.py
│   ├── dataset.py           → Dataset class
│   ├── utils.py             → Utilities
│   └── visualization.py     → Plotting functions
│
├── 📁 configs/              → Configuration files
│   └── default.yaml
│
├── 📁 notebooks/            → Jupyter notebooks
│   ├── flickr30k_exploration.ipynb        (original)
│   └── flickr30k_exploration_clean.ipynb  (recommended)
│
├── 📁 scripts/              → Utility scripts
│   └── download_flickr30k.py
│
├── 📁 data/                 → Dataset (not in git)
│   ├── images/
│   └── results.csv
│
└── 📁 tests/                → Unit tests (future)
```

---

## Troubleshooting

### Import Error: "No module named 'flickr30k'"
**Solution**: Install the package
```bash
pip install -e .
```

### Import Error in Notebook
**Solution**: Add this to first cell
```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent / 'src'))
```

### Dataset Not Found
**Solution**: Download dataset
```bash
python scripts/download_flickr30k.py
```

### YAML Error: "No module named 'yaml'"
**Solution**: Install pyyaml
```bash
pip install pyyaml
```

---

## Next Steps

1. ✅ **Explore**: Run `notebooks/flickr30k_exploration_clean.ipynb`
2. 🎯 **Feature Extraction**: Add visual/text feature extraction
3. 🔍 **Retrieval**: Implement cross-modal search
4. 📊 **Evaluation**: Add retrieval metrics
5. 🎓 **Training**: Fine-tune models for better alignment

---

## Getting Help

- 📖 Read `README.md` for full documentation
- 📋 Check `RESTRUCTURING_SUMMARY.md` for detailed changes
- 💬 Open an issue on the repository
- 📧 Contact project maintainer

---

**Happy coding! 🎉**
