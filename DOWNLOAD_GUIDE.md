# 📥 Simple Guide to Download Flickr30K Dataset

## Method 1: Direct Browser Download (EASIEST - NO KAGGLE API NEEDED!)

### Step 1: Download from Kaggle
1. **Go to**: https://www.kaggle.com/datasets/hsankesara/flickr-image-dataset
2. **Sign in** or create a free Kaggle account
3. **Click the "Download" button** (top right)
4. **Wait for download** (~2-5 GB, filename will be `archive.zip` or `flickr-image-dataset.zip`)
5. **Find the file** in your Downloads folder

### Step 2: Extract the Files
1. Right-click the downloaded `.zip` file
2. Select "Extract All..." or use 7-Zip/WinRAR
3. Extract to a temporary location

### Step 3: Organize Files (Choose one option)

#### Option A: Use the Python Script
```powershell
cd "d:\Giáo trình 20251\IT3930E - Project III\hybrid_multimodal_retrieval"
python download_flickr30k.py
```
- Select option 3 (organize extracted files)
- Provide the path where you extracted the files

#### Option B: Manual Organization
1. **Find the images**: Look for folder with ~31,000 `.jpg` files
2. **Copy all `.jpg` files** to:
   ```
   d:\Giáo trình 20251\IT3930E - Project III\hybrid_multimodal_retrieval\data\images\
   ```
3. **Find `results.csv`** (or `results_20130124.token`)
4. **Copy it** to:
   ```
   d:\Giáo trình 20251\IT3930E - Project III\hybrid_multimodal_retrieval\data\
   ```

---

## Method 2: Use Hugging Face Datasets (Alternative)

If Kaggle doesn't work, try Hugging Face:

### Install the library:
```powershell
pip install datasets
```

### Create a download script:
```python
# download_from_hf.py
from datasets import load_dataset

print("Downloading Flickr30K from Hugging Face...")
dataset = load_dataset("nlphuji/flickr30k", split="test")

print(f"Total examples: {len(dataset)}")
print("Dataset downloaded successfully!")
print("Note: You'll need to process this differently as it's in HF format")
```

---

## Method 3: Direct Links (If Available)

Some mirrors might have direct download links. Check:
- Academic torrent sites
- University mirrors
- Research group repositories

⚠️ **Note**: Always verify the data integrity and ensure you're downloading from legitimate sources.

---

## Verification

After organizing files, verify your setup:

```powershell
python download_flickr30k.py
```

Or check manually:
- `data/images/` should have ~31,000 `.jpg` files
- `data/results.csv` should exist and have ~158,000 rows

---

## Expected Directory Structure

```
hybrid_multimodal_retrieval/
├── data/
│   ├── images/
│   │   ├── 1000092795.jpg
│   │   ├── 10002456.jpg
│   │   ├── 1000268201.jpg
│   │   └── ... (~31,000 images)
│   └── results.csv
├── flickr30k_exploration.ipynb
└── download_flickr30k.py
```

---

## Troubleshooting

### Problem: "Download button is grayed out"
**Solution**: Make sure you're logged into Kaggle and have accepted the dataset terms.

### Problem: "Downloaded file is corrupt"
**Solution**: Try downloading again with a stable internet connection. Some browsers handle large files better (try Chrome or Firefox).

### Problem: "Can't find the images after extraction"
**Solution**: Look inside subfolders. The images might be in:
- `archive/flickr30k_images/flickr30k_images/`
- `archive/Images/`
- `flickr-image-dataset/flickr30k_images/`

### Problem: "results.csv is missing"
**Solution**: 
- Check for `results_20130124.token` instead
- The file might be in a subfolder called `results/` or `captions/`
- You can also download it separately from: https://github.com/BryanPlummer/flickr30k_entities

---

## Alternative: Sample Dataset for Testing

If you just want to test your code, you can start with a small sample:

1. Download just a few images manually from Flickr
2. Create a simple CSV with your own captions:
```csv
image_name,comment_number,comment
test1.jpg,0,A person walking in the park
test1.jpg,1,Someone enjoying a sunny day outdoors
```

This lets you develop and test your code while waiting for the full dataset to download.

---

## Next Steps

Once your data is organized:

1. **Verify the setup**:
   ```powershell
   python download_flickr30k.py
   ```

2. **Run the exploration notebook**:
   ```powershell
   jupyter notebook flickr30k_exploration.ipynb
   ```

3. **Start building your retrieval system!** 🚀
