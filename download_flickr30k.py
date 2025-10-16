"""
Script to download and organize Flickr30K dataset.

DOWNLOAD OPTIONS:
================

Option 1: Direct Download from Kaggle (No API needed)
------------------------------------------------------
1. Visit: https://www.kaggle.com/datasets/hsankesara/flickr-image-dataset
2. Click "Download" button (requires free Kaggle account)
3. Extract the downloaded zip file
4. Run this script to organize files

Option 2: Alternative Sources
------------------------------
- Hugging Face: https://huggingface.co/datasets/nlphuji/flickr30k
- Original source: http://shannon.cs.illinois.edu/DenotationGraph/
  (Requires form submission and approval)

Option 3: Manual Organization
------------------------------
If you already have the dataset:
- Place all .jpg files in: data/images/
- Place results.csv in: data/

This script will help you organize the files after download.
"""

import os
import zipfile
import tarfile
from pathlib import Path

def setup_directories():
    """Create necessary directories."""
    data_dir = Path("data")
    images_dir = data_dir / "images"
    
    data_dir.mkdir(exist_ok=True)
    images_dir.mkdir(exist_ok=True)
    
    print(f"✓ Created directory: {data_dir}")
    print(f"✓ Created directory: {images_dir}")
    return data_dir, images_dir

def download_with_browser():
    """
    Guide user to download Flickr30K via web browser.
    This is the simplest and most reliable method.
    """
    import webbrowser
    
    print("\n" + "=" * 70)
    print("OPENING KAGGLE DOWNLOAD PAGE IN YOUR BROWSER")
    print("=" * 70)
    print("\nSteps to follow:")
    print("1. Log in to Kaggle (create free account if needed)")
    print("2. Click the 'Download' button on the page")
    print("3. Wait for the zip file to download (~2-5 GB)")
    print("4. Come back here after download completes")
    print("=" * 70)
    
    input("\nPress ENTER to open Kaggle in your browser...")
    
    try:
        webbrowser.open('https://www.kaggle.com/datasets/hsankesara/flickr-image-dataset')
        print("✓ Browser opened!")
    except:
        print("⚠️  Could not open browser automatically.")
        print("Please manually visit:")
        print("https://www.kaggle.com/datasets/hsankesara/flickr-image-dataset")
    
    print("\n⏳ Waiting for you to download the dataset...")
    input("Press ENTER after you've downloaded the zip file...")
    
    return True

def organize_downloaded_files():
    """Organize files after extraction - search common locations."""
    data_dir = Path("data")
    images_dir = data_dir / "images"
    
    # Look for common extraction patterns in Downloads
    downloads = Path.home() / "Downloads"
    possible_source_dirs = [
        downloads / "flickr-image-dataset",
        downloads / "archive",
        downloads / "flickr30k_images",
        data_dir / "flickr30k_images" / "flickr30k_images",
        data_dir / "flickr30k_images",
        data_dir / "Images",
    ]
    
    found_images = False
    for source_dir in possible_source_dirs:
        if source_dir.exists() and source_dir.is_dir():
            # Check for images
            jpg_files = list(source_dir.glob("*.jpg"))
            if jpg_files:
                print(f"✓ Found {len(jpg_files)} images in: {source_dir}")
                organize_from_path(source_dir, images_dir, data_dir)
                found_images = True
                break
            
            # Check subdirectories
            for subdir in source_dir.iterdir():
                if subdir.is_dir():
                    jpg_files = list(subdir.glob("*.jpg"))
                    if jpg_files:
                        print(f"✓ Found {len(jpg_files)} images in: {subdir}")
                        organize_from_path(subdir, images_dir, data_dir)
                        found_images = True
                        break
            
            if found_images:
                break
    
    if not found_images:
        print("⚠️  Could not automatically find image files.")
        print("Please use option 3 to manually specify the extraction location.")


def organize_from_path(source_dir, target_images_dir, target_data_dir):
    """
    Organize files from a source directory to the proper locations.
    
    Args:
        source_dir: Path where extracted files are located
        target_images_dir: Path to data/images/
        target_data_dir: Path to data/
    """
    import shutil
    
    print(f"\nOrganizing files from: {source_dir}")
    print("=" * 70)
    
    moved_images = 0
    moved_captions = 0
    
    # Move image files
    for img_file in source_dir.glob("*.jpg"):
        target = target_images_dir / img_file.name
        if not target.exists():
            shutil.copy2(img_file, target)
            moved_images += 1
            if moved_images % 1000 == 0:
                print(f"  Copied {moved_images} images...")
    
    if moved_images > 0:
        print(f"✓ Copied {moved_images} images to {target_images_dir}")
    
    # Also check subdirectories for images
    for subdir in source_dir.iterdir():
        if subdir.is_dir() and subdir.name.lower() in ['images', 'flickr30k_images']:
            for img_file in subdir.glob("*.jpg"):
                target = target_images_dir / img_file.name
                if not target.exists():
                    shutil.copy2(img_file, target)
                    moved_images += 1
                    if moved_images % 1000 == 0:
                        print(f"  Copied {moved_images} images...")
    
    # Move caption files
    for caption_file in source_dir.glob("*.csv"):
        target = target_data_dir / caption_file.name
        if not target.exists():
            shutil.copy2(caption_file, target)
            print(f"✓ Copied captions file: {caption_file.name}")
            moved_captions += 1
    
    # Check for .token files too
    for caption_file in source_dir.glob("*.token"):
        target = target_data_dir / caption_file.name
        if not target.exists():
            shutil.copy2(caption_file, target)
            print(f"✓ Copied captions file: {caption_file.name}")
            moved_captions += 1
    
    print("=" * 70)
    
    if moved_images == 0 and moved_captions == 0:
        print("⚠️  No files were copied. They might already be in place.")
    
    return moved_images > 0 or moved_captions > 0

def extract_tar_gz(file_path, extract_to):
    """Extract .tar.gz files."""
    print(f"Extracting {file_path}...")
    with tarfile.open(file_path, 'r:gz') as tar:
        tar.extractall(extract_to)
    print(f"✓ Extracted to {extract_to}")

def extract_zip(file_path, extract_to):
    """Extract .zip files."""
    print(f"Extracting {file_path}...")
    with zipfile.ZipFile(file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"✓ Extracted to {extract_to}")

def verify_dataset():
    """Verify that the dataset is properly set up."""
    data_dir = Path("data")
    images_dir = data_dir / "images"
    
    # Check for images
    image_files = list(images_dir.glob("*.jpg"))
    print(f"\n📊 Dataset Verification:")
    print(f"  Images found: {len(image_files)}")
    
    # Check for captions
    caption_files = list(data_dir.glob("*.csv")) + list(data_dir.glob("*.token"))
    if caption_files:
        print(f"  Captions file: {caption_files[0].name}")
    else:
        print(f"  ⚠️  No captions file found")
    
    if len(image_files) >= 31000:
        print("\n✓ Dataset appears complete!")
    elif len(image_files) > 0:
        print(f"\n⚠️  Only {len(image_files)} images found (expected ~31,000)")
    else:
        print("\n❌ No images found. Please download the dataset manually.")

def main():
    print("=" * 70)
    print("           FLICKR30K DATASET SETUP TOOL")
    print("=" * 70)
    
    # Create directories
    data_dir, images_dir = setup_directories()
    
    print("\n" + "=" * 70)
    print("DOWNLOAD OPTIONS:")
    print("=" * 70)
    print("1. Download from Kaggle (via browser - RECOMMENDED)")
    print("2. I already have the dataset files")
    print("3. Help me organize extracted files")
    print()
    
    choice = input("Enter your choice (1, 2, or 3): ").strip()
    
    if choice == "1":
        # Browser-based download
        download_with_browser()
        
        print("\n" + "=" * 70)
        print("EXTRACT THE DOWNLOADED FILE")
        print("=" * 70)
        print("Please extract the zip file you just downloaded.")
        print(f"Look for: archive.zip or flickr-image-dataset.zip in your Downloads folder")
        print()
        downloads_folder = Path.home() / "Downloads"
        print(f"💡 Tip: Check your Downloads folder: {downloads_folder}")
        print()
        input("Press ENTER after extracting the zip file...")
        
        # Try to find and organize files
        print("\nLooking for extracted files...")
        organize_downloaded_files()
        
    elif choice == "2":
        print("\n" + "=" * 70)
        print("MANUAL FILE PLACEMENT")
        print("=" * 70)
        print(f"Please place your files in these locations:")
        print(f"  • All .jpg images → {images_dir.absolute()}")
        print(f"  • results.csv → {data_dir.absolute()}")
        print()
        input("Press ENTER after copying the files...")
        
    elif choice == "3":
        print("\n" + "=" * 70)
        print("FILE ORGANIZATION HELPER")
        print("=" * 70)
        print("I'll help you move files from your Downloads or extraction folder.")
        print()
        source_path = input("Enter the path where your extracted files are: ").strip().strip('"')
        
        if source_path and Path(source_path).exists():
            organize_from_path(Path(source_path), images_dir, data_dir)
        else:
            print("❌ Path not found. Please check and try again.")
    else:
        print("\n❌ Invalid choice. Please run the script again.")
        return
    
    # Verify
    print("\n")
    verify_dataset()
    
    print("\n" + "=" * 70)
    print("NEXT STEPS:")
    print("=" * 70)
    print("✓ Run the exploration notebook: flickr30k_exploration.ipynb")
    print("  Command: jupyter notebook flickr30k_exploration.ipynb")
    print("=" * 70)

if __name__ == "__main__":
    main()
