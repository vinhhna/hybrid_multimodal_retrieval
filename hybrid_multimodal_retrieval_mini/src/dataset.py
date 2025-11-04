"""Dataset handler for Flickr30K."""

import pandas as pd
from pathlib import Path
from PIL import Image


class Flickr30KDataset:
    """Load and access Flickr30K dataset."""
    
    def __init__(self, images_dir="data/images", captions_file="data/results.csv"):
        self.images_dir = Path(images_dir)
        self.captions_file = Path(captions_file)
        self.df = None
        
        if self.captions_file.exists():
            self.load_captions()
    
    def load_captions(self):
        """Load captions from CSV."""
        self.df = pd.read_csv(self.captions_file, sep='|', engine='python')
        self.df.columns = self.df.columns.str.strip()
        self.df = self.df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
        self.df.columns = ['image_name', 'comment_number', 'caption']
        print(f"Loaded {len(self.df)} captions for {self.df['image_name'].nunique()} images")
    
    def get_captions(self, image_name):
        """Get captions for an image."""
        return self.df[self.df['image_name'] == image_name]['caption'].tolist()
    
    def get_image(self, image_name):
        """Load an image."""
        image_path = self.images_dir / image_name
        if image_path.exists():
            return Image.open(image_path)
        return None
    
    def get_all_images(self):
        """Get list of all image names."""
        return self.df['image_name'].unique().tolist()
