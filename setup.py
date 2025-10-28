"""
Setup script for the flickr30k package.

Install with: pip install -e .
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file with error handling
readme_file = Path(__file__).parent / "README.md"
try:
    long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""
except (UnicodeDecodeError, OSError):
    # Fallback if README can't be read (e.g., encoding issues, permission errors)
    long_description = "A package for hybrid multimodal retrieval using Flickr30K dataset"

setup(
    name="flickr30k",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A package for hybrid multimodal retrieval using Flickr30K dataset",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/hybrid_multimodal_retrieval",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "matplotlib>=3.7.0",
        "Pillow>=10.0.0",
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
        "scikit-learn>=1.3.0",
    ],
    extras_require={
        "dev": [
            "jupyter>=1.0.0",
            "ipykernel>=6.25.0",
            "ipywidgets>=8.0.0",
            "pytest>=7.4.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
        ],
        "deep-learning": [
            "torch>=2.0.0",
            "torchvision>=0.15.0",
            "transformers>=4.30.0",
            "sentence-transformers>=2.2.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "flickr30k-download=scripts.download_flickr30k:main",
        ],
    },
)
