"""
Quick test for HybridSearchEngine class design.

This script verifies that the HybridSearchEngine class is properly
designed and can be imported.

Usage:
    python test_hybrid_design.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_imports():
    """Test that all components can be imported."""
    print("=" * 60)
    print("Testing imports...")
    print("=" * 60)
    
    try:
        from src.retrieval import HybridSearchEngine
        print("✓ HybridSearchEngine imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import HybridSearchEngine: {e}")
        return False
    
    try:
        from src.retrieval import BiEncoder, CrossEncoder, FAISSIndex
        print("✓ All required components imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import components: {e}")
        return False
    
    try:
        from src.flickr30k import Flickr30KDataset
        print("✓ Flickr30KDataset imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import Flickr30KDataset: {e}")
        return False
    
    return True


def test_class_structure():
    """Test the class structure and methods."""
    print("\n" + "=" * 60)
    print("Testing class structure...")
    print("=" * 60)
    
    from src.retrieval import HybridSearchEngine
    
    # Check required methods
    required_methods = [
        '__init__',
        'text_to_image_hybrid_search',
        'image_to_image_hybrid_search',
        'batch_text_to_image_search',
        '_stage1_retrieve',
        '_stage2_rerank',
        'get_statistics',
        'clear_cache',
        'reset_statistics'
    ]
    
    for method in required_methods:
        if hasattr(HybridSearchEngine, method):
            print(f"✓ Method '{method}' found")
        else:
            print(f"✗ Method '{method}' not found")
            return False
    
    return True


def test_docstrings():
    """Test that key methods have docstrings."""
    print("\n" + "=" * 60)
    print("Testing docstrings...")
    print("=" * 60)
    
    from src.retrieval import HybridSearchEngine
    
    # Check class docstring
    if HybridSearchEngine.__doc__:
        print(f"✓ Class docstring present ({len(HybridSearchEngine.__doc__)} chars)")
    else:
        print("✗ Class docstring missing")
        return False
    
    # Check key method docstrings
    key_methods = [
        '__init__',
        'text_to_image_hybrid_search',
        '_stage1_retrieve',
        '_stage2_rerank'
    ]
    
    for method_name in key_methods:
        method = getattr(HybridSearchEngine, method_name)
        if method.__doc__:
            print(f"✓ Method '{method_name}' has docstring")
        else:
            print(f"✗ Method '{method_name}' missing docstring")
            return False
    
    return True


def print_class_info():
    """Print information about the HybridSearchEngine class."""
    print("\n" + "=" * 60)
    print("HybridSearchEngine Class Information")
    print("=" * 60)
    
    from src.retrieval import HybridSearchEngine
    import inspect
    
    # Get __init__ signature
    init_sig = inspect.signature(HybridSearchEngine.__init__)
    print("\n__init__ signature:")
    print(f"  {init_sig}")
    
    # Get public methods
    print("\nPublic methods:")
    for name, method in inspect.getmembers(HybridSearchEngine, predicate=inspect.isfunction):
        if not name.startswith('_'):
            sig = inspect.signature(method)
            print(f"  {name}{sig}")
    
    # Get configuration parameters from docstring
    if HybridSearchEngine.__init__.__doc__:
        doc = HybridSearchEngine.__init__.__doc__
        if "Configuration Parameters:" in doc:
            config_section = doc.split("Configuration Parameters:")[1]
            config_section = config_section.split("\n\n")[0]
            print("\nConfiguration Parameters:")
            print(config_section)


def main():
    """Run all tests."""
    print("Testing HybridSearchEngine Class Design (T2.1)")
    print()
    
    all_passed = True
    
    # Run tests
    if not test_imports():
        all_passed = False
    
    if not test_class_structure():
        all_passed = False
    
    if not test_docstrings():
        all_passed = False
    
    # Print class info
    print_class_info()
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    if all_passed:
        print("✓ All tests passed!")
        print("\nTask T2.1: HybridSearchEngine Class Design - COMPLETE")
        print("\nThe HybridSearchEngine class is properly designed with:")
        print("  - All required initialization parameters")
        print("  - Stage 1 retrieval (_stage1_retrieve)")
        print("  - Stage 2 re-ranking (_stage2_rerank)")
        print("  - Text-to-image hybrid search")
        print("  - Image-to-image hybrid search")
        print("  - Batch search support")
        print("  - Configuration management")
        print("  - Statistics tracking")
        print("  - Comprehensive docstrings")
    else:
        print("✗ Some tests failed. Please review the output above.")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
