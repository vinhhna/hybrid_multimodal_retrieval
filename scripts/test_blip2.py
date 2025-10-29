"""
BLIP-2 Cross-Encoder Testing Script

Automated validation script to verify BLIP-2 integration is working correctly.
Tests model loading, scoring, batch processing, and error handling.

Phase 3: Cross-Encoder Reranking
Created: October 28, 2025
Updated: October 28, 2025 (Switched to Hugging Face transformers)

Usage:
    python scripts/test_blip2.py
"""

import sys
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from typing import List, Tuple

# Add project root to path if needed
try:
    from src.retrieval.cross_encoder import CrossEncoder
except ImportError:
    # If import fails, add parent directory to path and try again
    project_root = Path(__file__).resolve().parent.parent
    if project_root not in sys.path:
        sys.path.insert(0, str(project_root))
    from src.retrieval.cross_encoder import CrossEncoder


class BLIP2Tester:
    """Test suite for BLIP-2 cross-encoder."""
    
    def __init__(self, data_dir=None):
        self.results = []
        self.encoder = None
        
        # Find data directory - search upwards from script location
        if data_dir:
            self.data_dir = Path(data_dir)
        else:
            script_dir = Path(__file__).resolve().parent
            current_dir = script_dir
            
            # Search for data directory by going up the tree
            while current_dir != current_dir.parent:
                data_path = current_dir / 'data'
                if data_path.exists() and (data_path / 'images').exists():
                    self.data_dir = data_path
                    break
                current_dir = current_dir.parent
            else:
                # Fallback: check current working directory
                if (Path.cwd() / 'data' / 'images').exists():
                    self.data_dir = Path.cwd() / 'data'
                else:
                    # Last resort: create relative path
                    self.data_dir = Path('data')
        
    def log_result(self, test_name: str, passed: bool, message: str = ""):
        """Log test result."""
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status} - {test_name}: {message}")
        self.results.append({
            'test': test_name,
            'passed': passed,
            'message': message
        })
    
    def test_model_loading(self) -> bool:
        """Test 1: Model loading and initialization."""
        print("\n" + "="*60)
        print("TEST 1: Model Loading")
        print("="*60)
        
        # Load the default model
        model_options = [
            ('Salesforce/blip2-flan-t5-xl', 'Default model - fits P100 16GB GPU'),
        ]
        
        for model_name, description in model_options:
            try:
                print(f"\nTrying: {model_name} ({description})")
                
                self.encoder = CrossEncoder(
                    model_name=model_name,
                    device='cuda' if torch.cuda.is_available() else 'cpu',
                    use_fp16=True  # Always use FP16 to save memory
                )
                
                # Check device
                device_ok = self.encoder.device.type in ['cuda', 'cpu']
                
                # Check model loaded
                model_ok = self.encoder.model is not None
                
                # Get model info
                info = self.encoder.get_model_info()
                print(f"Model info: {info}")
                
                passed = device_ok and model_ok
                self.log_result(
                    "Model Loading",
                    passed,
                    f"Model: {model_name}, Device: {self.encoder.device}, FP16: {self.encoder.use_fp16}"
                )
                
                if passed:
                    print(f"✓ Successfully loaded: {model_name}")
                    return True
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower() or "cuda" in str(e).lower():
                    print(f"✗ OOM with {model_name}, trying next model...")
                    # Clear CUDA cache before trying next model
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    self.encoder = None
                    continue
                else:
                    print(f"✗ Runtime error with {model_name}: {str(e)}")
                    self.encoder = None
                    continue
            except Exception as e:
                print(f"✗ Failed to load {model_name}: {str(e)}")
                self.encoder = None
                continue
        
        # If all models failed
        self.log_result("Model Loading", False, "All model options failed - GPU may be too small")
        print("\n⚠️  SOLUTION: Your GPU doesn't have enough memory for any BLIP-2 model.")
        print("    Options:")
        print("    1. Use Kaggle T4 GPU (15GB) or P100 with smaller model")
        print("    2. Use CPU (slow but works): device='cpu'")
        print("    3. Use Google Colab with T4/V100 GPU")
        return False
    
    def test_single_pair_scoring(self) -> bool:
        """Test 2: Single pair scoring."""
        print("\n" + "="*60)
        print("TEST 2: Single Pair Scoring")
        print("="*60)
        
        if self.encoder is None:
            self.log_result("Single Pair Scoring", False, "Encoder not initialized")
            return False
        
        try:
            # Find a test image
            images_dir = self.data_dir / 'images'
            if not images_dir.exists():
                self.log_result("Single Pair Scoring", False, "No images directory found")
                return False
            
            # Get first image
            test_images = list(images_dir.glob('*.jpg'))[:1]
            if not test_images:
                self.log_result("Single Pair Scoring", False, "No test images found")
                return False
            
            test_image = test_images[0]
            test_query = "A photograph"
            
            print(f"Testing with: {test_image.name}")
            print(f"Query: '{test_query}'")
            
            # Score single pair
            score = self.encoder.score_pair(test_query, test_image)
            
            # Validate score
            score_valid = isinstance(score, (float, np.floating)) and 0 <= score <= 1
            
            print(f"Score: {score:.4f}")
            
            self.log_result(
                "Single Pair Scoring",
                score_valid,
                f"Score: {score:.4f}, Valid: {score_valid}"
            )
            return score_valid
            
        except Exception as e:
            self.log_result("Single Pair Scoring", False, f"Error: {str(e)}")
            import traceback; traceback.print_exc()
            return False
    
    def test_batch_scoring(self) -> bool:
        """Test 3: Batch scoring with different batch sizes."""
        print("\n" + "="*60)
        print("TEST 3: Batch Scoring")
        print("="*60)
        
        if self.encoder is None:
            self.log_result("Batch Scoring", False, "Encoder not initialized")
            return False
        
        try:
            # Get test images
            images_dir = self.data_dir / 'images'
            test_images = list(images_dir.glob('*.jpg'))[:10]
            
            if len(test_images) < 5:
                self.log_result("Batch Scoring", False, "Not enough test images")
                return False
            
            # Create test queries - MUST match number of images
            base_queries = [
                "A photograph",
                "People in a scene",
                "An outdoor image",
                "A colorful picture",
                "A scene with activity"
            ]
            # Repeat queries to match number of images
            test_queries = (base_queries * ((len(test_images) // len(base_queries)) + 1))[:len(test_images)]
            
            print(f"Testing with {len(test_images)} pairs (queries: {len(test_queries)}, images: {len(test_images)})")
            
            # Test different batch sizes
            batch_sizes = [2, 4, 8]
            all_passed = True
            
            for batch_size in batch_sizes:
                try:
                    print(f"\nTesting batch size: {batch_size}")
                    scores = self.encoder.score_pairs(
                        test_queries,
                        test_images,
                        batch_size=batch_size,
                        show_progress=True
                    )
                    
                    # Validate scores
                    scores_valid = (
                        len(scores) == len(test_queries) and
                        all(0 <= s <= 1 for s in scores)
                    )
                    
                    print(f"Scores: {scores}")
                    print(f"Valid: {scores_valid}")
                    
                    if not scores_valid:
                        all_passed = False
                        
                except Exception as e:
                    print(f"Batch size {batch_size} failed: {e}")
                    all_passed = False
            
            self.log_result("Batch Scoring", all_passed, f"Tested batch sizes: {batch_sizes}")
            return all_passed
            
        except Exception as e:
            self.log_result("Batch Scoring", False, f"Error: {str(e)}")
            import traceback; traceback.print_exc()
            return False
    
    def test_memory_handling(self) -> bool:
        """Test 4: Memory management and OOM handling."""
        print("\n" + "="*60)
        print("TEST 4: Memory Handling")
        print("="*60)
        
        if self.encoder is None:
            self.log_result("Memory Handling", False, "Encoder not initialized")
            return False
        
        try:
            # Check initial memory
            if torch.cuda.is_available():
                initial_memory = torch.cuda.memory_allocated()
                print(f"Initial GPU memory: {initial_memory / 1e6:.2f} MB")
            
            # Test with fallback batch size
            images_dir = self.data_dir / 'images'
            test_images = list(images_dir.glob('*.jpg'))[:5]
            test_queries = ["Test query"] * len(test_images)
            
            # Force small batch size
            scores = self.encoder.score_pairs(
                test_queries,
                test_images,
                batch_size=2,
                show_progress=False
            )
            
            # Check memory cleared
            if torch.cuda.is_available() and self.encoder.clear_cache:
                torch.cuda.empty_cache()
                final_memory = torch.cuda.memory_allocated()
                print(f"Final GPU memory: {final_memory / 1e6:.2f} MB")
            
            passed = len(scores) == len(test_queries)
            self.log_result("Memory Handling", passed, "Memory management OK")
            return passed
            
        except Exception as e:
            self.log_result("Memory Handling", False, f"Error: {str(e)}")
            import traceback; traceback.print_exc()
            return False
    
    def test_score_quality(self) -> bool:
        """Test 5: Validate score quality and consistency."""
        print("\n" + "="*60)
        print("TEST 5: Score Quality")
        print("="*60)
        
        if self.encoder is None:
            self.log_result("Score Quality", False, "Encoder not initialized")
            return False
        
        try:
            # Get test image
            images_dir = self.data_dir / 'images'
            test_image = list(images_dir.glob('*.jpg'))[0]
            
            # Test with relevant and irrelevant queries
            queries = [
                "A photograph",  # Generic - should score moderately
                "Random text xyz 123",  # Irrelevant - should score lower
            ]
            
            scores = self.encoder.score_pairs(
                queries,
                [test_image] * len(queries),
                show_progress=False
            )
            
            print(f"Scores: {list(zip(queries, scores))}")
            
            # Check scores are different and in valid range
            scores_valid = (
                all(0 <= s <= 1 for s in scores) and
                len(set(scores)) > 1  # Should have variation
            )
            
            self.log_result("Score Quality", scores_valid, f"Scores: {scores}")
            return scores_valid
            
        except Exception as e:
            self.log_result("Score Quality", False, f"Error: {str(e)}")
            import traceback; traceback.print_exc()
            return False
    
    def generate_report(self) -> None:
        """Generate final test report."""
        print("\n" + "="*60)
        print("TEST REPORT")
        print("="*60)
        
        total = len(self.results)
        passed = sum(1 for r in self.results if r['passed'])
        failed = total - passed
        
        print(f"\nTotal Tests: {total}")
        print(f"Passed: {passed}")
        print(f"Failed: {failed}")
        print(f"Success Rate: {100 * passed / total:.1f}%")
        
        print("\nDetailed Results:")
        for result in self.results:
            status = "✓" if result['passed'] else "✗"
            print(f"  {status} {result['test']}: {result['message']}")
        
        print("\n" + "="*60)
        
        if failed == 0:
            print("✓ ALL TESTS PASSED!")
        else:
            print(f"✗ {failed} TEST(S) FAILED")
        
        print("="*60 + "\n")
    
    def run_all_tests(self) -> int:
        """
        Run all tests.
        
        Returns:
            Exit code (0 = success, 1 = failure)
        """
        print("Starting BLIP-2 Cross-Encoder Test Suite")
        print("="*60)
        
        # Run tests
        tests = [
            self.test_model_loading,
            self.test_single_pair_scoring,
            self.test_batch_scoring,
            self.test_memory_handling,
            self.test_score_quality
        ]
        
        for test in tests:
            try:
                test()
            except Exception as e:
                print(f"Test failed with exception: {e}")
                import traceback; traceback.print_exc()
        
        # Generate report
        self.generate_report()
        
        # Return exit code
        all_passed = all(r['passed'] for r in self.results)
        return 0 if all_passed else 1


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test BLIP-2 Cross-Encoder')
    parser.add_argument('--data-dir', type=str, default=None,
                       help='Path to data directory (default: auto-detect)')
    args = parser.parse_args()
    
    tester = BLIP2Tester(data_dir=args.data_dir)
    exit_code = tester.run_all_tests()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
