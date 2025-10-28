"""
BLIP-2 Cross-Encoder Testing Script

Automated validation script to verify BLIP-2 integration is working correctly.
Tests model loading, scoring, batch processing, and error handling.

Phase 3: Cross-Encoder Reranking
Created: October 28, 2025

Usage:
    python scripts/test_blip2.py
"""

import sys
import torch
import numpy as np
from pathlib import Path
from PIL import Image
import logging
from typing import List, Tuple

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from retrieval.cross_encoder import CrossEncoder

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BLIP2Tester:
    """Test suite for BLIP-2 cross-encoder."""
    
    def __init__(self):
        self.results = []
        self.encoder = None
        self.data_dir = Path(__file__).parent.parent / 'data'
        
    def log_result(self, test_name: str, passed: bool, message: str = ""):
        """Log test result."""
        status = "✓ PASS" if passed else "✗ FAIL"
        logger.info(f"{status} - {test_name}: {message}")
        self.results.append({
            'test': test_name,
            'passed': passed,
            'message': message
        })
    
    def test_model_loading(self) -> bool:
        """Test 1: Model loading and initialization."""
        logger.info("\n" + "="*60)
        logger.info("TEST 1: Model Loading")
        logger.info("="*60)
        
        try:
            self.encoder = CrossEncoder(
                model_name='blip2_opt',
                model_type='pretrain_opt2.7b'
            )
            
            # Check device
            device_ok = self.encoder.device.type in ['cuda', 'cpu']
            
            # Check model loaded
            model_ok = self.encoder.model is not None
            
            # Get model info
            info = self.encoder.get_model_info()
            logger.info(f"Model info: {info}")
            
            passed = device_ok and model_ok
            self.log_result(
                "Model Loading",
                passed,
                f"Device: {self.encoder.device}, FP16: {self.encoder.use_fp16}"
            )
            return passed
            
        except Exception as e:
            self.log_result("Model Loading", False, f"Error: {str(e)}")
            return False
    
    def test_single_pair_scoring(self) -> bool:
        """Test 2: Single pair scoring."""
        logger.info("\n" + "="*60)
        logger.info("TEST 2: Single Pair Scoring")
        logger.info("="*60)
        
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
            
            logger.info(f"Testing with: {test_image.name}")
            logger.info(f"Query: '{test_query}'")
            
            # Score single pair
            score = self.encoder.score_pair(test_query, test_image)
            
            # Validate score
            score_valid = isinstance(score, (float, np.floating)) and 0 <= score <= 1
            
            logger.info(f"Score: {score:.4f}")
            
            self.log_result(
                "Single Pair Scoring",
                score_valid,
                f"Score: {score:.4f}, Valid: {score_valid}"
            )
            return score_valid
            
        except Exception as e:
            self.log_result("Single Pair Scoring", False, f"Error: {str(e)}")
            logger.exception(e)
            return False
    
    def test_batch_scoring(self) -> bool:
        """Test 3: Batch scoring with different batch sizes."""
        logger.info("\n" + "="*60)
        logger.info("TEST 3: Batch Scoring")
        logger.info("="*60)
        
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
            
            # Create test queries
            test_queries = [
                "A photograph",
                "People in a scene",
                "An outdoor image",
                "A colorful picture",
                "A scene with activity"
            ][:len(test_images)]
            
            logger.info(f"Testing with {len(test_images)} pairs")
            
            # Test different batch sizes
            batch_sizes = [2, 4, 8]
            all_passed = True
            
            for batch_size in batch_sizes:
                try:
                    logger.info(f"\nTesting batch size: {batch_size}")
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
                    
                    logger.info(f"Scores: {scores}")
                    logger.info(f"Valid: {scores_valid}")
                    
                    if not scores_valid:
                        all_passed = False
                        
                except Exception as e:
                    logger.error(f"Batch size {batch_size} failed: {e}")
                    all_passed = False
            
            self.log_result("Batch Scoring", all_passed, f"Tested batch sizes: {batch_sizes}")
            return all_passed
            
        except Exception as e:
            self.log_result("Batch Scoring", False, f"Error: {str(e)}")
            logger.exception(e)
            return False
    
    def test_memory_handling(self) -> bool:
        """Test 4: Memory management and OOM handling."""
        logger.info("\n" + "="*60)
        logger.info("TEST 4: Memory Handling")
        logger.info("="*60)
        
        if self.encoder is None:
            self.log_result("Memory Handling", False, "Encoder not initialized")
            return False
        
        try:
            # Check initial memory
            if torch.cuda.is_available():
                initial_memory = torch.cuda.memory_allocated()
                logger.info(f"Initial GPU memory: {initial_memory / 1e6:.2f} MB")
            
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
                logger.info(f"Final GPU memory: {final_memory / 1e6:.2f} MB")
            
            passed = len(scores) == len(test_queries)
            self.log_result("Memory Handling", passed, "Memory management OK")
            return passed
            
        except Exception as e:
            self.log_result("Memory Handling", False, f"Error: {str(e)}")
            logger.exception(e)
            return False
    
    def test_score_quality(self) -> bool:
        """Test 5: Validate score quality and consistency."""
        logger.info("\n" + "="*60)
        logger.info("TEST 5: Score Quality")
        logger.info("="*60)
        
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
            
            logger.info(f"Scores: {list(zip(queries, scores))}")
            
            # Check scores are different and in valid range
            scores_valid = (
                all(0 <= s <= 1 for s in scores) and
                len(set(scores)) > 1  # Should have variation
            )
            
            self.log_result("Score Quality", scores_valid, f"Scores: {scores}")
            return scores_valid
            
        except Exception as e:
            self.log_result("Score Quality", False, f"Error: {str(e)}")
            logger.exception(e)
            return False
    
    def generate_report(self) -> None:
        """Generate final test report."""
        logger.info("\n" + "="*60)
        logger.info("TEST REPORT")
        logger.info("="*60)
        
        total = len(self.results)
        passed = sum(1 for r in self.results if r['passed'])
        failed = total - passed
        
        logger.info(f"\nTotal Tests: {total}")
        logger.info(f"Passed: {passed}")
        logger.info(f"Failed: {failed}")
        logger.info(f"Success Rate: {100 * passed / total:.1f}%")
        
        logger.info("\nDetailed Results:")
        for result in self.results:
            status = "✓" if result['passed'] else "✗"
            logger.info(f"  {status} {result['test']}: {result['message']}")
        
        logger.info("\n" + "="*60)
        
        if failed == 0:
            logger.info("✓ ALL TESTS PASSED!")
        else:
            logger.warning(f"✗ {failed} TEST(S) FAILED")
        
        logger.info("="*60 + "\n")
    
    def run_all_tests(self) -> int:
        """
        Run all tests.
        
        Returns:
            Exit code (0 = success, 1 = failure)
        """
        logger.info("Starting BLIP-2 Cross-Encoder Test Suite")
        logger.info("="*60)
        
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
                logger.error(f"Test failed with exception: {e}")
                logger.exception(e)
        
        # Generate report
        self.generate_report()
        
        # Return exit code
        all_passed = all(r['passed'] for r in self.results)
        return 0 if all_passed else 1


def main():
    """Main entry point."""
    tester = BLIP2Tester()
    exit_code = tester.run_all_tests()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
