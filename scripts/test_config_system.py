"""
Test Configuration System (T3.2)

This script tests the configuration management system for hybrid search.

Tests:
1. Load default configuration
2. Load configuration from YAML file
3. Load preset configurations
4. Validate configuration
5. Get/set configuration values
6. Save configuration
7. Configuration validation errors

Usage:
    python scripts/test_config_system.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.retrieval.config import (
    load_config,
    load_preset,
    get_default_config,
    ConfigurationError,
    HybridSearchConfig
)


def test_default_config():
    """Test 1: Load default configuration."""
    print("\n" + "="*70)
    print("TEST 1: Load Default Configuration")
    print("="*70)
    
    try:
        config = load_config()
        print("\n✓ Default configuration loaded successfully")
        print(f"\n{config}")
        
        # Test default values
        assert config.get('stage1.k1') == 100
        assert config.get('stage2.k2') == 10
        assert config.get('stage2.batch_size') == 4
        print("\n✓ Default values verified")
        
        return True
    except Exception as e:
        print(f"\n✗ Error: {e}")
        return False


def test_load_yaml_config():
    """Test 2: Load configuration from YAML file."""
    print("\n" + "="*70)
    print("TEST 2: Load Configuration from YAML")
    print("="*70)
    
    try:
        config_path = project_root / 'configs' / 'hybrid_config.yaml'
        config = load_config(str(config_path))
        print(f"\n✓ Configuration loaded from {config_path}")
        print(f"\n{config}")
        
        # Verify loaded values
        print("\nConfiguration values:")
        print(f"  Stage 1 k1: {config.get('stage1.k1')}")
        print(f"  Stage 2 k2: {config.get('stage2.k2')}")
        print(f"  Batch size: {config.get('stage2.batch_size')}")
        print(f"  Use cache: {config.get('performance.use_cache')}")
        print(f"  Show progress: {config.get('performance.show_progress')}")
        
        return True
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_preset_configs():
    """Test 3: Load preset configurations."""
    print("\n" + "="*70)
    print("TEST 3: Load Preset Configurations")
    print("="*70)
    
    presets = ['fast', 'accurate', 'balanced', 'memory_efficient']
    
    for preset_name in presets:
        try:
            print(f"\n--- Testing preset: {preset_name} ---")
            config = load_preset(preset_name)
            
            print(f"  k1: {config.get('stage1.k1')}")
            print(f"  k2: {config.get('stage2.k2')}")
            print(f"  batch_size: {config.get('stage2.batch_size')}")
            
            if preset_name == 'fast':
                assert config.get('stage1.k1') == 50
                assert config.get('performance.use_cache') == True
            elif preset_name == 'accurate':
                assert config.get('stage1.k1') == 200
                assert config.get('stage2.k2') == 20
            elif preset_name == 'balanced':
                assert config.get('stage1.k1') == 100
                assert config.get('stage2.k2') == 10
            elif preset_name == 'memory_efficient':
                assert config.get('stage1.k1') == 50
                assert config.get('performance.optimize_memory') == True
            
            print(f"  ✓ Preset '{preset_name}' validated")
            
        except Exception as e:
            print(f"  ✗ Error loading preset '{preset_name}': {e}")
            return False
    
    print("\n✓ All presets loaded successfully")
    return True


def test_get_set_values():
    """Test 4: Get and set configuration values."""
    print("\n" + "="*70)
    print("TEST 4: Get and Set Configuration Values")
    print("="*70)
    
    try:
        config = load_config()
        
        # Test getting values
        print("\nGetting values:")
        k1 = config.get('stage1.k1')
        print(f"  stage1.k1 = {k1}")
        
        k2 = config.get('stage2.k2')
        print(f"  stage2.k2 = {k2}")
        
        # Test getting nested values
        model = config.get('stage2.model')
        print(f"  stage2.model = {model}")
        
        # Test default values
        non_existent = config.get('non.existent.key', 'default_value')
        print(f"  non.existent.key = {non_existent} (default)")
        
        print("\n✓ Getting values works")
        
        # Test setting values
        print("\nSetting values:")
        config.set('stage1.k1', 150)
        new_k1 = config.get('stage1.k1')
        print(f"  stage1.k1 = {new_k1} (changed from {k1})")
        assert new_k1 == 150
        
        config.set('stage2.k2', 15)
        new_k2 = config.get('stage2.k2')
        print(f"  stage2.k2 = {new_k2} (changed from {k2})")
        assert new_k2 == 15
        
        print("\n✓ Setting values works")
        
        return True
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_validation():
    """Test 5: Configuration validation."""
    print("\n" + "="*70)
    print("TEST 5: Configuration Validation")
    print("="*70)
    
    test_cases = [
        {
            'name': 'Invalid k1 (< 1)',
            'key': 'stage1.k1',
            'value': 0,
            'should_fail': True
        },
        {
            'name': 'Invalid k2 > k1',
            'setup': lambda c: c.set('stage1.k1', 50),
            'key': 'stage2.k2',
            'value': 100,
            'should_fail': True
        },
        {
            'name': 'Invalid batch_size (< 1)',
            'key': 'stage2.batch_size',
            'value': 0,
            'should_fail': True
        },
        {
            'name': 'Invalid device',
            'key': 'stage1.device',
            'value': 'invalid',
            'should_fail': True
        },
        {
            'name': 'Valid k1',
            'key': 'stage1.k1',
            'value': 200,
            'should_fail': False
        },
        {
            'name': 'Valid k2',
            'key': 'stage2.k2',
            'value': 20,
            'should_fail': False
        }
    ]
    
    for test_case in test_cases:
        print(f"\nTesting: {test_case['name']}")
        
        try:
            config = load_config()
            
            # Run setup if provided
            if 'setup' in test_case:
                test_case['setup'](config)
            
            # Try to set value
            config.set(test_case['key'], test_case['value'])
            
            if test_case['should_fail']:
                print(f"  ✗ Should have failed but didn't")
                return False
            else:
                print(f"  ✓ Valid configuration accepted")
                
        except ConfigurationError as e:
            if test_case['should_fail']:
                print(f"  ✓ Correctly rejected: {e}")
            else:
                print(f"  ✗ Should have succeeded but failed: {e}")
                return False
    
    print("\n✓ All validation tests passed")
    return True


def test_save_load():
    """Test 6: Save and load configuration."""
    print("\n" + "="*70)
    print("TEST 6: Save and Load Configuration")
    print("="*70)
    
    try:
        # Create custom configuration
        config = load_config()
        config.set('stage1.k1', 150)
        config.set('stage2.k2', 15)
        config.set('stage2.batch_size', 8)
        
        # Save to temp file
        temp_path = project_root / 'configs' / 'test_config_temp.yaml'
        config.save(str(temp_path))
        print(f"\n✓ Configuration saved to {temp_path}")
        
        # Load from temp file
        loaded_config = load_config(str(temp_path))
        print(f"✓ Configuration loaded from {temp_path}")
        
        # Verify values
        assert loaded_config.get('stage1.k1') == 150
        assert loaded_config.get('stage2.k2') == 15
        assert loaded_config.get('stage2.batch_size') == 8
        print("\n✓ Saved and loaded values match")
        
        # Clean up
        temp_path.unlink()
        print(f"✓ Cleaned up temp file")
        
        return True
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_to_dict():
    """Test 7: Convert configuration to dictionary."""
    print("\n" + "="*70)
    print("TEST 7: Convert Configuration to Dictionary")
    print("="*70)
    
    try:
        config = load_config()
        config_dict = config.to_dict()
        
        print("\n✓ Configuration converted to dictionary")
        print(f"\nConfiguration keys: {list(config_dict.keys())}")
        
        # Verify structure
        assert 'stage1' in config_dict
        assert 'stage2' in config_dict
        assert 'performance' in config_dict
        assert 'batch_search' in config_dict
        
        print("✓ Dictionary structure validated")
        
        return True
    except Exception as e:
        print(f"\n✗ Error: {e}")
        return False


def main():
    """Run all configuration tests."""
    print("\n" + "="*70)
    print("CONFIGURATION SYSTEM TEST SUITE (T3.2)")
    print("="*70)
    print("\nTesting configuration management for hybrid search system")
    
    tests = [
        ("Default Configuration", test_default_config),
        ("YAML Configuration", test_load_yaml_config),
        ("Preset Configurations", test_preset_configs),
        ("Get/Set Values", test_get_set_values),
        ("Validation", test_validation),
        ("Save/Load", test_save_load),
        ("To Dictionary", test_to_dict)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n✗ Test '{test_name}' failed with exception: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\n{'='*70}")
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n✓ All configuration tests passed! 🎉")
        print("="*70)
        return 0
    else:
        print(f"\n✗ {total - passed} test(s) failed")
        print("="*70)
        return 1


if __name__ == "__main__":
    exit(main())
