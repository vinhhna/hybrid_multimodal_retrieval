"""
Configuration loader for hybrid search system.

This module provides utilities for loading and validating configuration
from YAML files.
"""

import os
import yaml
from typing import Dict, Any, Optional
from pathlib import Path


class ConfigurationError(Exception):
    """Exception raised for configuration errors."""
    pass


class HybridSearchConfig:
    """
    Configuration manager for hybrid search system.
    
    Loads configuration from YAML file and provides validation
    and easy access to configuration parameters.
    """
    
    # Default configuration
    DEFAULT_CONFIG = {
        'stage1': {
            'model': 'ViT-B-32',
            'k1': 100,
            'device': 'cuda'
        },
        'stage2': {
            'model': 'Salesforce/blip2-opt-2.7b',
            'k2': 10,
            'batch_size': 4,
            'use_fp16': True,
            'device': 'cuda'
        },
        'performance': {
            'use_cache': False,
            'max_cache_size': 1000,
            'show_progress': True,
            'optimize_memory': True
        },
        'batch_search': {
            'stage1_batch_size': 32,
            'stage2_batch_size': 8
        },
        'image_to_image': {
            'k': 10,
            'use_hybrid': False
        },
        'advanced': {
            'score_fusion': 'replace',
            'clip_weight': 0.3,
            'blip2_weight': 0.7,
            'normalize_scores': True,
            'use_diversity': False,
            'diversity_lambda': 0.7
        },
        'logging': {
            'level': 'INFO',
            'log_timings': True,
            'track_statistics': True
        }
    }
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_path: Path to YAML configuration file.
                        If None, uses default configuration.
        """
        self.config = self._load_config(config_path)
        self._validate_config()
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Configuration dictionary
        """
        if config_path is None:
            # Use default configuration
            return self.DEFAULT_CONFIG.copy()
        
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise ConfigurationError(f"Configuration file not found: {config_path}")
        
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ConfigurationError(f"Error parsing YAML configuration: {e}")
        
        if config is None:
            # Empty file, use defaults
            return self.DEFAULT_CONFIG.copy()
        
        # Merge with defaults (user config overrides defaults)
        merged_config = self._deep_merge(self.DEFAULT_CONFIG.copy(), config)
        
        return merged_config
    
    def _deep_merge(self, base: Dict, override: Dict) -> Dict:
        """
        Deep merge two dictionaries.
        
        Args:
            base: Base dictionary
            override: Override dictionary
            
        Returns:
            Merged dictionary
        """
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def _validate_config(self):
        """
        Validate configuration parameters.
        
        Raises:
            ConfigurationError: If configuration is invalid
        """
        # Validate stage1
        if self.config['stage1']['k1'] < 1:
            raise ConfigurationError("stage1.k1 must be >= 1")
        
        if self.config['stage1']['device'] not in ['cuda', 'cpu']:
            raise ConfigurationError("stage1.device must be 'cuda' or 'cpu'")
        
        # Validate stage2
        if self.config['stage2']['k2'] < 1:
            raise ConfigurationError("stage2.k2 must be >= 1")
        
        if self.config['stage2']['k2'] > self.config['stage1']['k1']:
            raise ConfigurationError("stage2.k2 cannot be greater than stage1.k1")
        
        if self.config['stage2']['batch_size'] < 1:
            raise ConfigurationError("stage2.batch_size must be >= 1")
        
        if self.config['stage2']['device'] not in ['cuda', 'cpu']:
            raise ConfigurationError("stage2.device must be 'cuda' or 'cpu'")
        
        # Validate performance
        if self.config['performance']['max_cache_size'] < 0:
            raise ConfigurationError("performance.max_cache_size must be >= 0")
        
        # Validate batch_search
        if self.config['batch_search']['stage1_batch_size'] < 1:
            raise ConfigurationError("batch_search.stage1_batch_size must be >= 1")
        
        if self.config['batch_search']['stage2_batch_size'] < 1:
            raise ConfigurationError("batch_search.stage2_batch_size must be >= 1")
        
        # Validate advanced
        valid_fusion_methods = ['replace', 'weighted', 'rank_fusion']
        if self.config['advanced']['score_fusion'] not in valid_fusion_methods:
            raise ConfigurationError(
                f"advanced.score_fusion must be one of {valid_fusion_methods}"
            )
        
        if self.config['advanced']['clip_weight'] < 0 or self.config['advanced']['clip_weight'] > 1:
            raise ConfigurationError("advanced.clip_weight must be between 0 and 1")
        
        if self.config['advanced']['blip2_weight'] < 0 or self.config['advanced']['blip2_weight'] > 1:
            raise ConfigurationError("advanced.blip2_weight must be between 0 and 1")
        
        if self.config['advanced']['diversity_lambda'] < 0 or self.config['advanced']['diversity_lambda'] > 1:
            raise ConfigurationError("advanced.diversity_lambda must be between 0 and 1")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key.
        
        Supports nested keys using dot notation (e.g., 'stage1.k1').
        
        Args:
            key: Configuration key
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any):
        """
        Set configuration value by key.
        
        Supports nested keys using dot notation (e.g., 'stage1.k1').
        
        Args:
            key: Configuration key
            value: Configuration value
        """
        keys = key.split('.')
        config = self.config
        
        # Navigate to the parent dictionary
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        # Set the value
        config[keys[-1]] = value
        
        # Re-validate configuration
        self._validate_config()
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Get configuration as dictionary.
        
        Returns:
            Configuration dictionary
        """
        return self.config.copy()
    
    def save(self, path: str):
        """
        Save configuration to YAML file.
        
        Args:
            path: Path to save configuration
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False, sort_keys=False)
    
    def __repr__(self) -> str:
        """String representation of configuration."""
        return f"HybridSearchConfig({self.config})"
    
    def __str__(self) -> str:
        """Human-readable string representation."""
        lines = ["Hybrid Search Configuration:"]
        
        # Stage 1
        lines.append("\nStage 1 (CLIP):")
        lines.append(f"  Model: {self.get('stage1.model')}")
        lines.append(f"  k1 (candidates): {self.get('stage1.k1')}")
        lines.append(f"  Device: {self.get('stage1.device')}")
        
        # Stage 2
        lines.append("\nStage 2 (BLIP-2):")
        lines.append(f"  Model: {self.get('stage2.model')}")
        lines.append(f"  k2 (results): {self.get('stage2.k2')}")
        lines.append(f"  Batch size: {self.get('stage2.batch_size')}")
        lines.append(f"  FP16: {self.get('stage2.use_fp16')}")
        lines.append(f"  Device: {self.get('stage2.device')}")
        
        # Performance
        lines.append("\nPerformance:")
        lines.append(f"  Cache: {self.get('performance.use_cache')}")
        lines.append(f"  Max cache size: {self.get('performance.max_cache_size')}")
        lines.append(f"  Show progress: {self.get('performance.show_progress')}")
        lines.append(f"  Optimize memory: {self.get('performance.optimize_memory')}")
        
        return "\n".join(lines)


def load_config(config_path: Optional[str] = None) -> HybridSearchConfig:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file. If None, uses default config.
        
    Returns:
        HybridSearchConfig instance
        
    Example:
        >>> config = load_config('configs/hybrid_config.yaml')
        >>> k1 = config.get('stage1.k1')
        >>> config.set('stage2.k2', 20)
    """
    return HybridSearchConfig(config_path)


def get_default_config() -> Dict[str, Any]:
    """
    Get default configuration as dictionary.
    
    Returns:
        Default configuration dictionary
    """
    return HybridSearchConfig.DEFAULT_CONFIG.copy()


# Preset configurations
PRESETS = {
    'fast': {
        'stage1': {'k1': 50},
        'stage2': {'k2': 5, 'batch_size': 8},
        'performance': {'use_cache': True}
    },
    'accurate': {
        'stage1': {'k1': 200},
        'stage2': {'k2': 20, 'batch_size': 2}
    },
    'balanced': {
        'stage1': {'k1': 100},
        'stage2': {'k2': 10, 'batch_size': 4}
    },
    'memory_efficient': {
        'stage1': {'k1': 50},
        'stage2': {'k2': 5, 'batch_size': 2, 'use_fp16': True},
        'performance': {'optimize_memory': True}
    }
}


def load_preset(preset_name: str) -> HybridSearchConfig:
    """
    Load a preset configuration.
    
    Available presets:
    - 'fast': Prioritize speed over accuracy
    - 'accurate': Prioritize accuracy over speed
    - 'balanced': Balance between speed and accuracy
    - 'memory_efficient': For limited GPU memory (8GB)
    
    Args:
        preset_name: Name of preset
        
    Returns:
        HybridSearchConfig instance with preset applied
        
    Raises:
        ConfigurationError: If preset name is invalid
        
    Example:
        >>> config = load_preset('fast')
        >>> print(config.get('stage1.k1'))
        50
    """
    if preset_name not in PRESETS:
        raise ConfigurationError(
            f"Unknown preset '{preset_name}'. "
            f"Available presets: {list(PRESETS.keys())}"
        )
    
    # Start with default config
    config = HybridSearchConfig()
    
    # Apply preset
    preset = PRESETS[preset_name]
    for section, params in preset.items():
        for key, value in params.items():
            config.set(f"{section}.{key}", value)
    
    return config
