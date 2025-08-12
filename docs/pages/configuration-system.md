---
layout: default
title: Configuration System
description: Configuration file system usage and management
permalink: /pages/configuration-system/
---

# Configuration System

This document provides detailed introduction to the VIVTransformer project's configuration system design, including configuration file structure, parameter management, and dynamic configuration features.

## 📋 Table of Contents

- [Configuration System Overview](#configuration-system-overview)
- [Main Configuration File](#main-configuration-file)
- [Loss Configuration Management](#loss-configuration-management)
- [Dynamic Configuration](#dynamic-configuration)
- [Configuration Validation](#configuration-validation)
- [Environment Configuration](#environment-configuration)
- [Best Practices](#best-practices)

## Configuration System Overview

### 🏗️ Configuration Architecture

```
Configuration System
├── 📁 configs/
│   ├── config.yaml              # Main configuration file
│   ├── 📁 loss_configs/         # Loss function configurations
│   │   ├── loss_config_0.yaml
│   │   ├── loss_config_1.yaml
│   │   └── ...
│   ├── 📁 model_configs/        # Model configurations
│   ├── 📁 data_configs/         # Data configurations
│   └── 📁 experiment_configs/   # Experiment configurations
├── utils/config.py              # Configuration management tools
└── generate_loss_configs.py     # Configuration generation script
```

### 🎯 Design Principles

1. **Hierarchical Structure**: Configurations are organized hierarchically by functional modules
2. **Extensibility**: Support for dynamic addition of new configuration items
3. **Type Safety**: Strict type checking and validation
4. **Environment Adaptation**: Support for configuration overrides in different environments
5. **Version Control**: Version management for configuration changes

## Main Configuration File

### 📝 config.yaml Structure

```yaml
# ==================== Global Configuration ====================
global:
  # Basic settings
  seed: 42                        # Random seed
  deterministic: true             # Deterministic training
  device: "cuda:0"                # Computing device
  mixed_precision: false          # Mixed precision training
  
  # Memory management
  max_memory_fraction: 0.8        # GPU memory usage limit
  memory_growth: true             # Dynamic memory growth
  
  # Debug settings
  debug: false                    # Debug mode
  verbose: true                   # Verbose output
  log_level: "INFO"               # Log level

# ==================== Data Configuration ====================
data:
  # Data paths
  path: "data/viv_dataset.pt"     # Data file path
  cache_dir: "cache/"             # Cache directory
  
  # Data loading
  batch_size: 128                 # Batch size
  num_workers: 4                  # Number of data loading processes
  pin_memory: true                # Memory pinning
  prefetch_factor: 2              # Prefetch factor
  
  # Data preprocessing
  normalize: true                 # Data normalization
  crop_size: [400]                # Crop size
  use_augmentation: true          # Data augmentation
  
  # Data splitting
  train_ratio: 0.7                # Training set ratio
  valid_ratio: 0.15               # Validation set ratio
  test_ratio: 0.15                # Test set ratio

# ==================== Model Configuration ====================
model:
  # Basic architecture
  attention_type: "self"          # Attention mechanism type
  d_model: 256                    # Model dimension
  num_heads: 4                    # Number of attention heads
  num_layers: 6                   # Number of layers
  
  # Input/Output
  input_dim: 400                  # Input dimension
  output_dim: 40000               # Output dimension
  seq_len: 49                     # Sequence length
  
  # Regularization
  dropout: 0.1                    # Dropout rate
  layer_norm_eps: 1e-6            # Layer normalization epsilon
  
  # Initialization
  init_method: "xavier_uniform"   # Weight initialization method
  bias_init: 0.0                  # Bias initialization value

# ==================== Training Configuration ====================
training:
  # Basic settings
  epochs: 10                      # Number of training epochs
  learning_rate: 0.0001           # Learning rate
  weight_decay: 0.01              # Weight decay
  
  # Optimizer
  optimizer: "adamw"              # Optimizer type
  beta1: 0.9                      # Adam beta1
  beta2: 0.999                    # Adam beta2
  eps: 1e-8                       # Adam epsilon
  
  # Learning rate scheduling
  lr_scheduler: "cosine"          # Learning rate scheduler
  warmup_epochs: 2                # Warmup epochs
  min_lr: 1e-6                    # Minimum learning rate
  
  # Early stopping
  early_stop: true                # Enable early stopping
  early_stop_patience: 10         # Early stopping patience
  early_stop_delta: 1e-4          # Early stopping threshold
  
  # Gradient processing
  gradient_clip_norm: 1.0         # Gradient clipping
  gradient_accumulation_steps: 1  # Gradient accumulation steps

# ==================== Loss Configuration ====================
loss:
  # Basic loss
  base_loss: "mse"                # Base loss function
  base_weight: 0.5                # Base loss weight
  
  # SVD regularization
  use_svd_regularization: true    # Enable SVD regularization
  svd_weights: [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
  
  # Other losses
  auxiliary_losses: []            # Auxiliary loss functions

# ==================== Evaluation Configuration ====================
evaluation:
  # Evaluation frequency
  eval_frequency: 1               # Evaluation frequency (every N epochs)
  save_frequency: 5               # Model save frequency
  
  # Evaluation metrics
  metrics: ["mse", "mae", "r2"]   # Evaluation metrics
  
  # Visualization
  visualize_attention: true       # Visualize attention
  save_predictions: false         # Save predictions

# ==================== Logging Configuration ====================
logging:
  # Basic settings
  log_dir: "logs/"                # Log directory
  experiment_name: "vivtransformer" # Experiment name
  
  # TensorBoard
  use_tensorboard: true           # Enable TensorBoard
  tensorboard_port: 6006          # TensorBoard port
  
  # Log content
  log_model_summary: true         # Log model summary
  log_gradients: false            # Log gradients
  log_weights: false              # Log weights
  
  # Save settings
  save_config: true               # Save configuration file
  save_code: true                 # Save code snapshot

# ==================== Attention Test Configuration ====================
attention_test:
  # Test settings
  enabled: true                   # Enable attention testing
  types: ["self", "muse", "ufo", "eca", "se"]  # Attention types to test
  
  # Parallel settings
  parallel: false                 # Parallel testing
  max_workers: 4                  # Maximum worker processes
  
  # Result saving
  save_results: true              # Save test results
  results_dir: "attention_results/" # Results save directory
```

### 🔧 Configuration Loader

```python
import yaml
import os
from typing import Dict, Any, Optional
from pathlib import Path

class ConfigLoader:
    """Configuration loader"""
    
    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        self.config = None
        
    def load(self) -> Dict[str, Any]:
        """Load configuration file"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # Apply environment variable overrides
        self._apply_env_overrides()
        
        # Validate configuration
        self._validate_config()
        
        return self.config
    
    def _apply_env_overrides(self):
        """Apply environment variable overrides"""
        env_prefix = "VIV_"
        
        for key, value in os.environ.items():
            if key.startswith(env_prefix):
                config_key = key[len(env_prefix):].lower().replace('_', '.')
                self._set_nested_value(self.config, config_key, value)
    
    def _set_nested_value(self, config: dict, key_path: str, value: str):
        """Set nested configuration value"""
        keys = key_path.split('.')
        current = config
        
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        # Type conversion
        final_key = keys[-1]
        if final_key in current:
            original_type = type(current[final_key])
            if original_type == bool:
                current[final_key] = value.lower() in ('true', '1', 'yes')
            elif original_type == int:
                current[final_key] = int(value)
            elif original_type == float:
                current[final_key] = float(value)
            else:
                current[final_key] = value
```

## Loss Configuration Management

### 📊 Loss Configuration Generation

```python
import numpy as np
import yaml
from pathlib import Path

class LossConfigGenerator:
    """Loss configuration generator"""
    
    def __init__(self, output_dir: str = "configs/loss_configs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_configs(self, num_configs: int = 50):
        """Generate multiple loss configurations"""
        configs = []
        
        for i in range(num_configs):
            config = self._generate_single_config(i)
            configs.append(config)
            
            # Save individual config file
            config_path = self.output_dir / f"loss_config_{i}.yaml"
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        
        print(f"✅ Generated {num_configs} loss configuration files")
        return configs
    
    def _generate_single_config(self, config_id: int) -> Dict[str, Any]:
        """Generate single loss configuration"""
        
        # Base weight range
        base_weight = np.random.uniform(0.1, 1.0)
        
        # SVD weight generation strategies
        svd_strategies = [
            self._uniform_weights,
            self._decreasing_weights,
            self._increasing_weights,
            self._random_weights,
            self._sparse_weights
        ]
        
        strategy = svd_strategies[config_id % len(svd_strategies)]
        svd_weights = strategy()
        
        config = {
            'config_id': config_id,
            'base_weight': round(base_weight, 3),
            'svd_weights': [round(w, 4) for w in svd_weights],
            'description': f"Loss configuration {config_id}",
            'strategy': strategy.__name__
        }
        
        return config
    
    def _uniform_weights(self) -> List[float]:
        """Uniform weight strategy"""
        weight = np.random.uniform(0.01, 0.2)
        return [weight] * 10
    
    def _decreasing_weights(self) -> List[float]:
        """Decreasing weight strategy"""
        start_weight = np.random.uniform(0.1, 0.3)
        decay_factor = np.random.uniform(0.7, 0.9)
        weights = [start_weight * (decay_factor ** i) for i in range(10)]
        return weights
    
    def _increasing_weights(self) -> List[float]:
        """Increasing weight strategy"""
        start_weight = np.random.uniform(0.01, 0.05)
        growth_factor = np.random.uniform(1.1, 1.3)
        weights = [start_weight * (growth_factor ** i) for i in range(10)]
        return weights
    
    def _random_weights(self) -> List[float]:
        """Random weight strategy"""
        return np.random.uniform(0.01, 0.2, 10).tolist()
    
    def _sparse_weights(self) -> List[float]:
        """Sparse weight strategy"""
        weights = np.zeros(10)
        num_nonzero = np.random.randint(2, 6)
        indices = np.random.choice(10, num_nonzero, replace=False)
        weights[indices] = np.random.uniform(0.05, 0.3, num_nonzero)
        return weights.tolist()
```

### 📋 Loss Configuration Template

```yaml
# loss_config_template.yaml
config_id: 0
description: "Base loss configuration template"

# Base loss settings
base_weight: 0.5
base_loss_type: "mse"

# SVD regularization settings
use_svd_regularization: true
svd_weights: [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
svd_components: 10

# Advanced settings
weight_decay: 0.01
label_smoothing: 0.0
focal_loss_gamma: 2.0

# Dynamic weighting
dynamic_weighting: false
weight_schedule: "constant"  # constant, linear, cosine

# Loss combination
loss_combination: "weighted_sum"  # weighted_sum, adaptive, uncertainty

# Metadata
created_by: "auto_generator"
creation_time: "2024-01-01T00:00:00"
version: "1.0"
```

## Dynamic Configuration

### 🔄 Runtime Configuration Modification

```python
class DynamicConfig:
    """Dynamic configuration manager"""
    
    def __init__(self, base_config: Dict[str, Any]):
        self.base_config = base_config.copy()
        self.current_config = base_config.copy()
        self.config_history = [base_config.copy()]
    
    def update(self, updates: Dict[str, Any]):
        """Update configuration"""
        self._deep_update(self.current_config, updates)
        self.config_history.append(self.current_config.copy())
    
    def rollback(self, steps: int = 1):
        """Rollback configuration"""
        if len(self.config_history) > steps:
            self.current_config = self.config_history[-(steps + 1)].copy()
            self.config_history = self.config_history[:-(steps)]
    
    def reset(self):
        """Reset to base configuration"""
        self.current_config = self.base_config.copy()
        self.config_history = [self.base_config.copy()]
    
    def _deep_update(self, base_dict: dict, update_dict: dict):
        """Deep update dictionary"""
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value

# Usage example
config_manager = DynamicConfig(base_config)

# Dynamically adjust learning rate
config_manager.update({
    'training': {
        'learning_rate': 0.0005
    }
})

# Dynamically adjust batch size
config_manager.update({
    'data': {
        'batch_size': 64
    }
})
```

### 🎛️ Adaptive Configuration

```python
class AdaptiveConfig:
    """Adaptive configuration manager"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.performance_history = []
        self.adaptation_rules = self._setup_adaptation_rules()
    
    def adapt_based_on_performance(self, metrics: Dict[str, float]):
        """Adaptively adjust configuration based on performance metrics"""
        self.performance_history.append(metrics)
        
        # Apply adaptation rules
        for rule in self.adaptation_rules:
            if rule.should_apply(metrics, self.performance_history):
                updates = rule.get_updates(metrics, self.config)
                self._apply_updates(updates)
                print(f"🔄 Applied adaptation rule: {rule.name}")
    
    def _setup_adaptation_rules(self):
        """Setup adaptation rules"""
        return [
            LearningRateAdaptationRule(),
            BatchSizeAdaptationRule(),
            DropoutAdaptationRule(),
            EarlyStopAdaptationRule()
        ]

class AdaptationRule:
    """Base adaptation rule class"""
    
    def __init__(self, name: str):
        self.name = name
    
    def should_apply(self, current_metrics: Dict[str, float], 
                    history: List[Dict[str, float]]) -> bool:
        raise NotImplementedError
    
    def get_updates(self, metrics: Dict[str, float], 
                   config: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError

class LearningRateAdaptationRule(AdaptationRule):
    """Learning rate adaptation rule"""
    
    def __init__(self):
        super().__init__("Learning Rate Adaptation")
        self.patience = 3
        self.factor = 0.5
    
    def should_apply(self, current_metrics, history):
        if len(history) < self.patience + 1:
            return False
        
        # Check if recent losses haven't improved
        recent_losses = [m.get('loss', float('inf')) for m in history[-self.patience:]]
        return all(recent_losses[i] >= recent_losses[i-1] for i in range(1, len(recent_losses)))
    
    def get_updates(self, metrics, config):
        current_lr = config['training']['learning_rate']
        new_lr = current_lr * self.factor
        return {
            'training': {
                'learning_rate': max(new_lr, 1e-6)
            }
        }
```

## Configuration Validation

### ✅ Configuration Validator

```python
from typing import List, Tuple
import jsonschema

class ConfigValidator:
    """Configuration validator"""
    
    def __init__(self):
        self.schema = self._load_schema()
        self.custom_validators = self._setup_custom_validators()
    
    def validate(self, config: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate configuration"""
        errors = []
        
        # JSON Schema validation
        try:
            jsonschema.validate(config, self.schema)
        except jsonschema.ValidationError as e:
            errors.append(f"Schema validation error: {e.message}")
        
        # Custom validation
        for validator in self.custom_validators:
            validator_errors = validator.validate(config)
            errors.extend(validator_errors)
        
        return len(errors) == 0, errors
    
    def _load_schema(self) -> Dict[str, Any]:
        """Load configuration schema"""
        return {
            "type": "object",
            "properties": {
                "global": {
                    "type": "object",
                    "properties": {
                        "seed": {"type": "integer", "minimum": 0},
                        "device": {"type": "string"},
                        "deterministic": {"type": "boolean"}
                    },
                    "required": ["seed", "device"]
                },
                "data": {
                    "type": "object",
                    "properties": {
                        "batch_size": {"type": "integer", "minimum": 1},
                        "num_workers": {"type": "integer", "minimum": 0}
                    },
                    "required": ["batch_size"]
                },
                "model": {
                    "type": "object",
                    "properties": {
                        "d_model": {"type": "integer", "minimum": 1},
                        "num_heads": {"type": "integer", "minimum": 1},
                        "num_layers": {"type": "integer", "minimum": 1}
                    },
                    "required": ["d_model", "num_heads"]
                },
                "training": {
                    "type": "object",
                    "properties": {
                        "epochs": {"type": "integer", "minimum": 1},
                        "learning_rate": {"type": "number", "minimum": 0}
                    },
                    "required": ["epochs", "learning_rate"]
                }
            },
            "required": ["global", "data", "model", "training"]
        }
    
    def _setup_custom_validators(self) -> List['CustomValidator']:
        """Setup custom validators"""
        return [
            DeviceValidator(),
            AttentionTypeValidator(),
            DimensionValidator(),
            PathValidator()
        ]

class CustomValidator:
    """Base custom validator class"""
    
    def validate(self, config: Dict[str, Any]) -> List[str]:
        raise NotImplementedError

class DeviceValidator(CustomValidator):
    """Device validator"""
    
    def validate(self, config: Dict[str, Any]) -> List[str]:
        errors = []
        device = config.get('global', {}).get('device', '')
        
        if device.startswith('cuda:'):
            try:
                gpu_id = int(device.split(':')[1])
                if not torch.cuda.is_available():
                    errors.append("CUDA not available but CUDA device configured")
                elif gpu_id >= torch.cuda.device_count():
                    errors.append(f"GPU {gpu_id} does not exist")
            except (ValueError, IndexError):
                errors.append(f"Invalid device format: {device}")
        elif device != 'cpu':
            errors.append(f"Unsupported device type: {device}")
        
        return errors

class AttentionTypeValidator(CustomValidator):
    """Attention type validator"""
    
    def validate(self, config: Dict[str, Any]) -> List[str]:
        errors = []
        attention_type = config.get('model', {}).get('attention_type', '')
        
        valid_types = [
            'self', 'simplified_self', 'muse', 'ufo', 'sparse', 'lsh',
            'relative', 'axial', 'mobilevit', 'mobilevitv2', 'emsa',
            'dat', 'crossformer', 'moa', 'crisscross', 'se', 'sk',
            'cbam', 'bam', 'eca', 'psa', 'danet', 'cot', 'polarized'
        ]
        
        if attention_type not in valid_types:
            errors.append(f"Unsupported attention type: {attention_type}")
        
        return errors
```

## Environment Configuration

### 🌍 Environment Variable Support

```bash
# Environment variable configuration example
export VIV_GLOBAL_DEVICE="cuda:1"
export VIV_DATA_BATCH_SIZE="64"
export VIV_TRAINING_LEARNING_RATE="0.0005"
export VIV_GLOBAL_DEBUG="true"
```

### 🔧 Environment-Specific Configuration

```python
class EnvironmentConfig:
    """Environment-specific configuration management"""
    
    def __init__(self, base_config_path: str):
        self.base_config_path = base_config_path
        self.environment = self._detect_environment()
    
    def load_config(self) -> Dict[str, Any]:
        """Load environment-specific configuration"""
        # Load base configuration
        base_config = self._load_base_config()
        
        # Load environment-specific overrides
        env_overrides = self._load_environment_overrides()
        
        # Merge configurations
        final_config = self._merge_configs(base_config, env_overrides)
        
        return final_config
    
    def _detect_environment(self) -> str:
        """Detect runtime environment"""
        if os.getenv('VIV_ENV'):
            return os.getenv('VIV_ENV')
        elif os.getenv('CUDA_VISIBLE_DEVICES'):
            return 'gpu'
        else:
            return 'cpu'
    
    def _load_environment_overrides(self) -> Dict[str, Any]:
        """Load environment-specific override configuration"""
        env_configs = {
            'development': {
                'global': {'debug': True, 'verbose': True},
                'training': {'epochs': 2},
                'data': {'batch_size': 32}
            },
            'production': {
                'global': {'debug': False, 'verbose': False},
                'training': {'epochs': 50},
                'logging': {'log_level': 'WARNING'}
            },
            'testing': {
                'global': {'deterministic': True},
                'training': {'epochs': 1},
                'data': {'batch_size': 16}
            }
        }
        
        return env_configs.get(self.environment, {})
```

## Best Practices

### 📚 Configuration Management Best Practices

1. **Version Control**
   ```yaml
   # Include version information in config files
   version: "1.2.0"
   config_format_version: "2.0"
   last_modified: "2024-01-01T12:00:00Z"
   ```

2. **Documentation**
   ```yaml
   # Add comments for each configuration item
   training:
     learning_rate: 0.0001  # Initial learning rate, recommended range: [1e-5, 1e-2]
     epochs: 10             # Number of training epochs, adjust based on dataset size
   ```

3. **Default Values**
   ```python
   def get_config_value(config, key_path, default=None):
       """Safely get configuration value with default support"""
       keys = key_path.split('.')
       current = config
       
       for key in keys:
           if isinstance(current, dict) and key in current:
               current = current[key]
           else:
               return default
       
       return current
   ```

4. **Configuration Templates**
   ```python
   class ConfigTemplate:
       """Configuration template generator"""
       
       @staticmethod
       def generate_quick_start():
           """Generate quick start configuration"""
           return {
               'global': {'device': 'cpu', 'seed': 42},
               'data': {'batch_size': 32},
               'model': {'attention_type': 'self', 'd_model': 128},
               'training': {'epochs': 5, 'learning_rate': 0.001}
           }
       
       @staticmethod
       def generate_high_performance():
           """Generate high performance configuration"""
           return {
               'global': {'device': 'cuda:0', 'mixed_precision': True},
               'data': {'batch_size': 256, 'num_workers': 8},
               'model': {'attention_type': 'muse', 'd_model': 512},
               'training': {'epochs': 50, 'learning_rate': 0.0001}
           }
   ```

### 🔍 Configuration Debugging

```python
class ConfigDebugger:
    """Configuration debugging tool"""
    
    @staticmethod
    def print_config_summary(config: Dict[str, Any]):
        """Print configuration summary"""
        print("📋 Configuration Summary:")
        print(f"   Device: {config.get('global', {}).get('device', 'unknown')}")
        print(f"   Batch Size: {config.get('data', {}).get('batch_size', 'unknown')}")
        print(f"   Attention Type: {config.get('model', {}).get('attention_type', 'unknown')}")
        print(f"   Training Epochs: {config.get('training', {}).get('epochs', 'unknown')}")
    
    @staticmethod
    def validate_config_compatibility(config: Dict[str, Any]):
        """Validate configuration compatibility"""
        warnings = []
        
        # Check device and batch size compatibility
        device = config.get('global', {}).get('device', 'cpu')
        batch_size = config.get('data', {}).get('batch_size', 32)
        
        if device == 'cpu' and batch_size > 128:
            warnings.append("Recommend using smaller batch size in CPU mode")
        
        # Check attention type and model dimension
        attention_type = config.get('model', {}).get('attention_type', 'self')
        d_model = config.get('model', {}).get('d_model', 256)
        
        if attention_type in ['muse', 'ufo'] and d_model < 128:
            warnings.append("Efficient attention mechanisms recommend using larger model dimensions")
        
        return warnings
```

---

**💡 Tip**: A good configuration system is key to project success. It not only improves experiment reproducibility but also greatly simplifies the complexity of parameter tuning and experiment management.

---

*Need help? Check the [FAQ](faq) or [Troubleshooting](troubleshooting) pages.*
