---
layout: default
title: Tutorials and Examples
lang: en
ref: tutorials-examples
description: Comprehensive tutorials and examples for VIVTransformer usage
keywords: tutorial, examples, VIVTransformer, attention mechanism, configuration, training
---

# 📚 Tutorials and Examples

Welcome to the VIVTransformer tutorials and examples page! This page provides detailed guidance and practical examples to help you quickly master the use of VIVTransformer.

## 📖 Table of Contents

- [🚀 Quick Start Guide](#-quick-start-guide)
  - [Basic Model Usage](#basic-model-usage)
  - [Environment Setup](#environment-setup)
  - [Basic Configuration](#basic-configuration)
- [🧠 Attention Mechanism Usage](#-attention-mechanism-usage)
  - [Multiple Attention Mechanism Comparison](#multiple-attention-mechanism-comparison)
  - [Custom Attention Mechanisms](#custom-attention-mechanisms)
- [📊 Loss Function Configuration](#-loss-function-configuration)
  - [SVD Loss Function Detailed](#svd-loss-function-detailed)
- [⚙️ Advanced Configuration](#-advanced-configuration)
  - [Custom Training Loop](#custom-training-loop)
  - [Multi-GPU Training](#multi-gpu-training)
- [🎯 Practical Examples](#-practical-examples)
  - [Time Series Prediction](#time-series-prediction)
  - [Signal Processing](#signal-processing)
  - [Feature Extraction](#feature-extraction)

---

## 🚀 Quick Start Guide

### Basic Model Usage

```python
# basic_usage.py
import torch
import torch.nn as nn
from typing import Dict, Any
import yaml

# Assuming VIVTransformer is imported from the project
from vivtransformer.models import VIVTransformer
from vivtransformer.config import VIVConfig

def basic_example():
    """Basic VIVTransformer usage example"""
    
    print("🌟 VIVTransformer Basic Example")
    
    # 1. Create model configuration
    config = VIVConfig(
        d_model=512,
        n_heads=8,
        n_layers=6,
        d_ff=2048,
        max_seq_length=1024,
        dropout=0.1
    )
    
    # 2. Initialize model
    model = VIVTransformer(config)
    model.eval()
    
    print(f"Model created: {model.__class__.__name__}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 3. Create test data
    batch_size = 4
    seq_length = 256
    
    # Input data: [batch_size, seq_length, d_model]
    input_data = torch.randn(batch_size, seq_length, config.d_model)
    
    print(f"Input shape: {input_data.shape}")
    
    # 4. Forward pass
    with torch.no_grad():
        output = model(input_data)
    
    print(f"Output shape: {output.shape}")
    
    # 5. Extract attention weights (if supported)
    if hasattr(model, 'get_attention_weights'):
        attention_weights = model.get_attention_weights()
        if attention_weights is not None:
            print(f"Attention weights shape: {attention_weights.shape}")
    
    return model, output

def training_example():
    """Training example"""
    
    print("\n🏋️ VIVTransformer Training Example")
    
    # Model configuration
    config = VIVConfig(
        d_model=256,
        n_heads=8,
        n_layers=4,
        d_ff=1024,
        max_seq_length=512,
        dropout=0.1
    )
    
    # Create model
    model = VIVTransformer(config)
    model.train()
    
    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    
    # Training parameters
    num_epochs = 5
    batch_size = 8
    seq_length = 128
    
    print(f"Training for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        # Generate random training data
        input_data = torch.randn(batch_size, seq_length, config.d_model)
        target_data = torch.randn(batch_size, seq_length, config.d_model)
        
        # Forward pass
        output = model(input_data)
        loss = criterion(output, target_data)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        scheduler.step()
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.6f}, LR: {scheduler.get_last_lr()[0]:.2e}")
    
    print("✅ Training completed!")
    return model

if __name__ == "__main__":
    model, output = basic_example()
    trained_model = training_example()
```

### Environment Setup

```python
# environment_setup.py
import torch
import platform
import subprocess
import sys
from typing import List, Dict, Any

def check_environment() -> Dict[str, Any]:
    """Check and display environment information"""
    
    print("🔍 Environment Check")
    
    env_info = {
        'python_version': platform.python_version(),
        'platform': platform.platform(),
        'pytorch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
        'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
        'gpu_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'gpu_names': []
    }
    
    # Get GPU information
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3  # GB
            env_info['gpu_names'].append(f"{gpu_name} ({gpu_memory:.1f}GB)")
    
    # Display information
    print(f"Python Version: {env_info['python_version']}")
    print(f"Platform: {env_info['platform']}")
    print(f"PyTorch Version: {env_info['pytorch_version']}")
    print(f"CUDA Available: {env_info['cuda_available']}")
    
    if env_info['cuda_available']:
        print(f"CUDA Version: {env_info['cuda_version']}")
        print(f"GPU Count: {env_info['gpu_count']}")
        for i, gpu_name in enumerate(env_info['gpu_names']):
            print(f"  GPU {i}: {gpu_name}")
    
    return env_info

def install_dependencies(packages: List[str] = None):
    """Install required dependencies"""
    
    if packages is None:
        packages = [
            'torch>=1.12.0',
            'numpy>=1.21.0',
            'matplotlib>=3.5.0',
            'seaborn>=0.11.0',
            'pyyaml>=6.0',
            'tqdm>=4.64.0',
            'tensorboard>=2.9.0',
            'scikit-learn>=1.1.0'
        ]
    
    print("📦 Installing dependencies...")
    
    for package in packages:
        try:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
            print(f"✅ {package} installed successfully")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install {package}: {e}")

def setup_development_environment():
    """Set up development environment"""
    
    print("🛠️ Development Environment Setup")
    
    # Check current environment
    env_info = check_environment()
    
    # Install dependencies
    install_dependencies()
    
    # Create directories
    import os
    directories = [
        'data',
        'experiments',
        'logs',
        'checkpoints',
        'results',
        'configs'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"📁 Created directory: {directory}")
    
    # Create basic configuration file
    basic_config = {
        'model': {
            'd_model': 512,
            'n_heads': 8,
            'n_layers': 6,
            'd_ff': 2048,
            'dropout': 0.1,
            'max_seq_length': 1024
        },
        'training': {
            'batch_size': 32,
            'learning_rate': 1e-4,
            'num_epochs': 100,
            'warmup_steps': 1000,
            'gradient_clip_norm': 1.0
        },
        'data': {
            'train_path': 'data/train',
            'val_path': 'data/val',
            'test_path': 'data/test'
        }
    }
    
    import yaml
    with open('configs/basic_config.yaml', 'w') as f:
        yaml.dump(basic_config, f, default_flow_style=False)
    
    print("📄 Created basic configuration file: configs/basic_config.yaml")
    print("✅ Development environment setup completed!")

if __name__ == "__main__":
    setup_development_environment()
```

### Basic Configuration

```yaml
# basic_config.yaml

# Model configuration
model:
  name: "VIVTransformer"
  d_model: 512
  n_heads: 8
  n_layers: 6
  d_ff: 2048
  dropout: 0.1
  max_seq_length: 1024
  
  # Attention mechanism configuration
  attention:
    type: "scaled_dot_product"
    use_relative_position: true
    max_relative_position: 32
    dropout: 0.1

# Training configuration
training:
  batch_size: 32
  learning_rate: 1e-4
  num_epochs: 100
  warmup_steps: 1000
  gradient_clip_norm: 1.0
  
  # Optimizer
  optimizer:
    type: "adamw"
    weight_decay: 0.01
    betas: [0.9, 0.999]
    
  # Learning rate scheduler
  scheduler:
    type: "cosine_annealing"
    T_max: 100
    eta_min: 1e-6

# Loss function configuration
loss:
  primary:
    type: "svd_loss"
    alpha: 0.5
    beta: 0.3
    gamma: 0.2
  
  auxiliary:
    - type: "mse_loss"
      weight: 0.1
    - type: "l1_loss"
      weight: 0.05

# Data configuration
data:
  train_path: "data/train"
  val_path: "data/val"
  test_path: "data/test"
  
  # Data preprocessing
  preprocessing:
    normalize: true
    standardize: true
    augmentation:
      noise_std: 0.01
      time_shift_max: 5
      amplitude_scale_range: [0.9, 1.1]

# Experiment configuration
experiment:
  name: "basic_experiment"
  save_dir: "experiments"
  log_interval: 100
  save_interval: 1000
  
  # Monitoring metrics
  metrics:
    - "mse"
    - "mae"
    - "r2_score"
    - "attention_entropy"

# Hardware configuration
hardware:
  device: "auto"  # auto, cpu, cuda, cuda:0
  mixed_precision: true
  compile_model: false  # PyTorch 2.0+
```

### Python Configuration Class

```python
# config_example.py
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union
import yaml
from pathlib import Path

@dataclass
class AttentionConfig:
    """Attention mechanism configuration"""
    type: str = "scaled_dot_product"
    use_relative_position: bool = True
    max_relative_position: int = 32
    dropout: float = 0.1

@dataclass
class ModelConfig:
    """Model configuration"""
    name: str = "VIVTransformer"
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 2048
    dropout: float = 0.1
    max_seq_length: int = 1024
    attention: AttentionConfig = field(default_factory=AttentionConfig)
    
    def __post_init__(self):
        """Configuration validation"""
        if self.d_model % self.n_heads != 0:
            raise ValueError(f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})")
        
        if self.dropout < 0 or self.dropout > 1:
            raise ValueError(f"dropout must be in [0, 1] range, current value: {self.dropout}")

@dataclass
class TrainingConfig:
    """Training configuration"""
    batch_size: int = 32
    learning_rate: float = 1e-4
    num_epochs: int = 100
    warmup_steps: int = 1000
    gradient_clip_norm: float = 1.0
    
    optimizer_type: str = "adamw"
    weight_decay: float = 0.01
    
    scheduler_type: str = "cosine_annealing"
    scheduler_params: Dict = field(default_factory=lambda: {"T_max": 100, "eta_min": 1e-6})

@dataclass
class LossConfig:
    """Loss function configuration"""
    primary_type: str = "svd_loss"
    primary_params: Dict = field(default_factory=lambda: {"alpha": 0.5, "beta": 0.3, "gamma": 0.2})
    auxiliary_losses: List[Dict] = field(default_factory=list)

@dataclass
class VIVConfig:
    """VIVTransformer complete configuration"""
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    
    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path]) -> 'VIVConfig':
        """Load configuration from YAML file"""
        with open(yaml_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_dict(cls, config_dict: Dict) -> 'VIVConfig':
        """Create configuration from dictionary"""
        model_config = ModelConfig(**config_dict.get('model', {}))
        training_config = TrainingConfig(**config_dict.get('training', {}))
        loss_config = LossConfig(**config_dict.get('loss', {}))
        
        return cls(
            model=model_config,
            training=training_config,
            loss=loss_config
        )
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'model': self.model.__dict__,
            'training': self.training.__dict__,
            'loss': self.loss.__dict__
        }
    
    def save_yaml(self, yaml_path: Union[str, Path]):
        """Save as YAML file"""
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, allow_unicode=True)

# Usage example
if __name__ == "__main__":
    # 1. Create default configuration
    config = VIVConfig()
    print("Default configuration:")
    print(config)
    
    # 2. Load configuration from YAML
    # config = VIVConfig.from_yaml('config/basic_config.yaml')
    
    # 3. Custom configuration
    custom_config = VIVConfig(
        model=ModelConfig(
            d_model=256,
            n_heads=4,
            n_layers=4
        ),
        training=TrainingConfig(
            batch_size=64,
            learning_rate=2e-4
        )
    )
    
    print("\nCustom configuration:")
    print(custom_config)
    
    # 4. Save configuration
    custom_config.save_yaml('config/custom_config.yaml')
    print("\nConfiguration saved to config/custom_config.yaml")
```

---

## 🧠 Attention Mechanism Usage

### Multiple Attention Mechanism Comparison

```python
# attention_comparison.py
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, List, Tuple
import time

# Assuming these are attention mechanisms from the project
from vivtransformer.attention import (
    ScaledDotProductAttention,
    ExternalAttention,
    SEAttention,
    CBAMAttention,
    ECAAttention,
    CoordinateAttention
)

class AttentionBenchmark:
    """Attention mechanism benchmark testing"""
    
    def __init__(self, d_model: int = 512, seq_length: int = 128, batch_size: int = 32):
        self.d_model = d_model
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create test data
        self.test_data = torch.randn(batch_size, seq_length, d_model).to(self.device)
        
        # Attention mechanism dictionary
        self.attention_modules = {
            'Scaled Dot-Product': ScaledDotProductAttention(d_model, n_heads=8),
            'External Attention': ExternalAttention(d_model),
            'SE Attention': SEAttention(d_model),
            'CBAM': CBAMAttention(d_model),
            'ECA': ECAAttention(d_model),
            'Coordinate Attention': CoordinateAttention(d_model)
        }
        
        # Move to device
        for name, module in self.attention_modules.items():
            module.to(self.device)
    
    def benchmark_performance(self, num_runs: int = 100) -> Dict[str, Dict[str, float]]:
        """Performance benchmark testing"""
        
        results = {}
        
        for name, module in self.attention_modules.items():
            print(f"Testing {name}...")
            
            # Warmup
            for _ in range(10):
                with torch.no_grad():
                    _ = module(self.test_data)
            
            # Timing test
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            start_time = time.time()
            
            for _ in range(num_runs):
                with torch.no_grad():
                    output = module(self.test_data)
            
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            end_time = time.time()
            
            # Calculate metrics
            avg_time = (end_time - start_time) / num_runs * 1000  # ms
            throughput = (self.batch_size * num_runs) / (end_time - start_time)  # samples/sec
            
            # Memory usage (approximate)
            param_count = sum(p.numel() for p in module.parameters())
            memory_mb = param_count * 4 / (1024 * 1024)  # Assuming float32
            
            results[name] = {
                'avg_time_ms': avg_time,
                'throughput_samples_per_sec': throughput,
                'parameters': param_count,
                'memory_mb': memory_mb
            }
        
        return results
    
    def analyze_attention_patterns(self) -> Dict[str, torch.Tensor]:
        """Analyze attention patterns"""
        
        attention_weights = {}
        
        for name, module in self.attention_modules.items():
            if hasattr(module, 'get_attention_weights'):
                with torch.no_grad():
                    output = module(self.test_data)
                    weights = module.get_attention_weights()
                    if weights is not None:
                        attention_weights[name] = weights.cpu()
        
        return attention_weights
    
    def visualize_results(self, results: Dict[str, Dict[str, float]]):
        """Visualize results"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Average inference time
        names = list(results.keys())
        times = [results[name]['avg_time_ms'] for name in names]
        
        axes[0, 0].bar(names, times, color='skyblue')
        axes[0, 0].set_title('Average Inference Time (ms)')
        axes[0, 0].set_ylabel('Time (ms)')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. Throughput
        throughputs = [results[name]['throughput_samples_per_sec'] for name in names]
        
        axes[0, 1].bar(names, throughputs, color='lightgreen')
        axes[0, 1].set_title('Throughput (samples/sec)')
        axes[0, 1].set_ylabel('Samples/sec')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. Parameter count
        param_counts = [results[name]['parameters'] / 1000 for name in names]  # K parameters
        
        axes[1, 0].bar(names, param_counts, color='orange')
        axes[1, 0].set_title('Parameter Count (K)')
        axes[1, 0].set_ylabel('Parameters (thousands)')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 4. Memory usage
        memory_usage = [results[name]['memory_mb'] for name in names]
        
        axes[1, 1].bar(names, memory_usage, color='salmon')
        axes[1, 1].set_title('Memory Usage (MB)')
        axes[1, 1].set_ylabel('Memory (MB)')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('attention_benchmark_results.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def visualize_attention_weights(self, attention_weights: Dict[str, torch.Tensor]):
        """Visualize attention weights"""
        
        num_attentions = len(attention_weights)
        if num_attentions == 0:
            print("No available attention weights")
            return
        
        fig, axes = plt.subplots(1, num_attentions, figsize=(5 * num_attentions, 5))
        if num_attentions == 1:
            axes = [axes]
        
        for idx, (name, weights) in enumerate(attention_weights.items()):
            # Take attention weights of first sample's first head
            if weights.dim() == 4:  # [batch, heads, seq, seq]
                attn_map = weights[0, 0].numpy()
            elif weights.dim() == 3:  # [batch, seq, seq]
                attn_map = weights[0].numpy()
            else:
                continue
            
            # Only display first 32x32 region (if sequence is too long)
            display_size = min(32, attn_map.shape[0])
            attn_map = attn_map[:display_size, :display_size]
            
            im = axes[idx].imshow(attn_map, cmap='Blues', aspect='auto')
            axes[idx].set_title(f'{name}\nAttention Weights')
            axes[idx].set_xlabel('Key Position')
            axes[idx].set_ylabel('Query Position')
            
            # Add colorbar
            plt.colorbar(im, ax=axes[idx])
        
        plt.tight_layout()
        plt.savefig('attention_weights_visualization.png', dpi=300, bbox_inches='tight')
        plt.show()

# Usage example
def run_attention_comparison():
    """Run attention mechanism comparison"""
    
    print("🧠 Attention Mechanism Comparison Analysis")
    
    # Create benchmark test
    benchmark = AttentionBenchmark(d_model=512, seq_length=128, batch_size=32)
    
    # Performance testing
    print("\n📊 Performance benchmark testing...")
    results = benchmark.benchmark_performance(num_runs=50)
    
    # Print results
    print("\n📈 Performance test results:")
    print("-" * 80)
    print(f"{'Attention Mechanism':<20} {'Time(ms)':<12} {'Throughput':<15} {'Parameters':<12} {'Memory(MB)':<10}")
    print("-" * 80)
    
    for name, metrics in results.items():
        print(f"{name:<20} {metrics['avg_time_ms']:<12.2f} "
              f"{metrics['throughput_samples_per_sec']:<15.1f} "
              f"{metrics['parameters']:<12,} {metrics['memory_mb']:<10.2f}")
    
    # Visualize performance results
    benchmark.visualize_results(results)
    
    # Attention pattern analysis
    print("\n🔍 Attention pattern analysis...")
    attention_weights = benchmark.analyze_attention_patterns()
    
    if attention_weights:
        benchmark.visualize_attention_weights(attention_weights)
        print(f"✅ Analyzed {len(attention_weights)} attention mechanism weight patterns")
    else:
        print("⚠️ No available attention weights for analysis")
    
    print("\n🎉 Attention mechanism comparison completed!")
    return results, attention_weights

if __name__ == "__main__":
    results, attention_weights = run_attention_comparison()
```

### Custom Attention Mechanisms

```python
# custom_attention.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

class AdaptiveAttention(nn.Module):
    """Adaptive Attention Mechanism
    
    Dynamically adjusts attention computation based on input
    """
    
    def __init__(self, 
                 d_model: int, 
                 n_heads: int = 8, 
                 dropout: float = 0.1,
                 adaptive_threshold: float = 0.5):
        super().__init__()
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.adaptive_threshold = adaptive_threshold
        
        # Linear transformation layers
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model)
        
        # Adaptive gating
        self.adaptive_gate = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.ReLU(),
            nn.Linear(d_model // 4, 1),
            nn.Sigmoid()
        )
        
        # Local attention convolution
        self.local_conv = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, groups=d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
        
        # Store attention weights for visualization
        self.attention_weights = None
    
    def forward(self, 
                query: torch.Tensor, 
                key: torch.Tensor, 
                value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass
        
        Args:
            query: [batch_size, seq_len, d_model]
            key: [batch_size, seq_len, d_model]
            value: [batch_size, seq_len, d_model]
            mask: [batch_size, seq_len, seq_len] or None
            
        Returns:
            output: [batch_size, seq_len, d_model]
        """
        
        batch_size, seq_len, d_model = query.shape
        
        # Calculate adaptive weights
        adaptive_weights = self.adaptive_gate(query.mean(dim=1))  # [batch_size, 1]
        
        # Global attention
        global_output = self._global_attention(query, key, value, mask)
        
        # Local attention
        local_output = self._local_attention(query)
        
        # Adaptive fusion
        output = adaptive_weights.unsqueeze(1) * global_output + \
                (1 - adaptive_weights.unsqueeze(1)) * local_output
        
        return output
    
    def _global_attention(self, 
                         query: torch.Tensor, 
                         key: torch.Tensor, 
                         value: torch.Tensor,
                         mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Global attention computation"""
        
        batch_size, seq_len, d_model = query.shape
        
        # Linear transformations
        Q = self.w_q(query).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Calculate attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        # Apply mask
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1) == 0, -1e9)
        
        # Softmax normalization
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Save attention weights
        self.attention_weights = attention_weights.detach()
        
        # Apply attention
        context = torch.matmul(attention_weights, V)
        
        # Reshape output
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len, d_model
        )
        
        return self.w_o(context)
    
    def _local_attention(self, x: torch.Tensor) -> torch.Tensor:
        """Local attention computation"""
        
        # Transpose for convolution [batch_size, d_model, seq_len]
        x_conv = x.transpose(1, 2)
        
        # Local convolution
        local_features = self.local_conv(x_conv)
        
        # Transpose back to original shape
        local_output = local_features.transpose(1, 2)
        
        return local_output
    
    def get_attention_weights(self) -> Optional[torch.Tensor]:
        """Get attention weights"""
        return self.attention_weights

class HierarchicalAttention(nn.Module):
    """Hierarchical Attention Mechanism
    
    Computes attention at different hierarchical levels
    """
    
    def __init__(self, 
                 d_model: int, 
                 n_heads: int = 8, 
                 n_levels: int = 3,
                 dropout: float = 0.1):
        super().__init__()
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_levels = n_levels
        self.d_k = d_model // n_heads
        
        # Multi-level attention
        self.level_attentions = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=n_heads,
                dropout=dropout,
                batch_first=True
            ) for _ in range(n_levels)
        ])
        
        # Hierarchical fusion weights
        self.level_weights = nn.Parameter(torch.ones(n_levels) / n_levels)
        
        # Downsampling and upsampling
        self.downsample = nn.ModuleList([
            nn.Conv1d(d_model, d_model, kernel_size=2**i, stride=2**i)
            for i in range(1, n_levels)
        ])
        
        self.upsample = nn.ModuleList([
            nn.ConvTranspose1d(d_model, d_model, kernel_size=2**i, stride=2**i)
            for i in range(1, n_levels)
        ])
    
    def forward(self, 
                query: torch.Tensor, 
                key: torch.Tensor, 
                value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass"""
        
        batch_size, seq_len, d_model = query.shape
        level_outputs = []
        
        # Level 0: Original resolution
        output_0, _ = self.level_attentions[0](query, key, value, attn_mask=mask)
        level_outputs.append(output_0)
        
        # Other levels: Different resolutions
        current_q, current_k, current_v = query, key, value
        
        for level in range(1, self.n_levels):
            # Downsample
            current_q = self._downsample_sequence(current_q, level - 1)
            current_k = self._downsample_sequence(current_k, level - 1)
            current_v = self._downsample_sequence(current_v, level - 1)
            
            # Attention computation
            level_output, _ = self.level_attentions[level](current_q, current_k, current_v)
            
            # Upsample back to original resolution
            level_output = self._upsample_sequence(level_output, level - 1, seq_len)
            level_outputs.append(level_output)
        
        # Weighted fusion
        weights = F.softmax(self.level_weights, dim=0)
        final_output = sum(w * output for w, output in zip(weights, level_outputs))
        
        return final_output
    
    def _downsample_sequence(self, x: torch.Tensor, level: int) -> torch.Tensor:
        """Downsample sequence"""
        # [batch_size, seq_len, d_model] -> [batch_size, d_model, seq_len]
        x = x.transpose(1, 2)
        x = self.downsample[level](x)
        # [batch_size, d_model, new_seq_len] -> [batch_size, new_seq_len, d_model]
        x = x.transpose(1, 2)
        return x
    
    def _upsample_sequence(self, x: torch.Tensor, level: int, target_len: int) -> torch.Tensor:
        """Upsample sequence"""
        # [batch_size, seq_len, d_model] -> [batch_size, d_model, seq_len]
        x = x.transpose(1, 2)
        x = self.upsample[level](x)
        
        # Adjust to target length
        current_len = x.shape[-1]
        if current_len != target_len:
            x = F.interpolate(x, size=target_len, mode='linear', align_corners=False)
        
        # [batch_size, d_model, target_len] -> [batch_size, target_len, d_model]
        x = x.transpose(1, 2)
        return x

# Usage example
def test_custom_attention():
    """Test custom attention mechanisms"""
    
    print("🧠 Testing Custom Attention Mechanisms")
    
    # Create test data
    batch_size, seq_len, d_model = 4, 64, 256
    x = torch.randn(batch_size, seq_len, d_model)
    
    # Test adaptive attention
    print("\n🔄 Testing Adaptive Attention...")
    adaptive_attn = AdaptiveAttention(d_model, n_heads=8)
    adaptive_output = adaptive_attn(x, x, x)
    print(f"Input shape: {x.shape}")
    print(f"Adaptive attention output shape: {adaptive_output.shape}")
    
    # Get attention weights
    attn_weights = adaptive_attn.get_attention_weights()
    if attn_weights is not None:
        print(f"Attention weights shape: {attn_weights.shape}")
    
    # Test hierarchical attention
    print("\n🏗️ Testing Hierarchical Attention...")
    hierarchical_attn = HierarchicalAttention(d_model, n_heads=8, n_levels=3)
    hierarchical_output = hierarchical_attn(x, x, x)
    print(f"Hierarchical attention output shape: {hierarchical_output.shape}")
    
    # Parameter statistics
    adaptive_params = sum(p.numel() for p in adaptive_attn.parameters())
    hierarchical_params = sum(p.numel() for p in hierarchical_attn.parameters())
    
    print(f"\n📊 Parameter Statistics:")
    print(f"Adaptive attention parameter count: {adaptive_params:,}")
    print(f"Hierarchical attention parameter count: {hierarchical_params:,}")
    
    print("\n✅ Custom attention mechanism testing completed!")

if __name__ == "__main__":
    test_custom_attention()
```

---

## 📊 Loss Function Configuration

### SVD Loss Function Detailed

```python
# svd_loss_tutorial.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional, Dict, Any

class SVDLoss(nn.Module):
    """SVD Loss Function
    
    Loss function based on Singular Value Decomposition for maintaining low-rank structure
    """
    
    def __init__(self, 
                 alpha: float = 0.5, 
                 beta: float = 0.3, 
                 gamma: float = 0.2,
                 rank_penalty: bool = True,
                 adaptive_weights: bool = True):
        """
        Args:
            alpha: Reconstruction loss weight
            beta: Singular value loss weight
            gamma: Orthogonality loss weight
            rank_penalty: Whether to use rank penalty
            adaptive_weights: Whether to use adaptive weights
        """
        super().__init__()
        
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.rank_penalty = rank_penalty
        self.adaptive_weights = adaptive_weights
        
        # Adaptive weight parameters
        if adaptive_weights:
            self.weight_net = nn.Sequential(
                nn.Linear(3, 16),  # 3 loss components
                nn.ReLU(),
                nn.Linear(16, 3),
                nn.Softmax(dim=-1)
            )
    
    def forward(self, 
                pred: torch.Tensor, 
                target: torch.Tensor,
                return_components: bool = False) -> torch.Tensor:
        """
        Compute SVD loss
        
        Args:
            pred: Predictions [batch_size, seq_len, features]
            target: Targets [batch_size, seq_len, features]
            return_components: Whether to return loss components
            
        Returns:
            loss: Total loss
            components (optional): Loss component dictionary
        """
        
        # 1. Reconstruction loss (MSE)
        reconstruction_loss = F.mse_loss(pred, target)
        
        # 2. SVD decomposition
        pred_svd = self._compute_svd_loss(pred, target)
        target_svd = self._compute_svd_loss(target, target)
        
        # 3. Singular value loss
        singular_value_loss = F.mse_loss(pred_svd['singular_values'], target_svd['singular_values'])
        
        # 4. Orthogonality loss
        orthogonality_loss = self._compute_orthogonality_loss(pred_svd['U'], pred_svd['V'])
        
        # 5. Rank penalty (optional)
        rank_loss = 0.0
        if self.rank_penalty:
            rank_loss = self._compute_rank_penalty(pred_svd['singular_values'])
        
        # Loss components
        components = {
            'reconstruction': reconstruction_loss,
            'singular_value': singular_value_loss,
            'orthogonality': orthogonality_loss,
            'rank_penalty': rank_loss
        }
        
        # Calculate weights
        if self.adaptive_weights:
            # Use neural network to adaptively adjust weights
            loss_values = torch.stack([
                reconstruction_loss.detach(),
                singular_value_loss.detach(),
                orthogonality_loss.detach()
            ])
            weights = self.weight_net(loss_values.unsqueeze(0)).squeeze(0)
            alpha, beta, gamma = weights[0], weights[1], weights[2]
        else:
            alpha, beta, gamma = self.alpha, self.beta, self.gamma
        
        # Total loss
        total_loss = (
            alpha * reconstruction_loss +
            beta * singular_value_loss +
            gamma * orthogonality_loss +
            0.1 * rank_loss  # Fixed weight for rank penalty
        )
        
        if return_components:
            components['weights'] = {'alpha': alpha, 'beta': beta, 'gamma': gamma}
            components['total'] = total_loss
            return total_loss, components
        
        return total_loss
    
    def _compute_svd_loss(self, x: torch.Tensor, reference: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute SVD decomposition"""
        
        batch_size, seq_len, features = x.shape
        
        # Reshape to matrix form
        x_matrix = x.view(batch_size, seq_len * features)
        
        # SVD decomposition
        try:
            U, S, V = torch.svd(x_matrix)
        except RuntimeError:
            # If SVD fails, use backup method
            U, S, V = torch.svd(x_matrix + 1e-8 * torch.randn_like(x_matrix))
        
        return {
            'U': U,
            'singular_values': S,
            'V': V
        }
    
    def _compute_orthogonality_loss(self, U: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        """Compute orthogonality loss"""
        
        # U orthogonality
        U_orth = torch.matmul(U.transpose(-2, -1), U)
        I_U = torch.eye(U_orth.shape[-1], device=U.device, dtype=U.dtype)
        U_loss = F.mse_loss(U_orth, I_U.expand_as(U_orth))
        
        # V orthogonality
        V_orth = torch.matmul(V.transpose(-2, -1), V)
        I_V = torch.eye(V_orth.shape[-1], device=V.device, dtype=V.dtype)
        V_loss = F.mse_loss(V_orth, I_V.expand_as(V_orth))
        
        return (U_loss + V_loss) / 2
    
    def _compute_rank_penalty(self, singular_values: torch.Tensor) -> torch.Tensor:
        """Compute rank penalty"""
        
        # Use L1 norm of singular values as rank approximation
        rank_penalty = torch.sum(singular_values, dim=-1).mean()
        
        return rank_penalty

class CompositeLoss(nn.Module):
    """Composite Loss Function
    
    Combines multiple loss functions
    """
    
    def __init__(self, loss_configs: Dict[str, Dict[str, Any]]):
        """
        Args:
            loss_configs: Loss function configuration dictionary
                Example: {
                    'svd': {'type': 'SVDLoss', 'weight': 0.5, 'params': {...}},
                    'mse': {'type': 'MSELoss', 'weight': 0.3, 'params': {}},
                    'l1': {'type': 'L1Loss', 'weight': 0.2, 'params': {}}
                }
        """
        super().__init__()
        
        self.loss_functions = nn.ModuleDict()
        self.loss_weights = {}
        
        for name, config in loss_configs.items():
            loss_type = config['type']
            weight = config.get('weight', 1.0)
            params = config.get('params', {})
            
            # Create loss function
            if loss_type == 'SVDLoss':
                loss_fn = SVDLoss(**params)
            elif loss_type == 'MSELoss':
                loss_fn = nn.MSELoss(**params)
            elif loss_type == 'L1Loss':
                loss_fn = nn.L1Loss(**params)
            elif loss_type == 'SmoothL1Loss':
                loss_fn = nn.SmoothL1Loss(**params)
            elif loss_type == 'HuberLoss':
                loss_fn = nn.HuberLoss(**params)
            else:
                raise ValueError(f"Unsupported loss function type: {loss_type}")
            
            self.loss_functions[name] = loss_fn
            self.loss_weights[name] = weight
    
    def forward(self, 
                pred: torch.Tensor, 
                target: torch.Tensor,
                return_components: bool = False) -> torch.Tensor:
        """Compute composite loss"""
        
        total_loss = 0.0
        components = {}
        
        for name, loss_fn in self.loss_functions.items():
            weight = self.loss_weights[name]
            
            if isinstance(loss_fn, SVDLoss):
                loss_value, loss_components = loss_fn(pred, target, return_components=True)
                components[name] = loss_components
            else:
                loss_value = loss_fn(pred, target)
                components[name] = {'total': loss_value}
            
            total_loss += weight * loss_value
        
        if return_components:
            components['total'] = total_loss
            return total_loss, components
        
        return total_loss

def demonstrate_svd_loss():
    """Demonstrate SVD loss function"""
    
    print("📊 SVD Loss Function Demonstration")
    
    # Create test data
    batch_size, seq_len, features = 8, 64, 32
    
    # Create low-rank target data
    rank = 10
    U_true = torch.randn(batch_size, seq_len * features, rank)
    S_true = torch.abs(torch.randn(batch_size, rank)) + 0.1
    V_true = torch.randn(batch_size, rank, seq_len * features)
    
    target_matrix = torch.bmm(torch.bmm(U_true, torch.diag_embed(S_true)), V_true)
    target = target_matrix.view(batch_size, seq_len, features)
    
    # Create prediction data (add noise)
    noise = 0.1 * torch.randn_like(target)
    pred = target + noise
    
    print(f"Data shape: {target.shape}")
    print(f"True rank of target data: {rank}")
    
    # Test different loss functions
    loss_configs = {
        'svd_adaptive': {
            'type': 'SVDLoss',
            'weight': 1.0,
            'params': {
                'alpha': 0.5,
                'beta': 0.3,
                'gamma': 0.2,
                'adaptive_weights': True
            }
        },
        'svd_fixed': {
            'type': 'SVDLoss',
            'weight': 1.0,
            'params': {
                'alpha': 0.5,
                'beta': 0.3,
                'gamma': 0.2,
                'adaptive_weights': False
            }
        },
        'mse': {
            'type': 'MSELoss',
            'weight': 1.0,
            'params': {}
        }
    }
    
    results = {}
    
    for name, config in loss_configs.items():
        print(f"\nTesting {name}...")
        
        if config['type'] == 'SVDLoss':
            loss_fn = SVDLoss(**config['params'])
            loss_value, components = loss_fn(pred, target, return_components=True)
            
            print(f"  Total loss: {loss_value.item():.6f}")
            print(f"  Reconstruction loss: {components['reconstruction'].item():.6f}")
            print(f"  Singular value loss: {components['singular_value'].item():.6f}")
            print(f"  Orthogonality loss: {components['orthogonality'].item():.6f}")
            
            if 'weights' in components:
                weights = components['weights']
                print(f"  Adaptive weights: α={weights['alpha']:.3f}, β={weights['beta']:.3f}, γ={weights['gamma']:.3f}")
        
        else:
            loss_fn = nn.MSELoss()
            loss_value = loss_fn(pred, target)
            print(f"  Loss value: {loss_value.item():.6f}")
        
        results[name] = loss_value.item()
    
    # Visualize results
    plt.figure(figsize=(10, 6))
    
    names = list(results.keys())
    values = list(results.values())
    
    plt.bar(names, values, color=['skyblue', 'lightgreen', 'salmon'])
    plt.title('Loss Value Comparison for Different Loss Functions')
    plt.ylabel('Loss Value')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('loss_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n✅ SVD loss function demonstration completed!")

def loss_function_tutorial():
    """Loss function usage tutorial"""
    
    print("📚 Loss Function Usage Tutorial")
    
    # 1. Basic SVD loss
    print("\n1️⃣ Basic SVD Loss Usage")
    
    svd_loss = SVDLoss(alpha=0.5, beta=0.3, gamma=0.2)
    
    # Example data
    pred = torch.randn(4, 32, 16)
    target = torch.randn(4, 32, 16)
    
    loss = svd_loss(pred, target)
    print(f"SVD loss value: {loss.item():.6f}")
    
    # 2. Composite loss
    print("\n2️⃣ Composite Loss Usage")
    
    composite_config = {
        'svd': {
            'type': 'SVDLoss',
            'weight': 0.6,
            'params': {'alpha': 0.5, 'beta': 0.3, 'gamma': 0.2}
        },
        'mse': {
            'type': 'MSELoss',
            'weight': 0.3,
            'params': {}
        },
        'l1': {
            'type': 'L1Loss',
            'weight': 0.1,
            'params': {}
        }
    }
    
    composite_loss = CompositeLoss(composite_config)
    total_loss, components = composite_loss(pred, target, return_components=True)
    
    print(f"Composite loss total value: {total_loss.item():.6f}")
    print("Component losses:")
    for name, component in components.items():
        if name != 'total':
            if isinstance(component, dict) and 'total' in component:
                print(f"  {name}: {component['total'].item():.6f}")
    
    # 3. Usage in training
    print("\n3️⃣ Usage Example in Training")
    
    # Create simple model
    model = nn.Linear(16, 16)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Training loop example
    for epoch in range(5):
        optimizer.zero_grad()
        
        output = model(pred.view(-1, 16)).view(4, 32, 16)
        loss = svd_loss(output, target)
        
        loss.backward()
        optimizer.step()
        
        print(f"Epoch {epoch+1}, Loss: {loss.item():.6f}")
    
    print("\n✅ Loss function tutorial completed!")

if __name__ == "__main__":
    demonstrate_svd_loss()
    print("\n" + "="*50 + "\n")
    loss_function_tutorial()
```

---

*This tutorials and examples page provides detailed usage guidance and code examples. For more advanced features, please refer to the [API Documentation]({{ site.baseurl }}/pages/api-reference) and [Best Practices]({{ site.baseurl }}/pages/best-practices).*