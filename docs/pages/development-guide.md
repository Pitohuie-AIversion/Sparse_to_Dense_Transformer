---
layout: default
title: Development Guide
description: Development environment setup and contribution guidelines
nav_order: 19
parent: Development & Maintenance
permalink: /pages/development-guide/
---

# Development Guide

This document provides detailed development guidance for VIVTransformer project developers, including code standards, development workflows, testing guidelines, and contribution methods.

## 📋 Table of Contents {#table-of-contents}

- [Development Environment Setup](#development-environment-setup)
- [Code Standards](#code-standards)
- [Project Structure](#project-structure)
- [Development Workflow](#development-workflow)
- [Testing Guidelines](#testing-guidelines)
- [Documentation Writing](#documentation-writing)
- [Performance Optimization](#performance-optimization)
- [Contribution Guidelines](#contribution-guidelines)
- [Release Process](#release-process)

## Development Environment Setup {#development-environment-setup}

### 🛠️ Environment Preparation {#environment-preparation}

#### 1. Basic Environment {#1-basic-environment}

```bash
# 1. Clone project
git clone https://github.com/your-username/VIVTransformer.git
cd VIVTransformer

# 2. Create virtual environment
conda create -n vivtransformer-dev python=3.9
conda activate vivtransformer-dev

# 3. Install development dependencies
pip install -r requirements-dev.txt

# 4. Install project (development mode)
pip install -e .

# 5. Install pre-commit hooks
pre-commit install
```

#### 2. Development Tools Configuration {#2-development-tools-configuration}

**VS Code Configuration** (`.vscode/settings.json`):
```json
{
    "python.defaultInterpreterPath": "./venv/bin/python",
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": true,
    "python.linting.flake8Enabled": true,
    "python.formatting.provider": "black",
    "python.formatting.blackArgs": ["--line-length=88"],
    "python.sortImports.args": ["--profile", "black"],
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
        "source.organizeImports": true
    },
    "python.testing.pytestEnabled": true,
    "python.testing.pytestArgs": [
        "tests"
    ]
}
```

**PyCharm Configuration**:
- Set code style to Black
- Enable type checking
- Configure test runner to pytest
- Optimize imports

#### 3. Git Configuration {#3-git-configuration}

```bash
# Set user information
git config user.name "Your Name"
git config user.email "your.email@example.com"

# Set default branch
git config init.defaultBranch main

# Set line endings
git config core.autocrlf input  # Linux/Mac
git config core.autocrlf true   # Windows
```

### 📦 Dependency Management {#dependency-management}

#### requirements-dev.txt {#requirements-dev-txt}
```txt
# Basic dependencies
torch>=1.12.0
torchvision>=0.13.0
numpy>=1.21.0
pandas>=1.3.0
scipy>=1.7.0

# Development tools
black>=22.0.0
isort>=5.10.0
flake8>=4.0.0
mypy>=0.950
pylint>=2.13.0
pre-commit>=2.17.0

# Testing tools
pytest>=7.0.0
pytest-cov>=3.0.0
pytest-mock>=3.7.0
pytest-xdist>=2.5.0
hypothesis>=6.40.0

# Documentation tools
sphinx>=4.5.0
sphinx-rtd-theme>=1.0.0
myst-parser>=0.17.0

# Performance analysis
line-profiler>=3.5.0
memory-profiler>=0.60.0
py-spy>=0.3.0

# Visualization
matplotlib>=3.5.0
seaborn>=0.11.0
tensorboard>=2.8.0
wandb>=0.12.0

# Other tools
tqdm>=4.62.0
click>=8.0.0
rich>=12.0.0
typer>=0.4.0
```

## Code Standards {#code-standards}

### 🎨 Code Style {#code-style}

#### 1. Python Code Standards {#1-python-code-standards}

**Basic Principles**:
- Follow PEP 8 standards
- Use Black for code formatting
- Use isort for import sorting
- Limit line length to 88 characters

**Naming Conventions**:
```python
# Class names: PascalCase
class AttentionMechanism:
    pass

# Functions and variables: lowercase with underscores
def calculate_attention_weights():
    attention_scores = None
    return attention_scores

# Constants: uppercase with underscores
MAX_SEQUENCE_LENGTH = 512
DEFAULT_HIDDEN_SIZE = 768

# Private methods: prefix underscore
def _internal_helper_function():
    pass

# Special methods: double underscores
def __init__(self):
    pass
```

**Type Annotations**:
```python
from typing import Dict, List, Optional, Tuple, Union
import torch

def process_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    dropout_rate: float = 0.1
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Process attention computation
    
    Args:
        query: Query tensor [batch_size, seq_len, d_model]
        key: Key tensor [batch_size, seq_len, d_model]
        value: Value tensor [batch_size, seq_len, d_model]
        mask: Optional mask tensor
        dropout_rate: Dropout ratio
    
    Returns:
        Tuple of output tensor and attention weights
    """
    # Implementation code...
    pass
```

#### 2. Documentation String Standards {#2-documentation-string-standards}

**Google Style Docstrings**:
```python
class MultiHeadAttention(torch.nn.Module):
    """Multi-head attention mechanism implementation
    
    This class implements the multi-head attention mechanism in Transformer,
    supporting both self-attention and cross-attention.
    
    Attributes:
        d_model: Model dimension
        num_heads: Number of attention heads
        dropout_rate: Dropout ratio
    
    Example:
        >>> attention = MultiHeadAttention(d_model=512, num_heads=8)
        >>> output, weights = attention(query, key, value)
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout_rate: float = 0.1
    ):
        """Initialize multi-head attention layer
        
        Args:
            d_model: Model dimension, must be divisible by num_heads
            num_heads: Number of attention heads
            dropout_rate: Dropout ratio in range [0, 1]
        
        Raises:
            ValueError: When d_model is not divisible by num_heads
        """
        super().__init__()
        
        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads})")
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Linear transformation layers
        self.w_q = torch.nn.Linear(d_model, d_model)
        self.w_k = torch.nn.Linear(d_model, d_model)
        self.w_v = torch.nn.Linear(d_model, d_model)
        self.w_o = torch.nn.Linear(d_model, d_model)
        
        self.dropout = torch.nn.Dropout(dropout_rate)
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass
        
        Args:
            query: Query tensor [batch_size, seq_len, d_model]
            key: Key tensor [batch_size, seq_len, d_model]
            value: Value tensor [batch_size, seq_len, d_model]
            mask: Optional attention mask [batch_size, seq_len, seq_len]
        
        Returns:
            Tuple containing:
            - output: Output tensor [batch_size, seq_len, d_model]
            - attention_weights: Attention weights [batch_size, num_heads, seq_len, seq_len]
        
        Note:
            All input tensors must be on the same device
        """
        # Implementation code...
        pass
```

#### 3. Error Handling Standards {#3-error-handling-standards}
```python
class VIVTransformerError(Exception):
    """VIVTransformer base exception class"""
    pass

class ConfigurationError(VIVTransformerError):
    """Configuration error exception"""
    pass

class ModelError(VIVTransformerError):
    """Model related error exception"""
    pass

class DataError(VIVTransformerError):
    """Data related error exception"""
    pass

# Usage Example
def load_config(config_path: str) -> Dict:
    """Load configuration file
    
    Args:
        config_path: Configuration file path
    
    Returns:
        Configuration dictionary
    
    Raises:
        ConfigurationError: Configuration file format error or missing required fields
        FileNotFoundError: Configuration file not found
    """
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    except yaml.YAMLError as e:
        raise ConfigurationError(f"Configuration file format error: {e}")
    
    # Validate required fields
    required_fields = ['model', 'training', 'data']
    missing_fields = [field for field in required_fields if field not in config]
    
    if missing_fields:
        raise ConfigurationError(f"Configuration file missing required fields: {missing_fields}")
    
    return config
```

### 🧪 Code Quality Tools {#code-quality-tools}

#### 1. Pre-commit Hook Configuration {#1-pre-commit-hook-configuration}

**.pre-commit-config.yaml**:
```yaml
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.4.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
      - id: check-merge-conflict
      - id: debug-statements

  - repo: https://github.com/psf/black
    rev: 22.12.0
    hooks:
      - id: black
        language_version: python3
        args: [--line-length=88]

  - repo: https://github.com/pycqa/isort
    rev: 5.11.4
    hooks:
      - id: isort
        args: [--profile, black]

  - repo: https://github.com/pycqa/flake8
    rev: 6.0.0
    hooks:
      - id: flake8
        args: [--max-line-length=88, --extend-ignore=E203,W503]

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v0.991
    hooks:
      - id: mypy
        additional_dependencies: [types-PyYAML, types-requests]
```

#### 2. Configuration File {#2-configuration-file}

**setup.cfg**:
```ini
[flake8]
max-line-length = 88
extend-ignore = E203, W503, E501
exclude = 
    .git,
    __pycache__,
    build,
    dist,
    *.egg-info,
    .venv,
    .tox

[mypy]
python_version = 3.9
warn_return_any = True
warn_unused_configs = True
disallow_untyped_defs = True
disallow_incomplete_defs = True
check_untyped_defs = True
disallow_untyped_decorators = True
no_implicit_optional = True
warn_redundant_casts = True
warn_unused_ignores = True
warn_no_return = True
warn_unreachable = True
strict_equality = True

[tool:pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = 
    --strict-markers
    --strict-config
    --verbose
    --cov=vivtransformer
    --cov-report=term-missing
    --cov-report=html
    --cov-report=xml
    --cov-fail-under=80

[coverage:run]
source = vivtransformer
omit = 
    */tests/*
    */test_*
    setup.py
    */venv/*
    */__pycache__/*

[coverage:report]
exclude_lines =
    pragma: no cover
    def __repr__
    raise AssertionError
    raise NotImplementedError
    if __name__ == .__main__.:
```

**pyproject.toml**:
```toml
[tool.black]
line-length = 88
target-version = ['py39']
include = '\.pyi?$'
extend-exclude = '''
(
  /(
      \.eggs
    | \.git
    | \.hg
    | \.mypy_cache
    | \.tox
    | \.venv
    | _build
    | buck-out
    | build
    | dist
  )/
)
'''

[tool.isort]
profile = "black"
line_length = 88
multi_line_output = 3
include_trailing_comma = true
force_grid_wrap = 0
use_parentheses = true
ensure_newline_before_comments = true

[tool.pylint.messages_control]
disable = "C0330, C0326"

[tool.pylint.format]
max-line-length = "88"
```

## Project Structure {#project-structure}

### 📁 Directory Organization {#directory-organization}

```
VIVTransformer/
├── vivtransformer/              # Main source code
│   ├── __init__.py
│   ├── models/                  # Model definitions
│   │   ├── __init__.py
│   │   ├── base.py             # Base model class
│   │   ├── transformer.py      # Transformer implementation
│   │   └── attention/          # Attention mechanisms
│   │       ├── __init__.py
│   │       ├── multi_head.py
│   │       ├── sparse.py
│   │       └── linear.py
│   ├── data/                   # Data processing
│   │   ├── __init__.py
│   │   ├── datasets.py
│   │   ├── loaders.py
│   │   └── transforms.py
│   ├── training/               # Training related
│   │   ├── __init__.py
│   │   ├── trainer.py
│   │   ├── losses.py
│   │   └── optimizers.py
│   ├── utils/                  # Utility functions
│   │   ├── __init__.py
│   │   ├── config.py
│   │   ├── logging.py
│   │   └── metrics.py
│   └── cli/                    # Command line interface
│       ├── __init__.py
│       ├── train.py
│       └── evaluate.py
├── tests/                      # Test code
│   ├── __init__.py
│   ├── conftest.py            # pytest configuration
│   ├── unit/                  # Unit tests
│   │   ├── test_models.py
│   │   ├── test_attention.py
│   │   └── test_data.py
│   ├── integration/           # Integration tests
│   │   ├── test_training.py
│   │   └── test_pipeline.py
│   └── fixtures/              # Test fixtures
│       ├── sample_data.pt
│       └── test_config.yaml
├── docs/                      # Documentation
│   ├── source/
│   │   ├── conf.py
│   │   ├── index.rst
│   │   └── api/
│   └── build/
├── scripts/                   # Scripts
│   ├── setup_env.sh
│   ├── run_tests.sh
│   └── benchmark.py
├── config/                    # Configuration files
│   ├── default_config.yaml
│   ├── training_config.yaml
│   └── model_configs/
├── examples/                  # Example code
│   ├── basic_usage.py
│   ├── custom_attention.py
│   └── fine_tuning.py
├── requirements.txt           # Production dependencies
├── requirements-dev.txt       # Development dependencies
├── setup.py                   # Installation script
├── pyproject.toml            # Project configuration
├── README.md                 # Project description
├── CHANGELOG.md              # Changelog
├── LICENSE                   # License
└── .gitignore               # Git ignore file
```

### 🏗️ Module Architecture {#module-architecture}

#### 1. Single Responsibility Principle {#1-single-responsibility-principle}
Each module should have only one reason to change:

```python
# ❌ Bad: A class takes on multiple responsibilities {#bad-multiple-responsibilities}
class ModelTrainer:
    def __init__(self):
        pass
    
    def load_data(self):  # Data loading responsibility
        pass
    
    def train_model(self):  # Training responsibility
        pass
    
    def save_results(self):  # Result saving responsibility
        pass

# ✅ Good: Separated responsibilities {#good-separated-responsibilities}
class DataLoader:
    def load_data(self):
        pass

class ModelTrainer:
    def train_model(self):
        pass

class ResultSaver:
    def save_results(self):
        pass
```

#### 2. Open-Closed Principle {#2-open-closed-principle}
Open for extension, closed for modification:

```python
# Base attention interface {#base-attention-interface}
from abc import ABC, abstractmethod

class AttentionMechanism(ABC):
    """Base class for attention mechanisms"""
    
    @abstractmethod
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass"""
        pass

# Concrete implementations {#concrete-implementations}
class MultiHeadAttention(AttentionMechanism):
    def forward(self, query, key, value, mask=None):
        # Multi-head attention implementation
        pass

class SparseAttention(AttentionMechanism):
    def forward(self, query, key, value, mask=None):
        # Sparse attention implementation
        pass

# Attention factory {#attention-factory}
class AttentionFactory:
    _registry = {
        'multi_head': MultiHeadAttention,
        'sparse': SparseAttention,
    }
    
    @classmethod
    def create(cls, attention_type: str, **kwargs) -> AttentionMechanism:
        if attention_type not in cls._registry:
            raise ValueError(f"Unknown attention type: {attention_type}")
        
        return cls._registry[attention_type](**kwargs)
    
    @classmethod
    def register(cls, name: str, attention_class: type):
        """Register a new attention mechanism"""
        cls._registry[name] = attention_class

class VIVTransformer(torch.nn.Module):
    """VIVTransformer main model"""
    
    def __init__(
        self,
        attention_mechanism: AttentionMechanism,
        loss_function: torch.nn.Module,
        optimizer_factory: Callable,
        config: Dict[str, Any]
    ):
        super().__init__()
        
        self.attention = attention_mechanism
        self.loss_fn = loss_function
        self.optimizer_factory = optimizer_factory
        self.config = config
        
        # Build model layers
        self._build_layers()
    
    def _build_layers(self):
        """Build model layers"""
        self.embedding = torch.nn.Embedding(
            self.config['vocab_size'],
            self.config['d_model']
        )
        
        self.encoder_layers = torch.nn.ModuleList([
            TransformerEncoderLayer(
                d_model=self.config['d_model'],
                attention=self.attention,
                dropout=self.config['dropout']
            )
            for _ in range(self.config['num_layers'])
        ])
        
        self.output_projection = torch.nn.Linear(
            self.config['d_model'],
            self.config['output_size']
        )

class VIVTransformer(torch.nn.Module):
    """VIVTransformer main model"""
    
    def __init__(
        self,
        attention_mechanism: AttentionMechanism,
        loss_function: torch.nn.Module,
        optimizer_factory: Callable,
        config: Dict[str, Any]
    ):
        super().__init__()
        
        self.attention = attention_mechanism
        self.loss_fn = loss_function
        self.optimizer_factory = optimizer_factory
        self.config = config
        
        # Build model layers
        self._build_layers()
    
    def _build_layers(self):
        """Build model layers"""
        self.embedding = torch.nn.Embedding(
            self.config['vocab_size'],
            self.config['d_model']
        )
        
        self.encoder_layers = torch.nn.ModuleList([
            TransformerEncoderLayer(
                d_model=self.config['d_model'],
                attention=self.attention,
                dropout=self.config['dropout']
            )
            for _ in range(self.config['num_layers'])
        ])
        
        self.output_projection = torch.nn.Linear(
            self.config['d_model'],
            self.config['output_size']
        )

# Integration tests: Test component interactions
class TestVIVTransformerIntegration:
    def test_end_to_end_training(self):
        """Test end-to-end training"""
        config = {
            'd_model': 256,
            'num_heads': 4,
            'num_layers': 2,
            'vocab_size': 1000,
            'output_size': 10
        }
        
        model = VIVTransformer(config)
        optimizer = torch.optim.Adam(model.parameters())
        criterion = torch.nn.CrossEntropyLoss()
        
        # Simulate training data
        batch_size, seq_len = 4, 20
        input_ids = torch.randint(0, config['vocab_size'], (batch_size, seq_len))
        labels = torch.randint(0, config['output_size'], (batch_size,))
        
        # Forward pass
        output = model(input_ids)
        loss = criterion(output, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Verify training step completion
        assert loss.item() > 0
        assert all(p.grad is not None for p in model.parameters() if p.requires_grad)

## Testing Guidelines {#testing-guidelines}

### 🧪 Testing Framework {#testing-framework}

#### 1. pytest Configuration {#1-pytest-configuration}

**conftest.py**:
```python
# conftest.py
import pytest
import torch
import numpy as np
from vivtransformer import VIVTransformer

@pytest.fixture
def device():
    """Test device"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

@pytest.fixture
def sample_config():
    """Sample configuration"""
    return {
        'model': {
            'd_model': 256,
            'num_heads': 8,
            'num_layers': 6,
            'dropout': 0.1
        },
        'training': {
            'batch_size': 4,
            'learning_rate': 1e-4
        }
    }

@pytest.fixture
def sample_data(device):
    """Sample data"""
    batch_size = 4
    seq_len = 128
    d_model = 256
    
    x = torch.randn(batch_size, seq_len, d_model, device=device)
    y = torch.randn(batch_size, seq_len, d_model, device=device)
    
    return x, y

@pytest.fixture(autouse=True)
def set_random_seed():
    """Set random seed to ensure test reproducibility"""
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)

@pytest.fixture
def mock_model(sample_config, device):
    """Mock model"""
    model = VIVTransformer(sample_config['model'])
    model.to(device)
    model.eval()
    return model
```

#### 2. Testing Utility Functions {#2-testing-utility-functions}
```python
# tests/utils.py
import torch
import pytest

def assert_tensor_equal(actual, expected, rtol=1e-5):
    """Assert tensor equality"""
    assert actual.shape == expected.shape, f"Shape mismatch: {actual.shape} vs {expected.shape}"
    assert torch.allclose(actual, expected, rtol=rtol), "Tensor values are not equal"

def assert_tensor_shape(tensor, expected_shape):
    """Assert tensor shape"""
    assert tensor.shape == expected_shape, f"Shape mismatch: {tensor.shape} vs {expected_shape}"

def assert_no_nan_inf(tensor):
    """Assert tensor contains no NaN or Inf"""
    assert not torch.isnan(tensor).any(), "Tensor contains NaN"
    assert not torch.isinf(tensor).any(), "Tensor contains Inf"

def assert_has_gradients(model):
    """Assert model parameters have gradients"""
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Parameter {name} has no gradient"

def count_parameters(model):
    """Count model parameters"""
    return sum(p.numel() for p in model.parameters())

class ModelTestHelper:
    """Model testing helper class"""
    
    def __init__(self, model, device):
        self.model = model
        self.device = device
    
    def test_forward_pass(self, input_data):
        """Test forward pass"""
        self.model.eval()
        with torch.no_grad():
            output = self.model(input_data)
        
        # Check output
        assert_no_nan_inf(output)
        return output
    
    def test_backward_pass(self, input_data, target_data):
        """Test backward pass"""
        self.model.train()
        
        # Forward pass
        output = self.model(input_data)
        
        # Compute loss
        loss = torch.nn.functional.mse_loss(output, target_data)
        
        # Backward pass
        loss.backward()
        
        # Check gradients
        assert_has_gradients(self.model)
        
        return loss
    
    def test_deterministic(self, input_data):
        """Test model determinism"""
        self.model.eval()
        
        outputs = []
        for _ in range(3):
            with torch.no_grad():
                output = self.model(input_data)
            outputs.append(output)
        
        # Check all outputs are the same
        for i in range(1, len(outputs)):
            assert_tensor_equal(outputs[0], outputs[i])
    
    def test_device_consistency(self, input_data):
        """Test model device consistency"""
        # Check model parameter devices
        for name, param in self.model.named_parameters():
            assert param.device == self.device, f"Parameter {name} not on correct device"
        
        # Check input data device
        for key, tensor in input_data.items():
            assert tensor.device == self.device, f"{key} not on correct device"
```

#### 3. Performance Testing {#3-performance-testing}
```python
# tests/test_performance.py
import time
import torch
import pytest
from vivtransformer import VIVTransformer

class TestPerformance:
    """Performance testing"""
    
    @pytest.mark.slow
    def test_inference_speed(self, mock_model, sample_data, device):
        """Test inference speed"""
        model = mock_model
        x, _ = sample_data
        
        # Warmup
        for _ in range(10):
            with torch.no_grad():
                _ = model(x)
        
        # Sync GPU
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        # Measure time
        num_runs = 100
        start_time = time.time()
        
        for _ in range(num_runs):
            with torch.no_grad():
                output = model(x)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        end_time = time.time()
        
        avg_time = (end_time - start_time) / num_runs
        throughput = x.size(0) / avg_time
        
        print(f"Average inference time: {avg_time*1000:.2f}ms")
        print(f"Throughput: {throughput:.2f} samples/second")
        
        # Performance assertion (adjust based on actual situation)
        assert avg_time < 0.1, f"Inference time too long: {avg_time:.4f}s"
    
    @pytest.mark.gpu
    def test_memory_usage(self, sample_config, sample_data, device):
        """Test memory usage"""
        if device.type != 'cuda':
            pytest.skip("GPU memory testing only on CUDA devices")
        
        x, y = sample_data
        
        # Clear memory
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        # Measure memory usage
        model = VIVTransformer(sample_config['model']).to(device)
        
        output = model(x)
        loss = torch.nn.functional.mse_loss(output, y)
        loss.backward()
        
        peak_memory = torch.cuda.max_memory_allocated() / 1024**2  # MB
        current_memory = torch.cuda.memory_allocated() / 1024**2  # MB
        
        print(f"Peak memory usage: {peak_memory:.2f}MB")
        print(f"Current memory usage: {current_memory:.2f}MB")
        
        # Memory assertion (adjust based on actual situation)
        assert peak_memory < 1000, f"Memory usage too high: {peak_memory:.2f}MB"
    
    @pytest.mark.parametrize("batch_size", [1, 4, 8, 16, 32])
    def test_batch_size_scalability(self, mock_model, device, batch_size):
        """Test batch size scalability"""
        model = mock_model
        
        seq_len = 128
        d_model = 256
        
        x = torch.randn(batch_size, seq_len, d_model, device=device)
        
        start_time = time.time()
        with torch.no_grad():
            output = model(x)
        end_time = time.time()
        
        time_per_sample = (end_time - start_time) / batch_size
        
        print(f"Batch size {batch_size}: {time_per_sample*1000:.2f}ms/sample")
        
        # Check output shape
        expected_shape = (batch_size, sample_config['output_size'])
        assert output.shape == expected_shape
```

### 📊 Test Coverage {#test-coverage}
```bash
# Run tests and generate coverage report
pytest --cov=vivtransformer --cov-report=html tests/

# View coverage report
open htmlcov/index.html

# Set coverage threshold
pytest --cov=vivtransformer --cov-fail-under=80 tests/
```

## Performance Optimization {#performance-optimization}

### ⚡ Performance Analysis Tools {#performance-analysis-tools}

#### 1. Code Performance Analysis {#1-code-performance-analysis}
```python
# Using line_profiler
@profile
def compute_attention(query, key, value, mask=None):
    """Attention computation function"""
    # Compute attention scores
    scores = torch.matmul(query, key.transpose(-2, -1))
    
    # Scale
    d_k = query.size(-1)
    scores = scores / math.sqrt(d_k)
    
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    
    # Apply to values
    attention_weights = torch.softmax(scores, dim=-1)
    output = torch.matmul(attention_weights, value)
    
    return output, attention_weights

# Run analysis
kernprof -l -v script_with_attention.py
```

#### 2. Memory Analysis {#2-memory-analysis}
```python
# Using memory_profiler
from memory_profiler import profile

@profile
def training_step(model, data_loader, optimizer):
    """Training step"""
    model.train()
    
    for batch_idx, (data, target) in enumerate(data_loader):
        optimizer.zero_grad()
        
        output = model(data)
        loss = F.mse_loss(output, target)
        
        loss.backward()
        optimizer.step()
        
        if batch_idx % 100 == 0:
            print(f'Batch {batch_idx}, Loss: {loss.item():.6f}')

# Run analysis
python -m memory_profiler training_script.py
```

#### 3. PyTorch Profiler {#3-pytorch-profiler}
```python
import torch
from torch.profiler import profile, record_function, ProfilerActivity

def profile_model(model, input_data):
    """Use PyTorch Profiler to analyze model"""
    
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], 
                record_shapes=True, 
                profile_memory=True,
                with_stack=True) as prof:
        
        with record_function("model_inference"):
            for _ in range(10):
                model(input_data)
        
        with record_function("model_training"):
            optimizer = torch.optim.Adam(model.parameters())
            for _ in range(5):
                optimizer.zero_grad()
                output = model(input_data)
                loss = output.mean()
                loss.backward()
                optimizer.step()
    
    # Print key statistics
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    
    # Export trace for visualization
    prof.export_chrome_trace("trace.json")

# Usage example
model = VIVTransformer(config)
input_data = torch.randn(4, 128, 256)
profile_model(model, input_data)

# View results in TensorBoard
# tensorboard --logdir=./log
```

### 🚀 Optimization Tips {#optimization-tips}

#### 1. Computational Optimization {#1-computational-optimization}
```python
# Use torch.jit.script optimization
@torch.jit.script
def optimized_attention(query, key, value):
    """Optimized attention computation"""
    # Batch matrix multiplication
    scores = torch.bmm(query, key.transpose(1, 2))
    
    # Use in-place operations
    scores.div_(math.sqrt(query.size(-1)))
    
    # Use faster softmax implementation
    attention_weights = torch.softmax(scores, dim=-1)
    output = torch.bmm(attention_weights, value)
    
    return output

# Use mixed precision training
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for epoch in range(num_epochs):
    for batch in data_loader:
        optimizer.zero_grad()
        
        with autocast():
            output = model(batch)
            loss = criterion(output, target)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
```

#### 2. Memory Optimization {#2-memory-optimization}
```python
# Gradient checkpointing
import torch.utils.checkpoint as checkpoint

class MemoryEfficientTransformerLayer(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model)
    
    def forward(self, x):
        # Use gradient checkpointing to save memory
        x = checkpoint.checkpoint(self.attention, x, x, x)
        x = checkpoint.checkpoint(self.ffn, x)
        return x

# Dynamic batch size
class DynamicBatchSampler:
    def __init__(self, dataset, max_tokens=4096):
        self.dataset = dataset
        self.max_tokens = max_tokens
    
    def __iter__(self):
        batch = []
        batch_tokens = 0
        
        for idx in range(len(self.dataset)):
            seq_len = len(self.dataset[idx])
            
            if batch_tokens + seq_len > self.max_tokens and batch:
                yield batch
                batch = []
                batch_tokens = 0
            
            batch.append(idx)
            batch_tokens += seq_len
        
        if batch:
            yield batch
- IDE: VS Code, PyCharm
- Debugging: pdb, ipdb
- Profiling: line_profiler, py-spy
- Visualization: TensorBoard, Weights & Biases
- Collaboration: GitHub, GitLab

---

*Need help? Check [FAQ](faq) or [Troubleshooting](troubleshooting) pages.*
