# 💡 Best Practices Guide

> Collection of best practices for VIVTransformer development, training and deployment

---

## 📋 Table of Contents

- [🎯 Development Best Practices](#-development-best-practices)
- [🧠 Model Design Principles](#-model-design-principles)
- [📊 Data Processing Best Practices](#-data-processing-best-practices)
- [🏋️ Training Strategy Optimization](#️-training-strategy-optimization)
- [🔧 Code Quality Assurance](#-code-quality-assurance)
- [🚀 Performance Optimization Tips](#-performance-optimization-tips)
- [🔒 Security Best Practices](#-security-best-practices)
- [📈 Monitoring and Maintenance](#-monitoring-and-maintenance)
- [🌐 Team Collaboration Standards](#-team-collaboration-standards)
- [📚 Documentation Writing Guide](#-documentation-writing-guide)

---

## 🎯 Development Best Practices

### Project Structure Standards

```
viv-transformer/
├── src/                          # Source code directory
│   ├── models/                   # Model definitions
│   │   ├── __init__.py
│   │   ├── transformer.py
│   │   └── attention/
│   │       ├── __init__.py
│   │       ├── base.py
│   │       └── implementations/
│   ├── data/                     # Data processing
│   │   ├── __init__.py
│   │   ├── datasets.py
│   │   ├── preprocessing.py
│   │   └── augmentation.py
│   ├── training/                 # Training related
│   │   ├── __init__.py
│   │   ├── trainer.py
│   │   ├── losses.py
│   │   └── optimizers.py
│   ├── utils/                    # Utility functions
│   │   ├── __init__.py
│   │   ├── config.py
│   │   ├── logging.py
│   │   └── metrics.py
│   └── api/                      # API interfaces
│       ├── __init__.py
│       ├── server.py
│       └── endpoints/
├── config/                       # Configuration files
│   ├── base.yaml
│   ├── development.yaml
│   ├── production.yaml
│   └── experiments/
├── tests/                        # Test code
│   ├── unit/
│   ├── integration/
│   └── performance/
├── docs/                         # Documentation
│   ├── api/
│   ├── tutorials/
│   └── examples/
├── scripts/                      # Script files
│   ├── train.py
│   ├── evaluate.py
│   └── deploy.py
├── requirements/                 # Dependency management
│   ├── base.txt
│   ├── dev.txt
│   └── prod.txt
├── .github/                      # GitHub configuration
│   └── workflows/
├── docker/                       # Docker configuration
├── k8s/                         # Kubernetes configuration
└── README.md
```

### Code Organization Principles

```python
# best_practices_example.py
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, Tuple, Any
from dataclasses import dataclass
from pathlib import Path
import logging
import torch
import torch.nn as nn

# 1. Use type annotations
@dataclass
class ModelConfig:
    """Model configuration class
    
    Use dataclass to simplify configuration management and provide type safety
    """
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 6
    dropout: float = 0.1
    max_seq_length: int = 1024
    
    def __post_init__(self):
        """Configuration validation"""
        if self.d_model % self.n_heads != 0:
            raise ValueError(f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})")
        
        if self.dropout < 0 or self.dropout > 1:
            raise ValueError(f"dropout must be between 0 and 1, got {self.dropout}")

# 2. Use abstract base classes to define interfaces
class AttentionMechanism(ABC):
    """Attention mechanism abstract base class
    
    Define unified interface for easy extension and testing
    """
    
    @abstractmethod
    def forward(self, 
                query: torch.Tensor, 
                key: torch.Tensor, 
                value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass
        
        Args:
            query: Query tensor [batch_size, seq_len, d_model]
            key: Key tensor [batch_size, seq_len, d_model]
            value: Value tensor [batch_size, seq_len, d_model]
            mask: Optional mask tensor
            
        Returns:
            Attention output tensor [batch_size, seq_len, d_model]
        """
        pass
    
    @abstractmethod
    def get_attention_weights(self) -> Optional[torch.Tensor]:
        """Get attention weights for visualization"""
        pass

# 3. Implement specific attention mechanisms
class ScaledDotProductAttention(AttentionMechanism, nn.Module):
    """Scaled dot-product attention
    
    Standard Transformer attention mechanism implementation
    """
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.scale = self.d_k ** -0.5
        
        # Linear transformation layers
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.attention_weights = None
    
    def forward(self, 
                query: torch.Tensor, 
                key: torch.Tensor, 
                value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass implementation"""
        
        batch_size, seq_len, d_model = query.shape
        
        # Linear transformations
        Q = self.w_q(query).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Compute attention
        attention_output = self._scaled_dot_product_attention(Q, K, V, mask)
        
        # Reshape output
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, d_model
        )
        
        return self.w_o(attention_output)
    
    def _scaled_dot_product_attention(self, 
                                    Q: torch.Tensor, 
                                    K: torch.Tensor, 
                                    V: torch.Tensor,
                                    mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Scaled dot-product attention computation"""
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        
        # Apply mask
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Softmax normalization
        attention_weights = torch.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Save attention weights for visualization
        self.attention_weights = attention_weights.detach()
        
        # Apply attention weights
        return torch.matmul(attention_weights, V)
    
    def get_attention_weights(self) -> Optional[torch.Tensor]:
        """Get the attention weights from the last computation"""
        return self.attention_weights

# 4. Factory pattern for creating attention mechanisms
class AttentionFactory:
    """Attention mechanism factory class
    
    Use factory pattern to uniformly create different types of attention mechanisms
    """
    
    _registry: Dict[str, type] = {
        'scaled_dot_product': ScaledDotProductAttention,
        # More attention mechanisms can be registered
    }
    
    @classmethod
    def create(cls, 
               attention_type: str, 
               config: ModelConfig) -> AttentionMechanism:
        """Create attention mechanism instance
        
        Args:
            attention_type: Type of attention mechanism
            config: Model configuration
            
        Returns:
            Attention mechanism instance
            
        Raises:
            ValueError: Unsupported attention mechanism type
        """
        
        if attention_type not in cls._registry:
            available_types = list(cls._registry.keys())
            raise ValueError(
                f"Unsupported attention type: {attention_type}. "
                f"Available types: {available_types}"
            )
        
        attention_class = cls._registry[attention_type]
        return attention_class(
            d_model=config.d_model,
            n_heads=config.n_heads,
            dropout=config.dropout
        )
    
    @classmethod
    def register(cls, name: str, attention_class: type):
        """Register new attention mechanism
        
        Args:
            name: Attention mechanism name
            attention_class: Attention mechanism class
        """
        cls._registry[name] = attention_class
    
    @classmethod
    def list_available(cls) -> List[str]:
        """List all available attention mechanisms"""
        return list(cls._registry.keys())

# 5. Use context managers for resource management
class ModelTrainer:
    """Model trainer
    
    Use context manager to ensure proper resource cleanup
    """
    
    def __init__(self, model: nn.Module, config: ModelConfig):
        self.model = model
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def __enter__(self):
        """Enter training context"""
        self.model.train()
        self.logger.info("Starting training mode")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit training context"""
        self.model.eval()
        if exc_type is not None:
            self.logger.error(f"Error occurred during training: {exc_val}")
        else:
            self.logger.info("Training completed")
        
        # Clean GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def train_epoch(self, dataloader, optimizer, criterion):
        """Train one epoch"""
        total_loss = 0.0
        
        for batch_idx, (data, target) in enumerate(dataloader):
            optimizer.zero_grad()
            
            output = self.model(data)
            loss = criterion(output, target)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 100 == 0:
                self.logger.info(
                    f"Batch {batch_idx}, Loss: {loss.item():.6f}"
                )
        
        return total_loss / len(dataloader)

# Usage example
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Create configuration
    config = ModelConfig(
        d_model=512,
        n_heads=8,
        n_layers=6,
        dropout=0.1
    )
    
    # Create attention mechanism
    attention = AttentionFactory.create('scaled_dot_product', config)
    
    # Use trainer
    model = nn.Sequential()  # Simplified model
    
    with ModelTrainer(model, config) as trainer:
        # Training code
        pass
```

---

## 🧠 Model Design Principles

### Modular Design

```python
# modular_design.py
from typing import Dict, List, Optional, Callable
import torch
import torch.nn as nn
from abc import ABC, abstractmethod

class ModularComponent(ABC):
    """Modular component base class"""
    
    @abstractmethod
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        pass
    
    @abstractmethod
    def get_config(self) -> Dict:
        """Get component configuration"""
        pass
    
    @abstractmethod
    def get_parameters_count(self) -> int:
        """Get parameter count"""
        pass

class AttentionBlock(ModularComponent, nn.Module):
    """Attention block
    
    Pluggable attention mechanism implementation
    """
    
    def __init__(self, 
                 attention_type: str,
                 d_model: int,
                 n_heads: int,
                 dropout: float = 0.1,
                 use_residual: bool = True,
                 use_layer_norm: bool = True):
        super().__init__()
        
        self.attention_type = attention_type
        self.d_model = d_model
        self.n_heads = n_heads
        self.dropout = dropout
        self.use_residual = use_residual
        self.use_layer_norm = use_layer_norm
        
        # Create attention mechanism
        self.attention = self._create_attention()
        
        # Optional layer normalization
        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout_layer = nn.Dropout(dropout)
    
    def _create_attention(self) -> nn.Module:
        """Create specific attention mechanism"""
        # Here you can create different attention mechanisms based on attention_type
        return nn.MultiheadAttention(
            embed_dim=self.d_model,
            num_heads=self.n_heads,
            dropout=self.dropout,
            batch_first=True
        )
    
    def forward(self, 
                x: torch.Tensor, 
                mask: Optional[torch.Tensor] = None,
                **kwargs) -> torch.Tensor:
        """Forward propagation"""
        
        # Save input for residual connection
        residual = x if self.use_residual else None
        
        # Attention computation
        attn_output, _ = self.attention(x, x, x, attn_mask=mask)
        
        # Dropout
        attn_output = self.dropout_layer(attn_output)
        
        # Residual connection
        if residual is not None:
            attn_output = attn_output + residual
        
        # Layer normalization
        if self.use_layer_norm:
            attn_output = self.layer_norm(attn_output)
        
        return attn_output
    
    def get_config(self) -> Dict:
        """Get configuration"""
        return {
            'attention_type': self.attention_type,
            'd_model': self.d_model,
            'n_heads': self.n_heads,
            'dropout': self.dropout,
            'use_residual': self.use_residual,
            'use_layer_norm': self.use_layer_norm
        }
    
    def get_parameters_count(self) -> int:
        """Get parameter count"""
        return sum(p.numel() for p in self.parameters())

class FeedForwardBlock(ModularComponent, nn.Module):
    """Feed-forward network block"""
    
    def __init__(self,
                 d_model: int,
                 d_ff: int,
                 activation: str = 'relu',
                 dropout: float = 0.1,
                 use_residual: bool = True,
                 use_layer_norm: bool = True):
        super().__init__()
        
        self.d_model = d_model
        self.d_ff = d_ff
        self.activation = activation
        self.dropout = dropout
        self.use_residual = use_residual
        self.use_layer_norm = use_layer_norm
        
        # Feed-forward network
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        
        # Activation function
        self.activation_fn = self._get_activation_fn(activation)
        
        # Dropout
        self.dropout_layer = nn.Dropout(dropout)
        
        # Optional layer normalization
        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(d_model)
    
    def _get_activation_fn(self, activation: str) -> Callable:
        """Get activation function"""
        activations = {
            'relu': torch.relu,
            'gelu': torch.nn.functional.gelu,
            'swish': lambda x: x * torch.sigmoid(x),
            'mish': lambda x: x * torch.tanh(torch.nn.functional.softplus(x))
        }
        
        if activation not in activations:
            raise ValueError(f"Unsupported activation: {activation}")
        
        return activations[activation]
    
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Forward propagation"""
        
        # Save input for residual connection
        residual = x if self.use_residual else None
        
        # Feed-forward network
        ff_output = self.linear1(x)
        ff_output = self.activation_fn(ff_output)
        ff_output = self.dropout_layer(ff_output)
        ff_output = self.linear2(ff_output)
        ff_output = self.dropout_layer(ff_output)
        
        # Residual connection
        if residual is not None:
            ff_output = ff_output + residual
        
        # Layer normalization
        if self.use_layer_norm:
            ff_output = self.layer_norm(ff_output)
        
        return ff_output
    
    def get_config(self) -> Dict:
        """Get configuration"""
        return {
            'd_model': self.d_model,
            'd_ff': self.d_ff,
            'activation': self.activation,
            'dropout': self.dropout,
            'use_residual': self.use_residual,
            'use_layer_norm': self.use_layer_norm
        }
    
    def get_parameters_count(self) -> int:
        """Get parameter count"""
        return sum(p.numel() for p in self.parameters())

class ModularTransformerLayer(nn.Module):
    """Modular Transformer Layer
    
    Allows flexible composition of different components
    """
    
    def __init__(self, 
                 attention_block: AttentionBlock,
                 feedforward_block: FeedForwardBlock,
                 layer_order: List[str] = ['attention', 'feedforward']):
        super().__init__()
        
        self.attention_block = attention_block
        self.feedforward_block = feedforward_block
        self.layer_order = layer_order
        
        # Validate layer order
        valid_layers = {'attention', 'feedforward'}
        if not all(layer in valid_layers for layer in layer_order):
            raise ValueError(f"Invalid layer order. Valid layers: {valid_layers}")
    
    def forward(self, 
                x: torch.Tensor, 
                mask: Optional[torch.Tensor] = None,
                **kwargs) -> torch.Tensor:
        """Forward propagation"""
        
        for layer_type in self.layer_order:
            if layer_type == 'attention':
                x = self.attention_block(x, mask=mask, **kwargs)
            elif layer_type == 'feedforward':
                x = self.feedforward_block(x, **kwargs)
        
        return x
    
    def get_config(self) -> Dict:
        """Get configuration"""
        return {
            'attention_config': self.attention_block.get_config(),
            'feedforward_config': self.feedforward_block.get_config(),
            'layer_order': self.layer_order
        }
    
    def get_parameters_count(self) -> int:
        """Get parameter count"""
        return (
            self.attention_block.get_parameters_count() + 
            self.feedforward_block.get_parameters_count()
        )

# Usage example
if __name__ == "__main__":
    # Create attention block
    attention_block = AttentionBlock(
        attention_type='multihead',
        d_model=512,
        n_heads=8,
        dropout=0.1
    )
    
    # Create feed-forward block
    feedforward_block = FeedForwardBlock(
        d_model=512,
        d_ff=2048,
        activation='gelu',
        dropout=0.1
    )
    
    # Create Transformer layer
    transformer_layer = ModularTransformerLayer(
        attention_block=attention_block,
        feedforward_block=feedforward_block,
        layer_order=['attention', 'feedforward']
    )
    
    # Test
    x = torch.randn(32, 100, 512)  # [batch_size, seq_len, d_model]
    output = transformer_layer(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Parameter count: {transformer_layer.get_parameters_count():,}")
    print(f"Config: {transformer_layer.get_config()}")
```

### Extensibility Design

```python
# extensibility_design.py
from typing import Dict, Any, Type, Optional, List
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
import importlib
import inspect

class PluginRegistry:
    """Plugin registry
    
    Supports dynamic registration and loading of components
    """
    
    def __init__(self):
        self._registry: Dict[str, Dict[str, Any]] = {
            'attention': {},
            'loss': {},
            'optimizer': {},
            'scheduler': {},
            'metric': {}
        }
    
    def register(self, 
                 category: str, 
                 name: str, 
                 component_class: Type,
                 description: str = "",
                 **metadata) -> None:
        """Register component
        
        Args:
            category: Component category
            name: Component name
            component_class: Component class
            description: Description
            **metadata: Additional metadata
        """
        
        if category not in self._registry:
            self._registry[category] = {}
        
        self._registry[category][name] = {
            'class': component_class,
            'description': description,
            'metadata': metadata
        }
    
    def get(self, category: str, name: str) -> Optional[Type]:
        """Get component class"""
        
        if category in self._registry and name in self._registry[category]:
            return self._registry[category][name]['class']
        return None
    
    def list_components(self, category: str) -> List[str]:
        """List all components in the specified category"""
        
        if category in self._registry:
            return list(self._registry[category].keys())
        return []
    
    def get_info(self, category: str, name: str) -> Optional[Dict[str, Any]]:
        """Get component information"""
        
        if category in self._registry and name in self._registry[category]:
            return self._registry[category][name]
        return None
    
    def load_from_module(self, module_path: str, category: str) -> None:
        """Dynamically load components from a module
        
        Args:
            module_path: Module path, e.g., 'mypackage.attention.custom'
            category: Component category
        """
        
        try:
            module = importlib.import_module(module_path)
            
            # Find all classes in the module
            for name, obj in inspect.getmembers(module, inspect.isclass):
                # Check if it is a valid component class
                if self._is_valid_component(obj, category):
                    self.register(
                        category=category,
                        name=name.lower(),
                        component_class=obj,
                        description=obj.__doc__ or "",
                        module=module_path
                    )
                    
        except ImportError as e:
            raise ImportError(f"Failed to load module {module_path}: {e}")
    
    def _is_valid_component(self, obj: Type, category: str) -> bool:
        """Check if it is a valid component class"""
        
        # Basic checks
        if not inspect.isclass(obj):
            return False
        
        # Check if it is a subclass of nn.Module (for neural network components)
        if category in ['attention', 'loss'] and not issubclass(obj, nn.Module):
            return False
        
        # Check for required methods
        required_methods = {
            'attention': ['forward'],
            'loss': ['forward'],
            'optimizer': [],
            'scheduler': [],
            'metric': ['compute']
        }
        
        if category in required_methods:
            for method in required_methods[category]:
                if not hasattr(obj, method):
                    return False
        
        return True

# Global plugin registry
plugin_registry = PluginRegistry()

class ComponentFactory:
    """Component factory
    
    Create component instances using the registry
    """
    
    @staticmethod
    def create(category: str, 
               name: str, 
               config: Dict[str, Any],
               **kwargs) -> Any:
        """Create a component instance
        
        Args:
            category: Component category
            name: Component name
            config: Configuration parameters
            **kwargs: Additional parameters
            
        Returns:
            Component instance
        """
        
        component_class = plugin_registry.get(category, name)
        if component_class is None:
            available = plugin_registry.list_components(category)
            raise ValueError(
                f"Unknown {category} component: {name}. "
                f"Available: {available}"
            )
        
        # Merge config and additional parameters
        init_params = {**config, **kwargs}
        
        # Filter out unnecessary parameters
        sig = inspect.signature(component_class.__init__)
        valid_params = {}
        
        for param_name, param in sig.parameters.items():
            if param_name == 'self':
                continue
            
            if param_name in init_params:
                valid_params[param_name] = init_params[param_name]
            elif param.default == inspect.Parameter.empty:
                raise ValueError(
                    f"Missing required parameter '{param_name}' "
                    f"for {category} component '{name}'"
                )
        
        return component_class(**valid_params)

# Decorator for automatic component registration
def register_component(category: str, 
                      name: Optional[str] = None,
                      description: str = "",
                      **metadata):
    """Component registration decorator
    
    Args:
        category: Component category
        name: Component name (defaults to lowercase class name)
        description: Description
        **metadata: Additional metadata
    """
    
    def decorator(cls):
        component_name = name or cls.__name__.lower()
        plugin_registry.register(
            category=category,
            name=component_name,
            component_class=cls,
            description=description or cls.__doc__ or "",
            **metadata
        )
        return cls
    
    return decorator

# Usage example: register a custom attention mechanism
@register_component(
    category='attention',
    name='custom_attention',
    description='Custom attention mechanism',
    paper_url='https://example.com/paper',
    author='Your Name'
)
class CustomAttention(nn.Module):
    """Custom attention mechanism
    
    This is an example custom attention mechanism
    """
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.dropout = dropout
        
        # Custom implementation
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass"""
        output, _ = self.attention(x, x, x, attn_mask=mask)
        return output

# Configuration-driven model construction
class ConfigurableModel(nn.Module):
    """Configurable model
    
    Dynamically builds the model from configuration
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        
        # Build model components
        self._build_model()
    
    def _build_model(self):
        """Build model from configuration"""
        
        # Create attention layer
        attention_config = self.config.get('attention', {})
        self.attention = ComponentFactory.create(
            category='attention',
            name=attention_config.get('type', 'multihead'),
            config=attention_config.get('params', {})
        )
        
        # Create other components...
        # More components can be created based on the configuration
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        return self.attention(x)

# Usage example
if __name__ == "__main__":
    # List available components
    print("Available attention mechanisms:")
    for name in plugin_registry.list_components('attention'):
        info = plugin_registry.get_info('attention', name)
        print(f"  - {name}: {info['description']}")
    
    # Create model from configuration
    model_config = {
        'attention': {
            'type': 'custom_attention',
            'params': {
                'd_model': 512,
                'n_heads': 8,
                'dropout': 0.1
            }
        }
    }
    
    model = ConfigurableModel(model_config)
    
    # Test
    x = torch.randn(32, 100, 512)
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
```

---

## 📊 Data Processing Best Practices

### Data Validation and Cleaning

```python
# data_validation.py
import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
from pathlib import Path
import logging
from abc import ABC, abstractmethod

@dataclass
class DataQualityReport:
    """Data quality report"""
    
    total_samples: int
    valid_samples: int
    invalid_samples: int
    missing_values: Dict[str, int]
    outliers: Dict[str, int]
    duplicates: int
    data_types: Dict[str, str]
    statistics: Dict[str, Dict[str, float]]
    
    @property
    def validity_rate(self) -> float:
        """Data validity rate"""
        return self.valid_samples / self.total_samples if self.total_samples > 0 else 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict"""
        return {
            'total_samples': self.total_samples,
            'valid_samples': self.valid_samples,
            'invalid_samples': self.invalid_samples,
            'validity_rate': self.validity_rate,
            'missing_values': self.missing_values,
            'outliers': self.outliers,
            'duplicates': self.duplicates,
            'data_types': self.data_types,
            'statistics': self.statistics
        }

class DataValidator(ABC):
    """Abstract base class for data validators"""
    
    @abstractmethod
    def validate(self, data: Any) -> Tuple[bool, str]:
        """Validate data
        
        Returns:
            (is_valid, error_message)
        """
        pass

class RangeValidator(DataValidator):
    """Range validator"""
    
    def __init__(self, min_val: float, max_val: float, column: str):
        self.min_val = min_val
        self.max_val = max_val
        self.column = column
    
    def validate(self, data: pd.DataFrame) -> Tuple[bool, str]:
        """Validate numeric range"""
        
        if self.column not in data.columns:
            return False, f"Column '{self.column}' does not exist"
        
        values = data[self.column]
        out_of_range = (values < self.min_val) | (values > self.max_val)
        
        if out_of_range.any():
            count = out_of_range.sum()
            return False, f"Column '{self.column}' has {count} values out of range [{self.min_val}, {self.max_val}]"
        
        return True, ""

class TypeValidator(DataValidator):
    """Type validator"""
    
    def __init__(self, expected_types: Dict[str, type]):
        self.expected_types = expected_types
    
    def validate(self, data: pd.DataFrame) -> Tuple[bool, str]:
        """Validate data types"""
        
        for column, expected_type in self.expected_types.items():
            if column not in data.columns:
                return False, f"Column '{column}' does not exist"
            
            if expected_type == float:
                if not pd.api.types.is_numeric_dtype(data[column]):
                    return False, f"Column '{column}' should be numeric type"
            elif expected_type == str:
                if not pd.api.types.is_string_dtype(data[column]):
                    return False, f"Column '{column}' should be string type"
        
        return True, ""

class ShapeValidator(DataValidator):
    """Shape validator"""
    
    def __init__(self, expected_shape: Tuple[Optional[int], ...]):
        self.expected_shape = expected_shape
    
    def validate(self, data: Union[np.ndarray, torch.Tensor]) -> Tuple[bool, str]:
        """Validate data shape"""
        
        actual_shape = data.shape
        
        if len(actual_shape) != len(self.expected_shape):
            return False, f"Dimension mismatch: expected {len(self.expected_shape)}, actual {len(actual_shape)}"
        
        for i, (expected, actual) in enumerate(zip(self.expected_shape, actual_shape)):
            if expected is not None and expected != actual:
                return False, f"Dimension {i} size mismatch: expected {expected}, actual {actual}"
        
        return True, ""

class DataQualityAnalyzer:
    """Data quality analyzer"""
    
    def __init__(self, validators: List[DataValidator] = None):
        self.validators = validators or []
        self.logger = logging.getLogger(__name__)
    
    def analyze(self, data: pd.DataFrame) -> DataQualityReport:
        """Analyze data quality"""
        
        self.logger.info("Starting data quality analysis")
        
        # Basic statistics
        total_samples = len(data)
        
        # Missing value analysis
        missing_values = data.isnull().sum().to_dict()
        
        # Duplicate value analysis
        duplicates = data.duplicated().sum()
        
        # Data types
        data_types = {col: str(dtype) for col, dtype in data.dtypes.items()}
        
        # Statistical information
        statistics = {}
        for col in data.select_dtypes(include=[np.number]).columns:
            statistics[col] = {
                'mean': float(data[col].mean()),
                'std': float(data[col].std()),
                'min': float(data[col].min()),
                'max': float(data[col].max()),
                'median': float(data[col].median())
            }
        
        # Outlier detection (using IQR method)
        outliers = self._detect_outliers(data)
        
        # Validator checks
        valid_samples = total_samples
        for validator in self.validators:
            is_valid, error_msg = validator.validate(data)
            if not is_valid:
                self.logger.warning(f"Validation failed: {error_msg}")
                # Here you can adjust valid_samples based on specific validator type
        
        invalid_samples = total_samples - valid_samples
        
        report = DataQualityReport(
            total_samples=total_samples,
            valid_samples=valid_samples,
            invalid_samples=invalid_samples,
            missing_values=missing_values,
            outliers=outliers,
            duplicates=duplicates,
            data_types=data_types,
            statistics=statistics
        )
        
        self.logger.info(f"Data quality analysis completed, validity rate: {report.validity_rate:.2%}")
        return report
    
    def _detect_outliers(self, data: pd.DataFrame) -> Dict[str, int]:
        """Detect outliers using IQR method"""
        
        outliers = {}
        
        for col in data.select_dtypes(include=[np.number]).columns:
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outlier_mask = (data[col] < lower_bound) | (data[col] > upper_bound)
            outliers[col] = outlier_mask.sum()
        
        return outliers

class DataCleaner:
    """Data cleaner"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
    
    def clean(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean data"""
        
        self.logger.info("Starting data cleaning")
        cleaned_data = data.copy()
        
        # Handle missing values
        cleaned_data = self._handle_missing_values(cleaned_data)
        
        # Handle duplicates
        cleaned_data = self._handle_duplicates(cleaned_data)
        
        # Handle outliers
        cleaned_data = self._handle_outliers(cleaned_data)
        
        # Data type conversion
        cleaned_data = self._convert_data_types(cleaned_data)
        
        # Data standardization/normalization
        cleaned_data = self._normalize_data(cleaned_data)
        
        self.logger.info(f"Data cleaning completed, from {len(data)} rows to {len(cleaned_data)} rows")
        return cleaned_data
    
    def _handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values"""
        
        strategy = self.config.get('missing_value_strategy', 'drop')
        
        if strategy == 'drop':
            # Drop rows containing missing values
            return data.dropna()
        elif strategy == 'fill_mean':
            # Fill numeric columns with mean
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())
            return data
        elif strategy == 'fill_median':
            # Fill numeric columns with median
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].median())
            return data
        elif strategy == 'forward_fill':
            # Forward fill
            return data.fillna(method='ffill')
        else:
            return data
    
    def _handle_duplicates(self, data: pd.DataFrame) -> pd.DataFrame:
        """Handle duplicate values"""
        
        if self.config.get('remove_duplicates', True):
            return data.drop_duplicates()
        return data
    
    def _handle_outliers(self, data: pd.DataFrame) -> pd.DataFrame:
        """Handle outliers"""
        
        strategy = self.config.get('outlier_strategy', 'none')
        
        if strategy == 'remove':
            # Remove outliers using IQR method
            for col in data.select_dtypes(include=[np.number]).columns:
                Q1 = data[col].quantile(0.25)
                Q3 = data[col].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                data = data[(data[col] >= lower_bound) & (data[col] <= upper_bound)]
        
        elif strategy == 'clip':
            # Clip outliers
            for col in data.select_dtypes(include=[np.number]).columns:
                Q1 = data[col].quantile(0.25)
                Q3 = data[col].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                data[col] = data[col].clip(lower=lower_bound, upper=upper_bound)
        
        return data
    
    def _convert_data_types(self, data: pd.DataFrame) -> pd.DataFrame:
        """Convert data types"""
        
        type_mapping = self.config.get('type_mapping', {})
        
        for col, target_type in type_mapping.items():
            if col in data.columns:
                try:
                    data[col] = data[col].astype(target_type)
                except Exception as e:
                    self.logger.warning(f"Cannot convert column '{col}' to type '{target_type}': {e}")
        
        return data
    
    def _normalize_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Data standardization/normalization"""
        
        normalization = self.config.get('normalization', 'none')
        
        if normalization == 'standardize':
            # Z-score standardization
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            data[numeric_cols] = (data[numeric_cols] - data[numeric_cols].mean()) / data[numeric_cols].std()
        
        elif normalization == 'min_max':
            # Min-Max normalization
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            data[numeric_cols] = (data[numeric_cols] - data[numeric_cols].min()) / (data[numeric_cols].max() - data[numeric_cols].min())
        
        return data

# Usage example
if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)
    data = pd.DataFrame({
        'feature1': np.random.normal(0, 1, 1000),
        'feature2': np.random.uniform(-1, 1, 1000),
        'feature3': np.random.exponential(2, 1000),
        'label': np.random.randint(0, 2, 1000)
    })
    
    # Add some missing values and outliers
    data.loc[np.random.choice(1000, 50, replace=False), 'feature1'] = np.nan
    data.loc[np.random.choice(1000, 10, replace=False), 'feature2'] = 100  # outliers
    
    # Create validators
    validators = [
        RangeValidator(-5, 5, 'feature1'),
        RangeValidator(-2, 2, 'feature2'),
        TypeValidator({'feature1': float, 'feature2': float, 'label': int})
    ]
    
    # Data quality analysis
    analyzer = DataQualityAnalyzer(validators)
    quality_report = analyzer.analyze(data)
    
    print("Data Quality Report:")
    print(f"Total samples: {quality_report.total_samples}")
    print(f"Valid samples: {quality_report.valid_samples}")
    print(f"Validity rate: {quality_report.validity_rate:.2%}")
    print(f"Missing values: {quality_report.missing_values}")
    print(f"Outliers: {quality_report.outliers}")
    print(f"Duplicates: {quality_report.duplicates}")
    
    # Data cleaning
    cleaner_config = {
        'missing_value_strategy': 'fill_mean',
        'remove_duplicates': True,
        'outlier_strategy': 'clip',
        'normalization': 'standardize'
    }
    
    cleaner = DataCleaner(cleaner_config)
    cleaned_data = cleaner.clean(data)
    
    print(f"\nShape before cleaning: {data.shape}")
    print(f"Shape after cleaning: {cleaned_data.shape}")
    print(f"Missing values after cleaning: {cleaned_data.isnull().sum().sum()}")
```

---

*This best practices guide provides comprehensive development and usage recommendations. For more details, please refer to the full documentation.*