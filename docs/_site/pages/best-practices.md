# 💡 最佳实践指南

> VIVTransformer开发、训练和部署的最佳实践集合

---

## 📋 目录

- [🎯 开发最佳实践](#-开发最佳实践)
- [🧠 模型设计原则](#-模型设计原则)
- [📊 数据处理最佳实践](#-数据处理最佳实践)
- [🏋️ 训练策略优化](#️-训练策略优化)
- [🔧 代码质量保证](#-代码质量保证)
- [🚀 性能优化技巧](#-性能优化技巧)
- [🔒 安全最佳实践](#-安全最佳实践)
- [📈 监控与维护](#-监控与维护)
- [🌐 团队协作规范](#-团队协作规范)
- [📚 文档编写指南](#-文档编写指南)

---

## 🎯 开发最佳实践

### 项目结构规范

```
viv-transformer/
├── src/                          # 源代码目录
│   ├── models/                   # 模型定义
│   │   ├── __init__.py
│   │   ├── transformer.py
│   │   └── attention/
│   │       ├── __init__.py
│   │       ├── base.py
│   │       └── implementations/
│   ├── data/                     # 数据处理
│   │   ├── __init__.py
│   │   ├── datasets.py
│   │   ├── preprocessing.py
│   │   └── augmentation.py
│   ├── training/                 # 训练相关
│   │   ├── __init__.py
│   │   ├── trainer.py
│   │   ├── losses.py
│   │   └── optimizers.py
│   ├── utils/                    # 工具函数
│   │   ├── __init__.py
│   │   ├── config.py
│   │   ├── logging.py
│   │   └── metrics.py
│   └── api/                      # API接口
│       ├── __init__.py
│       ├── server.py
│       └── endpoints/
├── config/                       # 配置文件
│   ├── base.yaml
│   ├── development.yaml
│   ├── production.yaml
│   └── experiments/
├── tests/                        # 测试代码
│   ├── unit/
│   ├── integration/
│   └── performance/
├── docs/                         # 文档
│   ├── api/
│   ├── tutorials/
│   └── examples/
├── scripts/                      # 脚本文件
│   ├── train.py
│   ├── evaluate.py
│   └── deploy.py
├── requirements/                 # 依赖管理
│   ├── base.txt
│   ├── dev.txt
│   └── prod.txt
├── .github/                      # GitHub配置
│   └── workflows/
├── docker/                       # Docker配置
├── k8s/                         # Kubernetes配置
└── README.md
```

### 代码组织原则

```python
# best_practices_example.py
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, Tuple, Any
from dataclasses import dataclass
from pathlib import Path
import logging
import torch
import torch.nn as nn

# 1. 使用类型注解
@dataclass
class ModelConfig:
    """模型配置类
    
    使用dataclass简化配置管理，提供类型安全
    """
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 6
    dropout: float = 0.1
    max_seq_length: int = 1024
    
    def __post_init__(self):
        """配置验证"""
        if self.d_model % self.n_heads != 0:
            raise ValueError(f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})")
        
        if self.dropout < 0 or self.dropout > 1:
            raise ValueError(f"dropout must be between 0 and 1, got {self.dropout}")

# 2. 使用抽象基类定义接口
class AttentionMechanism(ABC):
    """注意力机制抽象基类
    
    定义统一接口，便于扩展和测试
    """
    
    @abstractmethod
    def forward(self, 
                query: torch.Tensor, 
                key: torch.Tensor, 
                value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """前向传播
        
        Args:
            query: 查询张量 [batch_size, seq_len, d_model]
            key: 键张量 [batch_size, seq_len, d_model]
            value: 值张量 [batch_size, seq_len, d_model]
            mask: 可选的掩码张量
            
        Returns:
            注意力输出张量 [batch_size, seq_len, d_model]
        """
        pass
    
    @abstractmethod
    def get_attention_weights(self) -> Optional[torch.Tensor]:
        """获取注意力权重用于可视化"""
        pass

# 3. 实现具体的注意力机制
class ScaledDotProductAttention(AttentionMechanism, nn.Module):
    """缩放点积注意力
    
    标准的Transformer注意力机制实现
    """
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.scale = self.d_k ** -0.5
        
        # 线性变换层
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
        """前向传播实现"""
        
        batch_size, seq_len, d_model = query.shape
        
        # 线性变换
        Q = self.w_q(query).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # 计算注意力
        attention_output = self._scaled_dot_product_attention(Q, K, V, mask)
        
        # 重塑输出
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, d_model
        )
        
        return self.w_o(attention_output)
    
    def _scaled_dot_product_attention(self, 
                                    Q: torch.Tensor, 
                                    K: torch.Tensor, 
                                    V: torch.Tensor,
                                    mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """缩放点积注意力计算"""
        
        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        
        # 应用掩码
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Softmax归一化
        attention_weights = torch.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # 保存注意力权重用于可视化
        self.attention_weights = attention_weights.detach()
        
        # 应用注意力权重
        return torch.matmul(attention_weights, V)
    
    def get_attention_weights(self) -> Optional[torch.Tensor]:
        """获取最后一次计算的注意力权重"""
        return self.attention_weights

# 4. 工厂模式创建注意力机制
class AttentionFactory:
    """注意力机制工厂类
    
    使用工厂模式统一创建不同类型的注意力机制
    """
    
    _registry: Dict[str, type] = {
        'scaled_dot_product': ScaledDotProductAttention,
        # 可以注册更多注意力机制
    }
    
    @classmethod
    def create(cls, 
               attention_type: str, 
               config: ModelConfig) -> AttentionMechanism:
        """创建注意力机制实例
        
        Args:
            attention_type: 注意力机制类型
            config: 模型配置
            
        Returns:
            注意力机制实例
            
        Raises:
            ValueError: 不支持的注意力机制类型
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
        """注册新的注意力机制
        
        Args:
            name: 注意力机制名称
            attention_class: 注意力机制类
        """
        cls._registry[name] = attention_class
    
    @classmethod
    def list_available(cls) -> List[str]:
        """列出所有可用的注意力机制"""
        return list(cls._registry.keys())

# 5. 使用上下文管理器进行资源管理
class ModelTrainer:
    """模型训练器
    
    使用上下文管理器确保资源正确释放
    """
    
    def __init__(self, model: nn.Module, config: ModelConfig):
        self.model = model
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def __enter__(self):
        """进入训练上下文"""
        self.model.train()
        self.logger.info("开始训练模式")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """退出训练上下文"""
        self.model.eval()
        if exc_type is not None:
            self.logger.error(f"训练过程中发生错误: {exc_val}")
        else:
            self.logger.info("训练完成")
        
        # 清理GPU内存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def train_epoch(self, dataloader, optimizer, criterion):
        """训练一个epoch"""
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

# 使用示例
if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(level=logging.INFO)
    
    # 创建配置
    config = ModelConfig(
        d_model=512,
        n_heads=8,
        n_layers=6,
        dropout=0.1
    )
    
    # 创建注意力机制
    attention = AttentionFactory.create('scaled_dot_product', config)
    
    # 使用训练器
    model = nn.Sequential()  # 简化的模型
    
    with ModelTrainer(model, config) as trainer:
        # 训练代码
        pass
```

---

## 🧠 模型设计原则

### 模块化设计

```python
# modular_design.py
from typing import Dict, List, Optional, Callable
import torch
import torch.nn as nn
from abc import ABC, abstractmethod

class ModularComponent(ABC):
    """模块化组件基类"""
    
    @abstractmethod
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        pass
    
    @abstractmethod
    def get_config(self) -> Dict:
        """获取组件配置"""
        pass
    
    @abstractmethod
    def get_parameters_count(self) -> int:
        """获取参数数量"""
        pass

class AttentionBlock(ModularComponent, nn.Module):
    """注意力块
    
    可插拔的注意力机制实现
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
        
        # 创建注意力机制
        self.attention = self._create_attention()
        
        # 可选的层归一化
        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout_layer = nn.Dropout(dropout)
    
    def _create_attention(self) -> nn.Module:
        """创建具体的注意力机制"""
        # 这里可以根据attention_type创建不同的注意力机制
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
        """前向传播"""
        
        # 保存输入用于残差连接
        residual = x if self.use_residual else None
        
        # 注意力计算
        attn_output, _ = self.attention(x, x, x, attn_mask=mask)
        
        # Dropout
        attn_output = self.dropout_layer(attn_output)
        
        # 残差连接
        if residual is not None:
            attn_output = attn_output + residual
        
        # 层归一化
        if self.use_layer_norm:
            attn_output = self.layer_norm(attn_output)
        
        return attn_output
    
    def get_config(self) -> Dict:
        """获取配置"""
        return {
            'attention_type': self.attention_type,
            'd_model': self.d_model,
            'n_heads': self.n_heads,
            'dropout': self.dropout,
            'use_residual': self.use_residual,
            'use_layer_norm': self.use_layer_norm
        }
    
    def get_parameters_count(self) -> int:
        """获取参数数量"""
        return sum(p.numel() for p in self.parameters())

class FeedForwardBlock(ModularComponent, nn.Module):
    """前馈网络块"""
    
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
        
        # 前馈网络
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        
        # 激活函数
        self.activation_fn = self._get_activation_fn(activation)
        
        # Dropout
        self.dropout_layer = nn.Dropout(dropout)
        
        # 可选的层归一化
        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(d_model)
    
    def _get_activation_fn(self, activation: str) -> Callable:
        """获取激活函数"""
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
        """前向传播"""
        
        # 保存输入用于残差连接
        residual = x if self.use_residual else None
        
        # 前馈网络
        ff_output = self.linear1(x)
        ff_output = self.activation_fn(ff_output)
        ff_output = self.dropout_layer(ff_output)
        ff_output = self.linear2(ff_output)
        ff_output = self.dropout_layer(ff_output)
        
        # 残差连接
        if residual is not None:
            ff_output = ff_output + residual
        
        # 层归一化
        if self.use_layer_norm:
            ff_output = self.layer_norm(ff_output)
        
        return ff_output
    
    def get_config(self) -> Dict:
        """获取配置"""
        return {
            'd_model': self.d_model,
            'd_ff': self.d_ff,
            'activation': self.activation,
            'dropout': self.dropout,
            'use_residual': self.use_residual,
            'use_layer_norm': self.use_layer_norm
        }
    
    def get_parameters_count(self) -> int:
        """获取参数数量"""
        return sum(p.numel() for p in self.parameters())

class ModularTransformerLayer(nn.Module):
    """模块化Transformer层
    
    可以灵活组合不同的组件
    """
    
    def __init__(self, 
                 attention_block: AttentionBlock,
                 feedforward_block: FeedForwardBlock,
                 layer_order: List[str] = ['attention', 'feedforward']):
        super().__init__()
        
        self.attention_block = attention_block
        self.feedforward_block = feedforward_block
        self.layer_order = layer_order
        
        # 验证层顺序
        valid_layers = {'attention', 'feedforward'}
        if not all(layer in valid_layers for layer in layer_order):
            raise ValueError(f"Invalid layer order. Valid layers: {valid_layers}")
    
    def forward(self, 
                x: torch.Tensor, 
                mask: Optional[torch.Tensor] = None,
                **kwargs) -> torch.Tensor:
        """前向传播"""
        
        for layer_type in self.layer_order:
            if layer_type == 'attention':
                x = self.attention_block(x, mask=mask, **kwargs)
            elif layer_type == 'feedforward':
                x = self.feedforward_block(x, **kwargs)
        
        return x
    
    def get_config(self) -> Dict:
        """获取配置"""
        return {
            'attention_config': self.attention_block.get_config(),
            'feedforward_config': self.feedforward_block.get_config(),
            'layer_order': self.layer_order
        }
    
    def get_parameters_count(self) -> int:
        """获取参数数量"""
        return (
            self.attention_block.get_parameters_count() + 
            self.feedforward_block.get_parameters_count()
        )

# 使用示例
if __name__ == "__main__":
    # 创建注意力块
    attention_block = AttentionBlock(
        attention_type='multihead',
        d_model=512,
        n_heads=8,
        dropout=0.1
    )
    
    # 创建前馈块
    feedforward_block = FeedForwardBlock(
        d_model=512,
        d_ff=2048,
        activation='gelu',
        dropout=0.1
    )
    
    # 创建Transformer层
    transformer_layer = ModularTransformerLayer(
        attention_block=attention_block,
        feedforward_block=feedforward_block,
        layer_order=['attention', 'feedforward']
    )
    
    # 测试
    x = torch.randn(32, 100, 512)  # [batch_size, seq_len, d_model]
    output = transformer_layer(x)
    
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    print(f"参数数量: {transformer_layer.get_parameters_count():,}")
    print(f"配置: {transformer_layer.get_config()}")
```

### 可扩展性设计

```python
# extensibility_design.py
from typing import Dict, Any, Type, Optional, List
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
import importlib
import inspect

class PluginRegistry:
    """插件注册表
    
    支持动态注册和加载组件
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
        """注册组件
        
        Args:
            category: 组件类别
            name: 组件名称
            component_class: 组件类
            description: 描述信息
            **metadata: 额外的元数据
        """
        
        if category not in self._registry:
            self._registry[category] = {}
        
        self._registry[category][name] = {
            'class': component_class,
            'description': description,
            'metadata': metadata
        }
    
    def get(self, category: str, name: str) -> Optional[Type]:
        """获取组件类"""
        
        if category in self._registry and name in self._registry[category]:
            return self._registry[category][name]['class']
        return None
    
    def list_components(self, category: str) -> List[str]:
        """列出指定类别的所有组件"""
        
        if category in self._registry:
            return list(self._registry[category].keys())
        return []
    
    def get_info(self, category: str, name: str) -> Optional[Dict[str, Any]]:
        """获取组件信息"""
        
        if category in self._registry and name in self._registry[category]:
            return self._registry[category][name]
        return None
    
    def load_from_module(self, module_path: str, category: str) -> None:
        """从模块动态加载组件
        
        Args:
            module_path: 模块路径，如 'mypackage.attention.custom'
            category: 组件类别
        """
        
        try:
            module = importlib.import_module(module_path)
            
            # 查找模块中的所有类
            for name, obj in inspect.getmembers(module, inspect.isclass):
                # 检查是否是有效的组件类
                if self._is_valid_component(obj, category):
                    self.register(
                        category=category,
                        name=name.lower(),
                        component_class=obj,
                        description=obj.__doc__ or "",
                        module=module_path
                    )
                    
        except ImportError as e:
            raise ImportError(f"无法加载模块 {module_path}: {e}")
    
    def _is_valid_component(self, obj: Type, category: str) -> bool:
        """检查是否是有效的组件类"""
        
        # 基本检查
        if not inspect.isclass(obj):
            return False
        
        # 检查是否继承自nn.Module（对于神经网络组件）
        if category in ['attention', 'loss'] and not issubclass(obj, nn.Module):
            return False
        
        # 检查是否有必需的方法
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

# 全局插件注册表
plugin_registry = PluginRegistry()

class ComponentFactory:
    """组件工厂
    
    使用注册表创建组件实例
    """
    
    @staticmethod
    def create(category: str, 
               name: str, 
               config: Dict[str, Any],
               **kwargs) -> Any:
        """创建组件实例
        
        Args:
            category: 组件类别
            name: 组件名称
            config: 配置参数
            **kwargs: 额外参数
            
        Returns:
            组件实例
        """
        
        component_class = plugin_registry.get(category, name)
        if component_class is None:
            available = plugin_registry.list_components(category)
            raise ValueError(
                f"Unknown {category} component: {name}. "
                f"Available: {available}"
            )
        
        # 合并配置和额外参数
        init_params = {**config, **kwargs}
        
        # 过滤掉不需要的参数
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

# 装饰器用于自动注册组件
def register_component(category: str, 
                      name: Optional[str] = None,
                      description: str = "",
                      **metadata):
    """组件注册装饰器
    
    Args:
        category: 组件类别
        name: 组件名称（默认使用类名的小写）
        description: 描述信息
        **metadata: 额外的元数据
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

# 使用示例：注册自定义注意力机制
@register_component(
    category='attention',
    name='custom_attention',
    description='自定义注意力机制',
    paper_url='https://example.com/paper',
    author='Your Name'
)
class CustomAttention(nn.Module):
    """自定义注意力机制
    
    这是一个示例自定义注意力机制
    """
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.dropout = dropout
        
        # 自定义实现
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """前向传播"""
        output, _ = self.attention(x, x, x, attn_mask=mask)
        return output

# 配置驱动的模型构建
class ConfigurableModel(nn.Module):
    """可配置的模型
    
    根据配置文件动态构建模型
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        
        # 构建模型组件
        self._build_model()
    
    def _build_model(self):
        """根据配置构建模型"""
        
        # 创建注意力层
        attention_config = self.config.get('attention', {})
        self.attention = ComponentFactory.create(
            category='attention',
            name=attention_config.get('type', 'multihead'),
            config=attention_config.get('params', {})
        )
        
        # 创建其他组件...
        # 这里可以根据配置创建更多组件
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        return self.attention(x)

# 使用示例
if __name__ == "__main__":
    # 查看可用组件
    print("可用的注意力机制:")
    for name in plugin_registry.list_components('attention'):
        info = plugin_registry.get_info('attention', name)
        print(f"  - {name}: {info['description']}")
    
    # 使用配置创建模型
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
    
    # 测试
    x = torch.randn(32, 100, 512)
    output = model(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
```

---

## 📊 数据处理最佳实践

### 数据验证和清洗

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
    """数据质量报告"""
    
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
        """数据有效率"""
        return self.valid_samples / self.total_samples if self.total_samples > 0 else 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
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
    """数据验证器抽象基类"""
    
    @abstractmethod
    def validate(self, data: Any) -> Tuple[bool, str]:
        """验证数据
        
        Returns:
            (is_valid, error_message)
        """
        pass

class RangeValidator(DataValidator):
    """范围验证器"""
    
    def __init__(self, min_val: float, max_val: float, column: str):
        self.min_val = min_val
        self.max_val = max_val
        self.column = column
    
    def validate(self, data: pd.DataFrame) -> Tuple[bool, str]:
        """验证数值范围"""
        
        if self.column not in data.columns:
            return False, f"列 '{self.column}' 不存在"
        
        values = data[self.column]
        out_of_range = (values < self.min_val) | (values > self.max_val)
        
        if out_of_range.any():
            count = out_of_range.sum()
            return False, f"列 '{self.column}' 有 {count} 个值超出范围 [{self.min_val}, {self.max_val}]"
        
        return True, ""

class TypeValidator(DataValidator):
    """类型验证器"""
    
    def __init__(self, expected_types: Dict[str, type]):
        self.expected_types = expected_types
    
    def validate(self, data: pd.DataFrame) -> Tuple[bool, str]:
        """验证数据类型"""
        
        for column, expected_type in self.expected_types.items():
            if column not in data.columns:
                return False, f"列 '{column}' 不存在"
            
            if expected_type == float:
                if not pd.api.types.is_numeric_dtype(data[column]):
                    return False, f"列 '{column}' 应为数值类型"
            elif expected_type == str:
                if not pd.api.types.is_string_dtype(data[column]):
                    return False, f"列 '{column}' 应为字符串类型"
        
        return True, ""

class ShapeValidator(DataValidator):
    """形状验证器"""
    
    def __init__(self, expected_shape: Tuple[Optional[int], ...]):
        self.expected_shape = expected_shape
    
    def validate(self, data: Union[np.ndarray, torch.Tensor]) -> Tuple[bool, str]:
        """验证数据形状"""
        
        actual_shape = data.shape
        
        if len(actual_shape) != len(self.expected_shape):
            return False, f"维度不匹配: 期望 {len(self.expected_shape)}, 实际 {len(actual_shape)}"
        
        for i, (expected, actual) in enumerate(zip(self.expected_shape, actual_shape)):
            if expected is not None and expected != actual:
                return False, f"第 {i} 维大小不匹配: 期望 {expected}, 实际 {actual}"
        
        return True, ""

class DataQualityAnalyzer:
    """数据质量分析器"""
    
    def __init__(self, validators: List[DataValidator] = None):
        self.validators = validators or []
        self.logger = logging.getLogger(__name__)
    
    def analyze(self, data: pd.DataFrame) -> DataQualityReport:
        """分析数据质量"""
        
        self.logger.info("开始数据质量分析")
        
        # 基本统计
        total_samples = len(data)
        
        # 缺失值分析
        missing_values = data.isnull().sum().to_dict()
        
        # 重复值分析
        duplicates = data.duplicated().sum()
        
        # 数据类型
        data_types = {col: str(dtype) for col, dtype in data.dtypes.items()}
        
        # 统计信息
        statistics = {}
        for col in data.select_dtypes(include=[np.number]).columns:
            statistics[col] = {
                'mean': float(data[col].mean()),
                'std': float(data[col].std()),
                'min': float(data[col].min()),
                'max': float(data[col].max()),
                'median': float(data[col].median())
            }
        
        # 异常值检测（使用IQR方法）
        outliers = self._detect_outliers(data)
        
        # 验证器检查
        valid_samples = total_samples
        for validator in self.validators:
            is_valid, error_msg = validator.validate(data)
            if not is_valid:
                self.logger.warning(f"验证失败: {error_msg}")
                # 这里可以根据具体验证器类型调整valid_samples
        
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
        
        self.logger.info(f"数据质量分析完成，有效率: {report.validity_rate:.2%}")
        return report
    
    def _detect_outliers(self, data: pd.DataFrame) -> Dict[str, int]:
        """使用IQR方法检测异常值"""
        
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
    """数据清洗器"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
    
    def clean(self, data: pd.DataFrame) -> pd.DataFrame:
        """清洗数据"""
        
        self.logger.info("开始数据清洗")
        cleaned_data = data.copy()
        
        # 处理缺失值
        cleaned_data = self._handle_missing_values(cleaned_data)
        
        # 处理重复值
        cleaned_data = self._handle_duplicates(cleaned_data)
        
        # 处理异常值
        cleaned_data = self._handle_outliers(cleaned_data)
        
        # 数据类型转换
        cleaned_data = self._convert_data_types(cleaned_data)
        
        # 数据标准化/归一化
        cleaned_data = self._normalize_data(cleaned_data)
        
        self.logger.info(f"数据清洗完成，从 {len(data)} 行清洗到 {len(cleaned_data)} 行")
        return cleaned_data
    
    def _handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """处理缺失值"""
        
        strategy = self.config.get('missing_value_strategy', 'drop')
        
        if strategy == 'drop':
            # 删除包含缺失值的行
            return data.dropna()
        elif strategy == 'fill_mean':
            # 用均值填充数值列
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())
            return data
        elif strategy == 'fill_median':
            # 用中位数填充数值列
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].median())
            return data
        elif strategy == 'forward_fill':
            # 前向填充
            return data.fillna(method='ffill')
        else:
            return data
    
    def _handle_duplicates(self, data: pd.DataFrame) -> pd.DataFrame:
        """处理重复值"""
        
        if self.config.get('remove_duplicates', True):
            return data.drop_duplicates()
        return data
    
    def _handle_outliers(self, data: pd.DataFrame) -> pd.DataFrame:
        """处理异常值"""
        
        strategy = self.config.get('outlier_strategy', 'none')
        
        if strategy == 'remove':
            # 使用IQR方法移除异常值
            for col in data.select_dtypes(include=[np.number]).columns:
                Q1 = data[col].quantile(0.25)
                Q3 = data[col].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                data = data[(data[col] >= lower_bound) & (data[col] <= upper_bound)]
        
        elif strategy == 'clip':
            # 截断异常值
            for col in data.select_dtypes(include=[np.number]).columns:
                Q1 = data[col].quantile(0.25)
                Q3 = data[col].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                data[col] = data[col].clip(lower=lower_bound, upper=upper_bound)
        
        return data
    
    def _convert_data_types(self, data: pd.DataFrame) -> pd.DataFrame:
        """转换数据类型"""
        
        type_mapping = self.config.get('type_mapping', {})
        
        for col, target_type in type_mapping.items():
            if col in data.columns:
                try:
                    data[col] = data[col].astype(target_type)
                except Exception as e:
                    self.logger.warning(f"无法转换列 '{col}' 到类型 '{target_type}': {e}")
        
        return data
    
    def _normalize_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """数据标准化/归一化"""
        
        normalization = self.config.get('normalization', 'none')
        
        if normalization == 'standardize':
            # Z-score标准化
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            data[numeric_cols] = (data[numeric_cols] - data[numeric_cols].mean()) / data[numeric_cols].std()
        
        elif normalization == 'min_max':
            # Min-Max归一化
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            data[numeric_cols] = (data[numeric_cols] - data[numeric_cols].min()) / (data[numeric_cols].max() - data[numeric_cols].min())
        
        return data

# 使用示例
if __name__ == "__main__":
    # 创建示例数据
    np.random.seed(42)
    data = pd.DataFrame({
        'feature1': np.random.normal(0, 1, 1000),
        'feature2': np.random.uniform(-1, 1, 1000),
        'feature3': np.random.exponential(2, 1000),
        'label': np.random.randint(0, 2, 1000)
    })
    
    # 添加一些缺失值和异常值
    data.loc[np.random.choice(1000, 50, replace=False), 'feature1'] = np.nan
    data.loc[np.random.choice(1000, 10, replace=False), 'feature2'] = 100  # 异常值
    
    # 创建验证器
    validators = [
        RangeValidator(-5, 5, 'feature1'),
        RangeValidator(-2, 2, 'feature2'),
        TypeValidator({'feature1': float, 'feature2': float, 'label': int})
    ]
    
    # 数据质量分析
    analyzer = DataQualityAnalyzer(validators)
    quality_report = analyzer.analyze(data)
    
    print("数据质量报告:")
    print(f"总样本数: {quality_report.total_samples}")
    print(f"有效样本数: {quality_report.valid_samples}")
    print(f"有效率: {quality_report.validity_rate:.2%}")
    print(f"缺失值: {quality_report.missing_values}")
    print(f"异常值: {quality_report.outliers}")
    print(f"重复值: {quality_report.duplicates}")
    
    # 数据清洗
    cleaner_config = {
        'missing_value_strategy': 'fill_mean',
        'remove_duplicates': True,
        'outlier_strategy': 'clip',
        'normalization': 'standardize'
    }
    
    cleaner = DataCleaner(cleaner_config)
    cleaned_data = cleaner.clean(data)
    
    print(f"\n清洗前数据形状: {data.shape}")
    print(f"清洗后数据形状: {cleaned_data.shape}")
    print(f"清洗后缺失值: {cleaned_data.isnull().sum().sum()}")
```

---

*本最佳实践指南提供了全面的开发和使用建议。更多详细信息请参考[完整文档](/)。*