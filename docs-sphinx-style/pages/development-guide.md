---
layout: default
title: Development Guide
description: 开发环境配置和贡献指南
nav_order: 19
parent: 开发与维护
permalink: /pages/development-guide/
---

# 开发指南 {#开发指南}

本文档为VIVTransformer项目的开发者提供详细的开发指导，包括代码规范、开发流程、测试指南和贡献方式。

## 📋 目录 {#目录}

- [开发环境设置](#开发环境设置)
- [代码规范](#代码规范)
- [项目结构](#项目结构)
- [开发流程](#开发流程)
- [测试指南](#测试指南)
- [文档编写](#文档编写)
- [性能优化](#性能优化)
- [贡献指南](#贡献指南)
- [发布流程](#发布流程)

## 开发环境设置 {#开发环境设置}

### 🛠️ 环境准备 {#环境准备}

#### 1. 基础环境 {#1-基础环境}

```bash
# 1. 克隆项目 {#1-克隆项目}
git clone https://github.com/your-username/VIVTransformer.git
cd VIVTransformer

# 2. 创建虚拟环境 {#2-创建虚拟环境}
conda create -n vivtransformer-dev python=3.9
conda activate vivtransformer-dev

# 3. 安装开发依赖 {#3-安装开发依赖}
pip install -r requirements-dev.txt

# 4. 安装项目（开发模式） {#4-安装项目-开发模式}
pip install -e .

# 5. 安装预提交钩子 {#5-安装预提交钩子}
pre-commit install
```

#### 2. 开发工具配置 {#2-开发工具配置}

**VS Code配置** (`.vscode/settings.json`):
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

**PyCharm配置**:
- 设置代码风格为Black
- 启用类型检查
- 配置测试运行器为pytest
- 设置导入优化

#### 3. Git配置 {#3-git配置}

```bash
# 设置用户信息 {#设置用户信息}
git config user.name "Your Name"
git config user.email "your.email@example.com"

# 设置默认分支 {#设置默认分支}
git config init.defaultBranch main

# 设置自动换行 {#设置自动换行}
git config core.autocrlf input  # Linux/Mac
git config core.autocrlf true   # Windows
```

### 📦 依赖管理 {#依赖管理}

#### requirements-dev.txt {#requirements-dev-txt}
```txt
# 基础依赖 {#基础依赖}
torch>=1.12.0
torchvision>=0.13.0
numpy>=1.21.0
pandas>=1.3.0
scipy>=1.7.0

# 开发工具 {#开发工具}
black>=22.0.0
isort>=5.10.0
flake8>=4.0.0
mypy>=0.950
pylint>=2.13.0
pre-commit>=2.17.0

# 测试工具 {#测试工具}
pytest>=7.0.0
pytest-cov>=3.0.0
pytest-mock>=3.7.0
pytest-xdist>=2.5.0
hypothesis>=6.40.0

# 文档工具 {#文档工具}
sphinx>=4.5.0
sphinx-rtd-theme>=1.0.0
myst-parser>=0.17.0

# 性能分析 {#性能分析}
line-profiler>=3.5.0
memory-profiler>=0.60.0
py-spy>=0.3.0

# 可视化 {#可视化}
matplotlib>=3.5.0
seaborn>=0.11.0
tensorboard>=2.8.0
wandb>=0.12.0

# 其他工具 {#其他工具}
tqdm>=4.62.0
click>=8.0.0
rich>=12.0.0
typer>=0.4.0
```

## 代码规范 {#代码规范}

### 🎨 代码风格 {#代码风格}

#### 1. Python代码规范 {#1-python代码规范}

**基本原则**:
- 遵循PEP 8标准
- 使用Black进行代码格式化
- 使用isort进行导入排序
- 行长度限制为88字符

**命名规范**:
```python
# 类名：大驼峰命名法 {#类名-大驼峰命名法}
class AttentionMechanism:
    pass

# 函数和变量：小写+下划线 {#函数和变量-小写-下划线}
def calculate_attention_weights():
    attention_scores = None
    return attention_scores

# 常量：全大写+下划线 {#常量-全大写-下划线}
MAX_SEQUENCE_LENGTH = 512
DEFAULT_HIDDEN_SIZE = 768

# 私有方法：前缀下划线 {#私有方法-前缀下划线}
def _internal_helper_function():
    pass

# 特殊方法：双下划线包围 {#特殊方法-双下划线包围}
def __init__(self):
    pass
```

**类型注解**:
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
    """处理注意力计算
    
    Args:
        query: 查询张量 [batch_size, seq_len, d_model]
        key: 键张量 [batch_size, seq_len, d_model]
        value: 值张量 [batch_size, seq_len, d_model]
        mask: 可选的掩码张量
        dropout_rate: Dropout比率
    
    Returns:
        输出张量和注意力权重的元组
    """
    # 实现代码...
    pass
```

#### 2. 文档字符串规范 {#2-文档字符串规范}

**Google风格文档字符串**:
```python
class MultiHeadAttention(torch.nn.Module):
    """多头注意力机制实现
    
    这个类实现了Transformer中的多头注意力机制，支持自注意力和交叉注意力。
    
    Attributes:
        d_model: 模型维度
        num_heads: 注意力头数量
        dropout_rate: Dropout比率
    
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
        """初始化多头注意力层
        
        Args:
            d_model: 模型维度，必须能被num_heads整除
            num_heads: 注意力头的数量
            dropout_rate: Dropout比率，范围[0, 1]
        
        Raises:
            ValueError: 当d_model不能被num_heads整除时
        """
        super().__init__()
        
        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) 必须能被 num_heads ({num_heads}) 整除")
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # 线性变换层
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
        """前向传播
        
        Args:
            query: 查询张量 [batch_size, seq_len, d_model]
            key: 键张量 [batch_size, seq_len, d_model]
            value: 值张量 [batch_size, seq_len, d_model]
            mask: 可选的注意力掩码 [batch_size, seq_len, seq_len]
        
        Returns:
            包含以下元素的元组:
            - output: 输出张量 [batch_size, seq_len, d_model]
            - attention_weights: 注意力权重 [batch_size, num_heads, seq_len, seq_len]
        
        Note:
            所有输入张量必须在同一设备上
        """
        # 实现代码...
        pass
```

#### 3. 错误处理规范 {#3-错误处理规范}

```python
class VIVTransformerError(Exception):
    """VIVTransformer基础异常类"""
    pass

class ConfigurationError(VIVTransformerError):
    """配置错误异常"""
    pass

class ModelError(VIVTransformerError):
    """模型相关错误异常"""
    pass

class DataError(VIVTransformerError):
    """数据相关错误异常"""
    pass

# 使用示例 {#使用示例}
def load_config(config_path: str) -> Dict:
    """加载配置文件
    
    Args:
        config_path: 配置文件路径
    
    Returns:
        配置字典
    
    Raises:
        ConfigurationError: 配置文件格式错误或缺少必要字段
        FileNotFoundError: 配置文件不存在
    """
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    except yaml.YAMLError as e:
        raise ConfigurationError(f"配置文件格式错误: {e}")
    
    # 验证必要字段
    required_fields = ['model', 'training', 'data']
    missing_fields = [field for field in required_fields if field not in config]
    
    if missing_fields:
        raise ConfigurationError(f"配置文件缺少必要字段: {missing_fields}")
    
    return config
```

### 🧪 代码质量工具 {#代码质量工具}

#### 1. 预提交钩子配置 {#1-预提交钩子配置}

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

#### 2. 配置文件 {#2-配置文件}

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

## 项目结构 {#项目结构}

### 📁 目录组织 {#目录组织}

```
VIVTransformer/
├── vivtransformer/              # 主要源代码
│   ├── __init__.py
│   ├── models/                  # 模型定义
│   │   ├── __init__.py
│   │   ├── base.py             # 基础模型类
│   │   ├── transformer.py      # Transformer实现
│   │   └── attention/          # 注意力机制
│   │       ├── __init__.py
│   │       ├── multi_head.py
│   │       ├── sparse.py
│   │       └── linear.py
│   ├── data/                   # 数据处理
│   │   ├── __init__.py
│   │   ├── datasets.py
│   │   ├── loaders.py
│   │   └── transforms.py
│   ├── training/               # 训练相关
│   │   ├── __init__.py
│   │   ├── trainer.py
│   │   ├── losses.py
│   │   └── optimizers.py
│   ├── utils/                  # 工具函数
│   │   ├── __init__.py
│   │   ├── config.py
│   │   ├── logging.py
│   │   └── metrics.py
│   └── cli/                    # 命令行接口
│       ├── __init__.py
│       ├── train.py
│       └── evaluate.py
├── tests/                      # 测试代码
│   ├── __init__.py
│   ├── conftest.py            # pytest配置
│   ├── unit/                  # 单元测试
│   │   ├── test_models.py
│   │   ├── test_attention.py
│   │   └── test_data.py
│   ├── integration/           # 集成测试
│   │   ├── test_training.py
│   │   └── test_pipeline.py
│   └── fixtures/              # 测试数据
│       ├── sample_data.pt
│       └── test_config.yaml
├── docs/                      # 文档
│   ├── source/
│   │   ├── conf.py
│   │   ├── index.rst
│   │   └── api/
│   └── build/
├── scripts/                   # 脚本文件
│   ├── setup_env.sh
│   ├── run_tests.sh
│   └── benchmark.py
├── config/                    # 配置文件
│   ├── default_config.yaml
│   ├── training_config.yaml
│   └── model_configs/
├── examples/                  # 示例代码
│   ├── basic_usage.py
│   ├── custom_attention.py
│   └── fine_tuning.py
├── requirements.txt           # 生产依赖
├── requirements-dev.txt       # 开发依赖
├── setup.py                   # 安装脚本
├── pyproject.toml            # 项目配置
├── README.md                 # 项目说明
├── CHANGELOG.md              # 变更日志
├── LICENSE                   # 许可证
└── .gitignore               # Git忽略文件
```

### 🏗️ 模块设计原则 {#模块设计原则}

#### 1. 单一职责原则 {#1-单一职责原则}
每个模块应该只有一个改变的理由：

```python
# ❌ 错误：一个类承担多个职责 {#错误-一个类承担多个职责}
class ModelTrainer:
    def __init__(self):
        pass
    
    def load_data(self):  # 数据加载职责
        pass
    
    def train_model(self):  # 训练职责
        pass
    
    def save_results(self):  # 结果保存职责
        pass

# ✅ 正确：职责分离 {#正确-职责分离}
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

#### 2. 开闭原则 {#2-开闭原则}
对扩展开放，对修改关闭：

```python
# 基础注意力接口 {#基础注意力接口}
from abc import ABC, abstractmethod

class AttentionMechanism(ABC):
    """注意力机制基类"""
    
    @abstractmethod
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向传播"""
        pass

# 具体实现 {#具体实现}
class MultiHeadAttention(AttentionMechanism):
    def forward(self, query, key, value, mask=None):
        # 多头注意力实现
        pass

class SparseAttention(AttentionMechanism):
    def forward(self, query, key, value, mask=None):
        # 稀疏注意力实现
        pass

# 注意力工厂 {#注意力工厂}
class AttentionFactory:
    _registry = {
        'multi_head': MultiHeadAttention,
        'sparse': SparseAttention,
    }
    
    @classmethod
    def create(cls, attention_type: str, **kwargs) -> AttentionMechanism:
        if attention_type not in cls._registry:
            raise ValueError(f"未知的注意力类型: {attention_type}")
        
        return cls._registry[attention_type](**kwargs)
    
    @classmethod
    def register(cls, name: str, attention_class: type):
        """注册新的注意力机制"""
        cls._registry[name] = attention_class
```

#### 3. 依赖注入 {#3-依赖注入}

```python
class VIVTransformer(torch.nn.Module):
    """VIVTransformer主模型"""
    
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
        
        # 构建模型层
        self._build_layers()
    
    def _build_layers(self):
        """构建模型层"""
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
```

## 开发流程 {#开发流程}

### 🔄 Git工作流 {#git工作流}

#### 1. 分支策略 {#1-分支策略}

```
main                    # 主分支，稳定版本
├── develop            # 开发分支，集成最新功能
├── feature/xxx        # 功能分支
├── bugfix/xxx         # 错误修复分支
├── hotfix/xxx         # 热修复分支
└── release/vx.x.x     # 发布分支
```

#### 2. 开发流程 {#2-开发流程}

```bash
# 1. 从develop创建功能分支 {#1-从develop创建功能分支}
git checkout develop
git pull origin develop
git checkout -b feature/new-attention-mechanism

# 2. 开发功能 {#2-开发功能}
# 编写代码... {#编写代码}
# 编写测试... {#编写测试}
# 更新文档... {#更新文档}

# 3. 提交代码 {#3-提交代码}
git add .
git commit -m "feat: 添加新的注意力机制

- 实现稀疏注意力算法
- 添加相应的单元测试
- 更新API文档

Closes #123"

# 4. 推送分支 {#4-推送分支}
git push origin feature/new-attention-mechanism

# 5. 创建Pull Request {#5-创建pull-request}
# 在GitHub上创建PR，请求合并到develop分支 {#在github上创建pr-请求合并到develop分支}

# 6. 代码审查和合并 {#6-代码审查和合并}
# 经过代码审查后，合并到develop分支 {#经过代码审查后-合并到develop分支}

# 7. 清理分支 {#7-清理分支}
git checkout develop
git pull origin develop
git branch -d feature/new-attention-mechanism
```

#### 3. 提交信息规范 {#3-提交信息规范}

**提交信息格式**:
```xml
<type>(<scope>): <subject>

<body>

<footer>
```

**类型说明**:
- `feat`: 新功能
- `fix`: 错误修复
- `docs`: 文档更新
- `style`: 代码格式化
- `refactor`: 代码重构
- `test`: 测试相关
- `chore`: 构建过程或辅助工具的变动

**示例**:
```bash
# 新功能 {#新功能}
git commit -m "feat(attention): 添加线性注意力机制

实现了O(n)复杂度的线性注意力算法，显著提升长序列处理效率。

- 添加LinearAttention类
- 实现高效的矩阵运算
- 添加性能基准测试

Closes #456"

# 错误修复 {#错误修复}
git commit -m "fix(training): 修复梯度累积bug

修复了在使用梯度累积时损失计算错误的问题。

Fixes #789"

# 文档更新 {#文档更新}
git commit -m "docs: 更新API文档

- 添加新注意力机制的使用示例
- 修正参数说明中的错误
- 更新性能对比表格"
```

### 🧪 测试驱动开发 {#测试驱动开发}

#### 1. TDD流程 {#1-tdd流程}

```python
# 1. 编写失败的测试 {#1-编写失败的测试}
def test_linear_attention_forward():
    """测试线性注意力前向传播"""
    batch_size, seq_len, d_model = 2, 10, 64
    
    attention = LinearAttention(d_model=d_model)
    
    query = torch.randn(batch_size, seq_len, d_model)
    key = torch.randn(batch_size, seq_len, d_model)
    value = torch.randn(batch_size, seq_len, d_model)
    
    output, weights = attention(query, key, value)
    
    # 检查输出形状
    assert output.shape == (batch_size, seq_len, d_model)
    assert weights.shape == (batch_size, seq_len, seq_len)
    
    # 检查注意力权重归一化
    assert torch.allclose(weights.sum(dim=-1), torch.ones(batch_size, seq_len))

# 2. 运行测试（应该失败） {#2-运行测试-应该失败}
# pytest tests/test_attention.py::test_linear_attention_forward {#pytest-tests-test-attention-py-test-linear-attention-forward}

# 3. 编写最小实现 {#3-编写最小实现}
class LinearAttention(torch.nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        # 最小实现...
    
    def forward(self, query, key, value):
        # 最小实现使测试通过
        pass

# 4. 运行测试（应该通过） {#4-运行测试-应该通过}
# 5. 重构代码 {#5-重构代码}
# 6. 重复循环 {#6-重复循环}
```

#### 2. 测试分层 {#2-测试分层}

```python
# 单元测试：测试单个函数或类 {#单元测试-测试单个函数或类}
class TestMultiHeadAttention:
    def test_init(self):
        """测试初始化"""
        attention = MultiHeadAttention(d_model=512, num_heads=8)
        assert attention.d_model == 512
        assert attention.num_heads == 8
        assert attention.d_k == 64
    
    def test_forward_shape(self):
        """测试前向传播输出形状"""
        attention = MultiHeadAttention(d_model=512, num_heads=8)
        
        batch_size, seq_len = 2, 10
        x = torch.randn(batch_size, seq_len, 512)
        
        output, weights = attention(x, x, x)
        
        assert output.shape == (batch_size, seq_len, 512)
        assert weights.shape == (batch_size, 8, seq_len, seq_len)
    
    @pytest.mark.parametrize("d_model,num_heads", [
        (512, 8),
        (768, 12),
        (1024, 16)
    ])
    def test_different_configs(self, d_model, num_heads):
        """测试不同配置"""
        attention = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        
        batch_size, seq_len = 2, 5
        x = torch.randn(batch_size, seq_len, d_model)
        
        output, weights = attention(x, x, x)
        
        assert output.shape == (batch_size, seq_len, d_model)
        assert weights.shape == (batch_size, num_heads, seq_len, seq_len)

# 集成测试：测试组件间交互 {#集成测试-测试组件间交互}
class TestVIVTransformerIntegration:
    def test_end_to_end_training(self):
        """测试端到端训练"""
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
        
        # 模拟训练数据
        batch_size, seq_len = 4, 20
        input_ids = torch.randint(0, config['vocab_size'], (batch_size, seq_len))
        labels = torch.randint(0, config['output_size'], (batch_size,))
        
        # 前向传播
        output = model(input_ids)
        loss = criterion(output, labels)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 验证训练步骤完成
        assert loss.item() > 0
        assert all(p.grad is not None for p in model.parameters() if p.requires_grad)
```

## 测试指南 {#测试指南}

### 🧪 测试框架 {#测试框架}

#### 1. pytest配置 {#1-pytest配置}

**conftest.py**:
```python
import pytest
import torch
import numpy as np
from typing import Dict, Any

@pytest.fixture
def device():
    """测试设备"""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@pytest.fixture
def sample_config() -> Dict[str, Any]:
    """示例配置"""
    return {
        'd_model': 256,
        'num_heads': 4,
        'num_layers': 2,
        'vocab_size': 1000,
        'output_size': 10,
        'dropout': 0.1,
        'max_seq_len': 512
    }

@pytest.fixture
def sample_data(device):
    """示例数据"""
    batch_size, seq_len, d_model = 2, 10, 256
    
    return {
        'input_ids': torch.randint(0, 1000, (batch_size, seq_len)).to(device),
        'attention_mask': torch.ones(batch_size, seq_len).to(device),
        'labels': torch.randint(0, 10, (batch_size,)).to(device),
        'embeddings': torch.randn(batch_size, seq_len, d_model).to(device)
    }

@pytest.fixture(autouse=True)
def set_random_seed():
    """设置随机种子以确保测试可重现"""
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)

@pytest.fixture
def mock_model(sample_config, device):
    """模拟模型"""
    from vivtransformer.models import VIVTransformer
    
    model = VIVTransformer(sample_config)
    model.to(device)
    model.eval()
    
    return model
```

#### 2. 测试工具函数 {#2-测试工具函数}

```python
# tests/utils.py {#tests-utils-py}
import torch
import numpy as np
from typing import Any, Dict, List

def assert_tensor_equal(actual: torch.Tensor, expected: torch.Tensor, rtol: float = 1e-5):
    """断言张量相等"""
    assert actual.shape == expected.shape, f"形状不匹配: {actual.shape} vs {expected.shape}"
    assert torch.allclose(actual, expected, rtol=rtol), "张量值不相等"

def assert_tensor_shape(tensor: torch.Tensor, expected_shape: tuple):
    """断言张量形状"""
    assert tensor.shape == expected_shape, f"形状不匹配: {tensor.shape} vs {expected_shape}"

def assert_no_nan_inf(tensor: torch.Tensor):
    """断言张量不包含NaN或Inf"""
    assert not torch.isnan(tensor).any(), "张量包含NaN"
    assert not torch.isinf(tensor).any(), "张量包含Inf"

def assert_gradients_exist(model: torch.nn.Module):
    """断言模型参数有梯度"""
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"参数 {name} 没有梯度"

def count_parameters(model: torch.nn.Module) -> int:
    """计算模型参数数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class ModelTester:
    """模型测试辅助类"""
    
    def __init__(self, model: torch.nn.Module, device: torch.device):
        self.model = model
        self.device = device
    
    def test_forward_pass(self, input_data: Dict[str, torch.Tensor]):
        """测试前向传播"""
        self.model.eval()
        
        with torch.no_grad():
            output = self.model(**input_data)
        
        # 检查输出
        assert_no_nan_inf(output)
        
        return output
    
    def test_backward_pass(self, input_data: Dict[str, torch.Tensor], target: torch.Tensor):
        """测试反向传播"""
        self.model.train()
        
        # 前向传播
        output = self.model(**input_data)
        
        # 计算损失
        criterion = torch.nn.CrossEntropyLoss()
        loss = criterion(output, target)
        
        # 反向传播
        loss.backward()
        
        # 检查梯度
        assert_gradients_exist(self.model)
        
        return loss
    
    def test_model_determinism(self, input_data: Dict[str, torch.Tensor], num_runs: int = 3):
        """测试模型确定性"""
        self.model.eval()
        
        outputs = []
        
        for _ in range(num_runs):
            with torch.no_grad():
                output = self.model(**input_data)
                outputs.append(output.clone())
        
        # 检查所有输出是否相同
        for i in range(1, num_runs):
            assert_tensor_equal(outputs[0], outputs[i])
    
    def test_model_device_consistency(self, input_data: Dict[str, torch.Tensor]):
        """测试模型设备一致性"""
        # 检查模型参数设备
        model_device = next(self.model.parameters()).device
        assert model_device == self.device
        
        # 检查输入数据设备
        for key, tensor in input_data.items():
            assert tensor.device == self.device, f"{key} 不在正确设备上"
```

#### 3. 性能测试 {#3-性能测试}

```python
# tests/test_performance.py {#tests-test-performance-py}
import time
import pytest
import torch
from memory_profiler import profile

class TestPerformance:
    """性能测试"""
    
    @pytest.mark.slow
    def test_inference_speed(self, mock_model, sample_data, device):
        """测试推理速度"""
        model = mock_model
        input_data = sample_data['embeddings']
        
        # 预热
        for _ in range(10):
            with torch.no_grad():
                _ = model(input_data)
        
        # 同步GPU
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        # 测量时间
        start_time = time.perf_counter()
        
        num_runs = 100
        with torch.no_grad():
            for _ in range(num_runs):
                _ = model(input_data)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        end_time = time.perf_counter()
        
        avg_time = (end_time - start_time) / num_runs
        throughput = input_data.size(0) / avg_time  # samples/second
        
        print(f"平均推理时间: {avg_time*1000:.2f}ms")
        print(f"吞吐量: {throughput:.2f} samples/second")
        
        # 性能断言（根据实际情况调整）
        assert avg_time < 0.1, f"推理时间过长: {avg_time:.4f}s"
    
    @pytest.mark.slow
    def test_memory_usage(self, mock_model, sample_data, device):
        """测试内存使用"""
        if device.type != 'cuda':
            pytest.skip("仅在CUDA设备上测试GPU内存")
        
        model = mock_model
        input_data = sample_data['embeddings']
        
        # 清理内存
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        # 测量内存使用
        with torch.no_grad():
            output = model(input_data)
        
        peak_memory = torch.cuda.max_memory_allocated() / 1024**2  # MB
        current_memory = torch.cuda.memory_allocated() / 1024**2  # MB
        
        print(f"峰值内存使用: {peak_memory:.2f}MB")
        print(f"当前内存使用: {current_memory:.2f}MB")
        
        # 内存断言（根据实际情况调整）
        assert peak_memory < 1000, f"内存使用过多: {peak_memory:.2f}MB"
    
    @pytest.mark.parametrize("batch_size", [1, 4, 8, 16])
    def test_batch_scaling(self, sample_config, device, batch_size):
        """测试批次大小扩展性"""
        from vivtransformer.models import VIVTransformer
        
        model = VIVTransformer(sample_config).to(device)
        
        seq_len, d_model = 50, sample_config['d_model']
        input_data = torch.randn(batch_size, seq_len, d_model).to(device)
        
        start_time = time.perf_counter()
        
        with torch.no_grad():
            output = model(input_data)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        end_time = time.perf_counter()
        
        time_per_sample = (end_time - start_time) / batch_size
        
        print(f"批次大小 {batch_size}: {time_per_sample*1000:.2f}ms/sample")
        
        # 检查输出形状
        expected_shape = (batch_size, sample_config['output_size'])
        assert output.shape == expected_shape
```

### 📊 测试覆盖率 {#测试覆盖率}

```bash
# 运行测试并生成覆盖率报告 {#运行测试并生成覆盖率报告}
pytest --cov=vivtransformer --cov-report=html --cov-report=term-missing

# 查看覆盖率报告 {#查看覆盖率报告}
open htmlcov/index.html

# 设置覆盖率阈值 {#设置覆盖率阈值}
pytest --cov=vivtransformer --cov-fail-under=80
```

## 性能优化 {#性能优化}

### ⚡ 性能分析工具 {#性能分析工具}

#### 1. 代码性能分析 {#1-代码性能分析}

```python
# 使用line_profiler {#使用line-profiler}
@profile
def attention_computation(query, key, value):
    """注意力计算函数"""
    # 计算注意力分数
    scores = torch.matmul(query, key.transpose(-2, -1))
    
    # 缩放
    scores = scores / math.sqrt(query.size(-1))
    
    # Softmax
    attention_weights = torch.softmax(scores, dim=-1)
    
    # 应用到值
    output = torch.matmul(attention_weights, value)
    
    return output, attention_weights

# 运行分析 {#运行分析}
# kernprof -l -v script.py {#kernprof-l-v-script-py}
```

#### 2. 内存分析 {#2-内存分析}

```python
# 使用memory_profiler {#使用memory-profiler}
from memory_profiler import profile

@profile
def train_step(model, data, target, optimizer, criterion):
    """训练步骤"""
    optimizer.zero_grad()
    
    output = model(data)
    loss = criterion(output, target)
    
    loss.backward()
    optimizer.step()
    
    return loss.item()

# 运行分析 {#运行分析}
# python -m memory_profiler script.py {#python-m-memory-profiler-script-py}
```

#### 3. PyTorch Profiler {#3-pytorch-profiler}

```python
import torch.profiler

def profile_model(model, input_data, num_steps=10):
    """使用PyTorch Profiler分析模型"""
    
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(
            wait=1,
            warmup=1,
            active=3,
            repeat=2
        ),
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/profiler'),
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        
        for step in range(num_steps):
            with torch.no_grad():
                output = model(input_data)
            
            prof.step()
    
    # 打印关键统计信息
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    
    return prof

# 使用示例 {#使用示例}
prof = profile_model(model, sample_input)

# 在TensorBoard中查看结果 {#在tensorboard中查看结果}
# tensorboard --logdir=./log/profiler {#tensorboard-logdir-log-profiler}
```

### 🚀 优化技巧 {#优化技巧}

#### 1. 计算优化 {#1-计算优化}

```python
# 使用torch.jit.script优化 {#使用torch-jit-script优化}
@torch.jit.script
def optimized_attention(query, key, value, mask=None):
    """优化的注意力计算"""
    # 批量矩阵乘法
    scores = torch.bmm(query, key.transpose(1, 2))
    
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    
    # 使用更快的softmax实现
    attention_weights = torch.nn.functional.softmax(scores, dim=-1)
    
    output = torch.bmm(attention_weights, value)
    
    return output, attention_weights

# 使用混合精度训练 {#使用混合精度训练}
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for data, target in dataloader:
    optimizer.zero_grad()
    
    with autocast():
        output = model(data)
        loss = criterion(output, target)
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

#### 2. 内存优化 {#2-内存优化}

```python
# 梯度检查点 {#梯度检查点}
from torch.utils.checkpoint import checkpoint

class MemoryEfficientTransformerLayer(torch.nn.Module):
    def __init__(self, attention, feed_forward):
        super().__init__()
        self.attention = attention
        self.feed_forward = feed_forward
    
    def forward(self, x):
        # 使用梯度检查点节省内存
        x = checkpoint(self.attention, x)
        x = checkpoint(self.feed_forward, x)
        return x

# 动态批次大小 {#动态批次大小}
class DynamicBatchSampler:
    def __init__(self, dataset, max_tokens=4096):
        self.dataset = dataset
        self.max_tokens = max_tokens
    
    def __iter__(self):
        batch = []
        current_tokens = 0
        
        for idx in range(len(self.dataset)):
            item_tokens = len(self.dataset[idx]['input_ids'])
            
            if current_tokens + item_tokens > self.max_tokens and batch:
                yield batch
                batch = []
                current_tokens = 0
            
            batch.append(idx)
            current_tokens += item_tokens
        
        if batch:
            yield batch
```

---

**💡 开发提示**:
1. 始终编写测试先于实现代码
2. 保持代码简洁和可读性
3. 定期进行代码审查
4. 使用类型注解提高代码质量
5. 关注性能但不过早优化
6. 及时更新文档和注释

**🔧 工具推荐**:
- IDE: VS Code, PyCharm
- 调试: pdb, ipdb
- 性能分析: line_profiler, py-spy
- 可视化: TensorBoard, Weights & Biases
- 协作: GitHub, GitLab

---

*需要帮助？查看 [FAQ](faq) 或 [故障排除](troubleshooting) 页面。*
