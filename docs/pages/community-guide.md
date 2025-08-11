# 🤝 社区贡献指南

> 欢迎加入VIVTransformer开源社区！本指南将帮助您了解如何参与项目贡献

---

## 📋 目录

- [🌟 欢迎加入](#-欢迎加入)
- [🎯 贡献方式](#-贡献方式)
- [📝 代码贡献](#-代码贡献)
- [📚 文档贡献](#-文档贡献)
- [🐛 问题报告](#-问题报告)
- [💡 功能建议](#-功能建议)
- [🧪 测试贡献](#-测试贡献)
- [🎨 设计贡献](#-设计贡献)
- [📊 数据贡献](#-数据贡献)
- [🏆 贡献者认可](#-贡献者认可)
- [📞 联系我们](#-联系我们)

---

## 🌟 欢迎加入

### 项目愿景

VIVTransformer致力于成为**涡激振动分析领域的开源标杆**，通过结合深度学习与物理建模，为科研和工程应用提供强大的工具。

### 社区价值观

- **🔬 科学严谨**: 基于扎实的理论基础和实验验证
- **🤝 开放合作**: 欢迎不同背景的贡献者
- **📈 持续改进**: 不断优化性能和用户体验
- **🎓 知识共享**: 促进学术交流和技术传播
- **🌍 包容多元**: 尊重不同观点和文化背景

### 贡献者类型

| 贡献者类型 | 技能要求 | 贡献内容 | 时间投入 |
|------------|----------|----------|----------|
| **核心开发者** | 深度学习、流体力学 | 核心算法、架构设计 | 20+ 小时/周 |
| **功能开发者** | Python、机器学习 | 新功能、工具开发 | 10-20 小时/周 |
| **文档维护者** | 技术写作、教学 | 文档、教程编写 | 5-15 小时/周 |
| **测试工程师** | 软件测试、质量保证 | 测试用例、性能测试 | 5-10 小时/周 |
| **社区管理者** | 沟通协调、项目管理 | 社区运营、活动组织 | 5-10 小时/周 |
| **偶尔贡献者** | 任意技能 | 问题报告、小修复 | 1-5 小时/周 |

---

## 🎯 贡献方式

### 快速开始

#### 1. 环境准备

```bash
# 1. Fork项目到您的GitHub账户
# 2. 克隆您的Fork
git clone https://github.com/YOUR_USERNAME/VIVTransformer.git
cd VIVTransformer

# 3. 添加上游仓库
git remote add upstream https://github.com/ORIGINAL_OWNER/VIVTransformer.git

# 4. 创建开发环境
conda create -n vivtransformer-dev python=3.9
conda activate vivtransformer-dev

# 5. 安装开发依赖
pip install -e ".[dev]"
pip install pre-commit
pre-commit install
```

#### 2. 开发流程

```bash
# 1. 同步最新代码
git checkout main
git pull upstream main

# 2. 创建功能分支
git checkout -b feature/your-feature-name

# 3. 进行开发
# ... 编写代码 ...

# 4. 运行测试
pytest tests/
python -m pytest tests/ --cov=vivtransformer

# 5. 代码格式化
black vivtransformer/
isort vivtransformer/
flake8 vivtransformer/

# 6. 提交更改
git add .
git commit -m "feat: add your feature description"

# 7. 推送到您的Fork
git push origin feature/your-feature-name

# 8. 创建Pull Request
```

### 贡献指南

#### 代码风格

**Python代码规范**:
```python
# 使用类型注解
def calculate_attention_weights(
    query: torch.Tensor,
    key: torch.Tensor,
    mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """计算注意力权重
    
    Args:
        query: 查询张量 [batch_size, seq_len, d_model]
        key: 键张量 [batch_size, seq_len, d_model]
        mask: 可选的掩码张量 [batch_size, seq_len, seq_len]
    
    Returns:
        注意力权重张量 [batch_size, seq_len, seq_len]
    """
    # 计算注意力分数
    scores = torch.matmul(query, key.transpose(-2, -1))
    
    # 应用掩码
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    
    # 应用softmax
    attention_weights = torch.softmax(scores, dim=-1)
    
    return attention_weights
```

**文档字符串规范**:
```python
class VIVTransformer(nn.Module):
    """VIV Transformer模型
    
    这是一个专门用于涡激振动分析的Transformer模型，集成了
    多种注意力机制和物理约束损失函数。
    
    Args:
        d_model: 模型维度
        n_heads: 注意力头数
        n_layers: 层数
        attention_type: 注意力机制类型
        loss_config: 损失函数配置
    
    Example:
        >>> model = VIVTransformer(
        ...     d_model=512,
        ...     n_heads=8,
        ...     n_layers=6,
        ...     attention_type='external'
        ... )
        >>> output = model(input_tensor)
    
    Note:
        模型支持多种注意力机制，详见attention_mechanisms模块。
    """
```

#### 提交信息规范

**提交信息格式**:
```
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
```
feat(attention): add external attention mechanism

- Implement external attention as described in paper
- Add configuration options for external attention
- Include unit tests for new attention mechanism
- Update documentation with usage examples

Closes #123
```

---

## 📝 代码贡献

### 核心模块开发

#### 1. 注意力机制开发

**新注意力机制模板**:
```python
# vivtransformer/attention_mechanisms/your_attention.py

import torch
import torch.nn as nn
from typing import Optional, Tuple
from .base_attention import BaseAttention

class YourAttention(BaseAttention):
    """您的注意力机制实现
    
    基于论文: [Paper Title] (Author et al., Year)
    论文链接: https://arxiv.org/abs/XXXX.XXXXX
    
    Args:
        d_model: 模型维度
        n_heads: 注意力头数
        dropout: Dropout概率
        **kwargs: 其他参数
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        dropout: float = 0.1,
        **kwargs
    ):
        super().__init__(d_model, n_heads, dropout)
        
        # 您的特定参数
        self.your_param = kwargs.get('your_param', default_value)
        
        # 您的网络层
        self.your_layer = nn.Linear(d_model, d_model)
    
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
            mask: 掩码张量 [batch_size, seq_len, seq_len]
        
        Returns:
            output: 输出张量 [batch_size, seq_len, d_model]
            attention_weights: 注意力权重 [batch_size, n_heads, seq_len, seq_len]
        """
        batch_size, seq_len, d_model = query.shape
        
        # 您的注意力计算逻辑
        # ...
        
        return output, attention_weights
    
    @staticmethod
    def get_config_template() -> dict:
        """返回配置模板"""
        return {
            'type': 'your_attention',
            'n_heads': 8,
            'dropout': 0.1,
            'your_param': 'default_value'
        }
```

**注册新注意力机制**:
```python
# vivtransformer/attention_mechanisms/__init__.py

from .your_attention import YourAttention

# 注册到工厂
ATTENTION_REGISTRY['your_attention'] = YourAttention
```

#### 2. 损失函数开发

**新损失函数模板**:
```python
# vivtransformer/losses/your_loss.py

import torch
import torch.nn as nn
from typing import Dict, Any
from .base_loss import BaseLoss

class YourLoss(BaseLoss):
    """您的损失函数实现
    
    Args:
        weight: 损失权重
        **kwargs: 其他参数
    """
    
    def __init__(self, weight: float = 1.0, **kwargs):
        super().__init__(weight)
        self.your_param = kwargs.get('your_param', default_value)
    
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """计算损失
        
        Args:
            predictions: 预测值
            targets: 目标值
            **kwargs: 其他输入
        
        Returns:
            损失值
        """
        # 您的损失计算逻辑
        loss = your_loss_calculation(predictions, targets)
        
        return self.weight * loss
    
    def get_metrics(self) -> Dict[str, float]:
        """返回相关指标"""
        return {
            'your_loss': self.last_loss_value,
            'your_metric': self.calculate_your_metric()
        }
```

#### 3. 数据处理模块

**新数据处理器模板**:
```python
# vivtransformer/data/processors/your_processor.py

import numpy as np
from typing import Dict, Any, Tuple
from .base_processor import BaseProcessor

class YourDataProcessor(BaseProcessor):
    """您的数据处理器
    
    Args:
        config: 处理器配置
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.your_param = config.get('your_param', default_value)
    
    def process(
        self,
        data: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """处理数据
        
        Args:
            data: 输入数据
        
        Returns:
            processed_data: 处理后的数据
            metadata: 处理元数据
        """
        # 您的数据处理逻辑
        processed_data = your_processing_function(data)
        
        metadata = {
            'original_shape': data.shape,
            'processed_shape': processed_data.shape,
            'processing_time': self.processing_time
        }
        
        return processed_data, metadata
```

### 性能优化贡献

#### 1. CUDA核函数优化

```cpp
// vivtransformer/csrc/attention_cuda.cu

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void optimized_attention_kernel(
    const float* query,
    const float* key,
    const float* value,
    float* output,
    int batch_size,
    int seq_len,
    int d_model
) {
    // 您的优化CUDA核函数
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < batch_size * seq_len * d_model) {
        // 计算逻辑
    }
}

torch::Tensor optimized_attention_cuda(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value
) {
    // CUDA函数包装
    auto output = torch::zeros_like(query);
    
    const int threads = 256;
    const int blocks = (query.numel() + threads - 1) / threads;
    
    optimized_attention_kernel<<<blocks, threads>>>(
        query.data_ptr<float>(),
        key.data_ptr<float>(),
        value.data_ptr<float>(),
        output.data_ptr<float>(),
        query.size(0),
        query.size(1),
        query.size(2)
    );
    
    return output;
}
```

#### 2. 内存优化

```python
# vivtransformer/utils/memory_optimization.py

import torch
from typing import Iterator, Tuple

class GradientCheckpointing:
    """梯度检查点优化"""
    
    @staticmethod
    def checkpoint_sequential(
        functions: Iterator[torch.nn.Module],
        segments: int,
        input: torch.Tensor
    ) -> torch.Tensor:
        """序列化梯度检查点"""
        def run_function(start, end, functions):
            def forward(input):
                for j in range(start, end + 1):
                    input = functions[j](input)
                return input
            return forward
        
        if segments == 1:
            return torch.utils.checkpoint.checkpoint(
                run_function(0, len(functions) - 1, functions),
                input
            )
        
        # 分段检查点
        segment_size = len(functions) // segments
        for i in range(segments):
            start_idx = i * segment_size
            end_idx = (i + 1) * segment_size - 1
            if i == segments - 1:
                end_idx = len(functions) - 1
            
            input = torch.utils.checkpoint.checkpoint(
                run_function(start_idx, end_idx, functions),
                input
            )
        
        return input
```

---

## 📚 文档贡献

### 文档类型

#### 1. API文档

**函数文档模板**:
```python
def complex_function(
    param1: torch.Tensor,
    param2: Optional[str] = None,
    param3: Dict[str, Any] = None
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """复杂函数的详细文档
    
    这个函数执行复杂的计算，包括多个步骤和条件判断。
    适用于需要精确控制计算过程的场景。
    
    Args:
        param1: 输入张量，形状为 [batch_size, seq_len, d_model]
            - batch_size: 批次大小，通常为32-512
            - seq_len: 序列长度，支持可变长度
            - d_model: 模型维度，必须是8的倍数
        param2: 可选的字符串参数，支持以下值:
            - 'mode1': 使用标准计算模式
            - 'mode2': 使用优化计算模式
            - None: 自动选择模式
        param3: 配置字典，包含以下键值:
            - 'threshold': float, 阈值参数 (默认: 0.5)
            - 'iterations': int, 迭代次数 (默认: 10)
            - 'verbose': bool, 是否输出详细信息 (默认: False)
    
    Returns:
        result: 计算结果张量，形状与param1相同
        metrics: 计算指标字典，包含:
            - 'computation_time': 计算时间(秒)
            - 'memory_usage': 内存使用量(MB)
            - 'convergence_score': 收敛分数(0-1)
    
    Raises:
        ValueError: 当param1的维度不正确时
        RuntimeError: 当计算过程中出现数值不稳定时
        MemoryError: 当内存不足时
    
    Example:
        基本使用:
        
        >>> import torch
        >>> input_tensor = torch.randn(32, 128, 512)
        >>> result, metrics = complex_function(input_tensor)
        >>> print(f"计算时间: {metrics['computation_time']:.3f}s")
        
        高级配置:
        
        >>> config = {
        ...     'threshold': 0.8,
        ...     'iterations': 20,
        ...     'verbose': True
        ... }
        >>> result, metrics = complex_function(
        ...     input_tensor,
        ...     param2='mode2',
        ...     param3=config
        ... )
    
    Note:
        - 该函数在GPU上运行时性能最佳
        - 对于大型输入，建议使用梯度检查点
        - 支持混合精度训练
    
    See Also:
        - :func:`related_function`: 相关函数
        - :class:`RelatedClass`: 相关类
        - :doc:`../tutorials/advanced_usage`: 高级使用教程
    
    References:
        [1] Author et al. "Paper Title". Journal Name, 2024.
        [2] https://example.com/documentation
    """
```

#### 2. 教程文档

**教程结构模板**:
```markdown
# 教程标题

> 简短描述教程内容和目标读者

## 学习目标

完成本教程后，您将能够:
- [ ] 目标1
- [ ] 目标2
- [ ] 目标3

## 前置要求

- Python 3.8+
- PyTorch 1.9+
- 基础的深度学习知识

## 步骤1: 环境准备

### 安装依赖

```bash
pip install vivtransformer
```

### 验证安装

```python
import vivtransformer
print(f"VIVTransformer版本: {vivtransformer.__version__}")
```

## 步骤2: 数据准备

### 数据格式

```python
# 数据应该是以下格式
data = {
    'input': torch.tensor(...),  # [batch_size, seq_len, features]
    'target': torch.tensor(...), # [batch_size, seq_len, targets]
    'metadata': {...}            # 元数据字典
}
```

### 数据加载

```python
from vivtransformer.data import VIVDataLoader

# 创建数据加载器
loader = VIVDataLoader(
    data_path='path/to/data',
    batch_size=32,
    shuffle=True
)

# 迭代数据
for batch in loader:
    input_data = batch['input']
    target_data = batch['target']
    # 处理数据...
```

## 步骤3: 模型配置

### 基础配置

```python
from vivtransformer import VIVTransformer

# 创建模型
model = VIVTransformer(
    d_model=512,
    n_heads=8,
    n_layers=6,
    attention_type='external'
)
```

### 高级配置

```yaml
# config.yaml
model:
  d_model: 512
  n_heads: 8
  n_layers: 6
  attention_type: 'external'
  
loss:
  type: 'svd_enhanced'
  weights:
    reconstruction: 1.0
    singular_value: 0.5
    orthogonality: 0.3

training:
  learning_rate: 1e-4
  batch_size: 32
  epochs: 100
```

```python
from vivtransformer.config import load_config

# 从配置文件加载
config = load_config('config.yaml')
model = VIVTransformer.from_config(config.model)
```

## 常见问题

### Q: 如何处理内存不足？

A: 可以尝试以下方法:
1. 减小batch_size
2. 使用梯度累积
3. 启用梯度检查点

```python
# 梯度累积示例
accumulation_steps = 4
for i, batch in enumerate(dataloader):
    loss = model(batch)
    loss = loss / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### Q: 如何选择注意力机制？

A: 不同注意力机制的特点:
- `scaled_dot_product`: 标准注意力，计算效率高
- `external`: 外部注意力，适合长序列
- `se_attention`: 通道注意力，适合特征选择

## 总结

本教程介绍了...

## 下一步

- [ ] 阅读高级配置教程
- [ ] 尝试自定义注意力机制
- [ ] 参与社区讨论
```

#### 3. 示例代码

**完整示例模板**:
```python
#!/usr/bin/env python3
"""
示例: VIVTransformer基础使用

这个示例展示了如何使用VIVTransformer进行涡激振动预测。
包括数据准备、模型训练和结果可视化的完整流程。

作者: [Your Name]
日期: 2024-01-01
版本: 1.0.0
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple

# VIVTransformer相关导入
from vivtransformer import VIVTransformer
from vivtransformer.data import VIVDataset, VIVDataLoader
from vivtransformer.losses import SVDEnhancedLoss
from vivtransformer.utils import set_seed, get_device
from vivtransformer.visualization import plot_attention_weights, plot_predictions


def main():
    """主函数"""
    # 设置随机种子
    set_seed(42)
    
    # 获取设备
    device = get_device()
    print(f"使用设备: {device}")
    
    # 1. 数据准备
    print("\n=== 数据准备 ===")
    train_dataset, val_dataset = prepare_data()
    
    train_loader = VIVDataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4
    )
    
    val_loader = VIVDataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4
    )
    
    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(val_dataset)}")
    
    # 2. 模型创建
    print("\n=== 模型创建 ===")
    model = create_model(device)
    print(f"模型参数量: {count_parameters(model):,}")
    
    # 3. 训练配置
    print("\n=== 训练配置 ===")
    criterion = SVDEnhancedLoss(
        reconstruction_weight=1.0,
        singular_value_weight=0.5,
        orthogonality_weight=0.3
    )
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=1e-5
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=100,
        eta_min=1e-6
    )
    
    # 4. 模型训练
    print("\n=== 开始训练 ===")
    train_losses, val_losses = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        epochs=100,
        device=device
    )
    
    # 5. 结果评估
    print("\n=== 结果评估 ===")
    evaluate_model(model, val_loader, device)
    
    # 6. 可视化
    print("\n=== 结果可视化 ===")
    visualize_results(model, val_loader, device)
    
    # 7. 模型保存
    print("\n=== 保存模型 ===")
    save_model(model, 'vivtransformer_trained.pth')
    
    print("\n训练完成！")


def prepare_data() -> Tuple[VIVDataset, VIVDataset]:
    """准备训练和验证数据"""
    # 生成示例数据（实际使用时替换为真实数据）
    np.random.seed(42)
    
    # 模拟圆柱绕流数据
    n_samples = 1000
    seq_len = 128
    n_features = 64
    
    # 输入特征：速度场、压力场等
    X = np.random.randn(n_samples, seq_len, n_features).astype(np.float32)
    
    # 目标：位移、速度等
    y = np.random.randn(n_samples, seq_len, 2).astype(np.float32)
    
    # 添加一些物理相关性
    for i in range(n_samples):
        # 模拟周期性振动
        t = np.linspace(0, 10, seq_len)
        frequency = np.random.uniform(0.1, 2.0)
        amplitude = np.random.uniform(0.5, 2.0)
        
        y[i, :, 0] = amplitude * np.sin(2 * np.pi * frequency * t)
        y[i, :, 1] = amplitude * np.cos(2 * np.pi * frequency * t)
    
    # 分割数据集
    split_idx = int(0.8 * n_samples)
    
    train_dataset = VIVDataset(
        X[:split_idx],
        y[:split_idx],
        transform=None
    )
    
    val_dataset = VIVDataset(
        X[split_idx:],
        y[split_idx:],
        transform=None
    )
    
    return train_dataset, val_dataset


def create_model(device: torch.device) -> VIVTransformer:
    """创建VIVTransformer模型"""
    model = VIVTransformer(
        input_dim=64,
        output_dim=2,
        d_model=512,
        n_heads=8,
        n_layers=6,
        attention_type='external',
        dropout=0.1,
        max_seq_len=128
    )
    
    return model.to(device)


def count_parameters(model: nn.Module) -> int:
    """计算模型参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_model(
    model: nn.Module,
    train_loader: VIVDataLoader,
    val_loader: VIVDataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    epochs: int,
    device: torch.device
) -> Tuple[List[float], List[float]]:
    """训练模型"""
    train_losses = []
    val_losses = []
    
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        
        for batch_idx, batch in enumerate(train_loader):
            inputs = batch['input'].to(device)
            targets = batch['target'].to(device)
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.6f}")
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                inputs = batch['input'].to(device)
                targets = batch['target'].to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item()
        
        # 计算平均损失
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        # 学习率调度
        scheduler.step()
        
        print(f"Epoch {epoch+1}/{epochs}:")
        print(f"  训练损失: {avg_train_loss:.6f}")
        print(f"  验证损失: {avg_val_loss:.6f}")
        print(f"  学习率: {scheduler.get_last_lr()[0]:.8f}")
        
        # 早停检查
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # 保存最佳模型
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"早停触发，在第{epoch+1}轮停止训练")
            break
    
    return train_losses, val_losses


def evaluate_model(
    model: nn.Module,
    val_loader: VIVDataLoader,
    device: torch.device
) -> Dict[str, float]:
    """评估模型性能"""
    model.eval()
    
    total_mse = 0.0
    total_mae = 0.0
    total_samples = 0
    
    predictions = []
    targets = []
    
    with torch.no_grad():
        for batch in val_loader:
            inputs = batch['input'].to(device)
            batch_targets = batch['target'].to(device)
            
            outputs = model(inputs)
            
            # 计算指标
            mse = torch.mean((outputs - batch_targets) ** 2)
            mae = torch.mean(torch.abs(outputs - batch_targets))
            
            total_mse += mse.item() * inputs.size(0)
            total_mae += mae.item() * inputs.size(0)
            total_samples += inputs.size(0)
            
            # 收集预测结果
            predictions.append(outputs.cpu().numpy())
            targets.append(batch_targets.cpu().numpy())
    
    # 计算平均指标
    avg_mse = total_mse / total_samples
    avg_mae = total_mae / total_samples
    avg_rmse = np.sqrt(avg_mse)
    
    # 计算R²
    predictions = np.concatenate(predictions, axis=0)
    targets = np.concatenate(targets, axis=0)
    
    ss_res = np.sum((targets - predictions) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    r2_score = 1 - (ss_res / ss_tot)
    
    metrics = {
        'MSE': avg_mse,
        'MAE': avg_mae,
        'RMSE': avg_rmse,
        'R²': r2_score
    }
    
    print("评估结果:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.6f}")
    
    return metrics


def visualize_results(
    model: nn.Module,
    val_loader: VIVDataLoader,
    device: torch.device
) -> None:
    """可视化结果"""
    model.eval()
    
    # 获取一个批次的数据
    batch = next(iter(val_loader))
    inputs = batch['input'].to(device)
    targets = batch['target'].to(device)
    
    with torch.no_grad():
        outputs = model(inputs)
        
        # 如果模型返回注意力权重
        if hasattr(model, 'get_attention_weights'):
            attention_weights = model.get_attention_weights()
    
    # 转换为numpy
    inputs_np = inputs.cpu().numpy()
    outputs_np = outputs.cpu().numpy()
    targets_np = targets.cpu().numpy()
    
    # 创建图形
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 绘制预测vs真实值
    sample_idx = 0
    time_steps = np.arange(outputs_np.shape[1])
    
    # X方向位移
    axes[0, 0].plot(time_steps, targets_np[sample_idx, :, 0], 'b-', label='真实值', linewidth=2)
    axes[0, 0].plot(time_steps, outputs_np[sample_idx, :, 0], 'r--', label='预测值', linewidth=2)
    axes[0, 0].set_title('X方向位移预测')
    axes[0, 0].set_xlabel('时间步')
    axes[0, 0].set_ylabel('位移')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Y方向位移
    axes[0, 1].plot(time_steps, targets_np[sample_idx, :, 1], 'b-', label='真实值', linewidth=2)
    axes[0, 1].plot(time_steps, outputs_np[sample_idx, :, 1], 'r--', label='预测值', linewidth=2)
    axes[0, 1].set_title('Y方向位移预测')
    axes[0, 1].set_xlabel('时间步')
    axes[0, 1].set_ylabel('位移')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # 误差分析
    error_x = np.abs(targets_np[sample_idx, :, 0] - outputs_np[sample_idx, :, 0])
    error_y = np.abs(targets_np[sample_idx, :, 1] - outputs_np[sample_idx, :, 1])
    
    axes[1, 0].plot(time_steps, error_x, 'g-', label='X方向误差', linewidth=2)
    axes[1, 0].plot(time_steps, error_y, 'm-', label='Y方向误差', linewidth=2)
    axes[1, 0].set_title('预测误差')
    axes[1, 0].set_xlabel('时间步')
    axes[1, 0].set_ylabel('绝对误差')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # 相位图
    axes[1, 1].plot(targets_np[sample_idx, :, 0], targets_np[sample_idx, :, 1], 'b-', label='真实轨迹', linewidth=2)
    axes[1, 1].plot(outputs_np[sample_idx, :, 0], outputs_np[sample_idx, :, 1], 'r--', label='预测轨迹', linewidth=2)
    axes[1, 1].set_title('振动轨迹')
    axes[1, 1].set_xlabel('X位移')
    axes[1, 1].set_ylabel('Y位移')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    axes[1, 1].axis('equal')
    
    plt.tight_layout()
    plt.savefig('prediction_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 如果有注意力权重，绘制注意力图
    if 'attention_weights' in locals():
        plot_attention_weights(attention_weights, save_path='attention_weights.png')


def save_model(model: nn.Module, path: str) -> None:
    """保存模型"""
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': model.get_config(),
        'timestamp': torch.tensor(time.time())
    }, path)
    
    print(f"模型已保存到: {path}")


if __name__ == '__main__':
    main()
```

---

## 🐛 问题报告

### 报告流程

#### 1. 搜索现有问题

在报告新问题前，请先搜索[GitHub Issues](https://github.com/your-repo/VIVTransformer/issues)确认问题是否已存在。

#### 2. 问题模板

```markdown
---
name: Bug报告
about: 报告软件缺陷
title: '[BUG] Bug report'
labels: bug
assignees: ''
---

## 问题描述

简洁清晰地描述遇到的问题。

## 复现步骤

详细描述如何复现这个问题：

1. 执行 '...'
2. 点击 '....'
3. 滚动到 '....'
4. 看到错误

## 期望行为

描述您期望发生的行为。

## 实际行为

描述实际发生的行为。

## 截图

如果适用，添加截图来帮助解释您的问题。

## 环境信息

- 操作系统: [例如 Ubuntu 20.04]
- Python版本: [例如 3.9.7]
- PyTorch版本: [例如 1.12.0]
- VIVTransformer版本: [例如 1.0.0]
- GPU信息: [例如 NVIDIA RTX 3080]
- CUDA版本: [例如 11.6]

## 错误日志

```
粘贴完整的错误信息和堆栈跟踪
```

## 最小复现代码

```python
# 提供能够复现问题的最小代码示例
import vivtransformer

# 您的代码...
```

## 额外信息

添加任何其他有助于解决问题的信息。
```

#### 3. 问题分类

**优先级标签**:
- `critical`: 严重问题，影响核心功能
- `high`: 高优先级，影响重要功能
- `medium`: 中等优先级
- `low`: 低优先级，改进建议

**类型标签**:
- `bug`: 软件缺陷
- `performance`: 性能问题
- `documentation`: 文档问题
- `compatibility`: 兼容性问题

### 问题调试指南

#### 1. 收集信息

```python
# 系统信息收集脚本
import sys
import torch
import vivtransformer
import platform

def collect_system_info():
    """收集系统信息用于问题报告"""
    info = {
        'platform': platform.platform(),
        'python_version': sys.version,
        'pytorch_version': torch.__version__,
        'vivtransformer_version': vivtransformer.__version__,
        'cuda_available': torch.cuda.is_available(),
        'cuda_version': torch.version.cuda if torch.cuda.is_available() else 'N/A',
        'gpu_count': torch.cuda.device_count() if torch.cuda.is_available() else 0
    }
    
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory
            info[f'gpu_{i}'] = f'{gpu_name} ({gpu_memory // 1024**3} GB)'
    
    return info

# 使用示例
if __name__ == '__main__':
    system_info = collect_system_info()
    for key, value in system_info.items():
        print(f'{key}: {value}')
```

#### 2. 日志配置

```python
# 启用详细日志
import logging
from vivtransformer.utils import setup_logging

# 设置日志级别
setup_logging(level=logging.DEBUG, log_file='vivtransformer_debug.log')

# 在代码中添加日志
logger = logging.getLogger(__name__)
logger.debug("调试信息")
logger.info("一般信息")
logger.warning("警告信息")
logger.error("错误信息")
```

---

## 💡 功能建议

### 建议流程

#### 1. 功能请求模板

```markdown
---
name: 功能请求
about: 建议新功能或改进
title: '[FEATURE] Feature request'
labels: enhancement
assignees: ''
---

## 功能描述

清晰简洁地描述您希望添加的功能。

## 问题背景

描述这个功能要解决的问题。例如：我总是感到沫丧当[...]

## 解决方案

描述您希望实现的解决方案。

## 替代方案

描述您考虑过的任何替代解决方案或功能。

## 使用场景

描述这个功能的具体使用场景：

1. 场景1: ...
2. 场景2: ...
3. 场景3: ...

## 预期API设计

```python
# 提供您期望的API设计示例
from vivtransformer import YourNewFeature

# 使用示例
feature = YourNewFeature(param1=value1, param2=value2)
result = feature.process(input_data)
```

## 实现复杂度

- [ ] 简单 (1-2天)
- [ ] 中等 (1-2周)
- [ ] 复杂 (1个月+)

## 相关资源

- 相关论文: [链接]
- 参考实现: [链接]
- 相关讨论: [链接]

## 额外信息

添加任何其他相关信息或截图。
```

#### 2. 功能评估标准

**评估维度**:
- **需求强度**: 社区需求程度
- **技术可行性**: 实现难度和技术风险
- **维护成本**: 长期维护的复杂度
- **兼容性**: 与现有功能的兼容性
- **性能影响**: 对整体性能的影响

**优先级矩阵**:

| 需求强度 | 实现难度 | 优先级 |
|----------|----------|--------|
| 高 | 低 | P0 (立即实现) |
| 高 | 中 | P1 (下个版本) |
| 高 | 高 | P2 (规划中) |
| 中 | 低 | P1 (下个版本) |
| 中 | 中 | P2 (规划中) |
| 中 | 高 | P3 (考虑中) |
| 低 | 低 | P2 (规划中) |
| 低 | 中/高 | P3 (考虑中) |

---

## 🧪 测试贡献

### 测试类型

#### 1. 单元测试

```python
# tests/test_attention_mechanisms.py

import pytest
import torch
from vivtransformer.attention_mechanisms import ExternalAttention

class TestExternalAttention:
    """外部注意力机制测试"""
    
    @pytest.fixture
    def attention_module(self):
        """创建测试用的注意力模块"""
        return ExternalAttention(
            d_model=512,
            n_heads=8,
            dropout=0.1
        )
    
    @pytest.fixture
    def sample_input(self):
        """创建测试输入"""
        batch_size, seq_len, d_model = 2, 10, 512
        return {
            'query': torch.randn(batch_size, seq_len, d_model),
            'key': torch.randn(batch_size, seq_len, d_model),
            'value': torch.randn(batch_size, seq_len, d_model)
        }
    
    def test_forward_pass(self, attention_module, sample_input):
        """测试前向传播"""
        output, attention_weights = attention_module(
            sample_input['query'],
            sample_input['key'],
            sample_input['value']
        )
        
        # 检查输出形状
        assert output.shape == sample_input['query'].shape
        assert attention_weights.shape == (2, 8, 10, 10)  # [batch, heads, seq, seq]
        
        # 检查注意力权重和为1
        assert torch.allclose(
            attention_weights.sum(dim=-1),
            torch.ones_like(attention_weights.sum(dim=-1)),
            atol=1e-6
        )
    
    def test_with_mask(self, attention_module, sample_input):
        """测试带掩码的注意力"""
        batch_size, seq_len = 2, 10
        mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).repeat(batch_size, 1, 1)
        
        output, attention_weights = attention_module(
            sample_input['query'],
            sample_input['key'],
            sample_input['value'],
            mask=mask
        )
        
        # 检查掩码位置的注意力权重为0
        masked_positions = (mask == 0)
        assert torch.allclose(
            attention_weights[masked_positions.unsqueeze(1).expand_as(attention_weights)],
            torch.zeros_like(attention_weights[masked_positions.unsqueeze(1).expand_as(attention_weights)]),
            atol=1e-6
        )
    
    def test_gradient_flow(self, attention_module, sample_input):
        """测试梯度流"""
        # 启用梯度计算
        for tensor in sample_input.values():
            tensor.requires_grad_(True)
        
        output, _ = attention_module(
            sample_input['query'],
            sample_input['key'],
            sample_input['value']
        )
        
        # 计算损失并反向传播
        loss = output.sum()
        loss.backward()
        
        # 检查梯度存在
        for tensor in sample_input.values():
            assert tensor.grad is not None
            assert not torch.allclose(tensor.grad, torch.zeros_like(tensor.grad))
    
    @pytest.mark.parametrize("d_model,n_heads", [
        (256, 4),
        (512, 8),
        (1024, 16)
    ])
    def test_different_configurations(self, d_model, n_heads):
        """测试不同配置"""
        attention = ExternalAttention(d_model=d_model, n_heads=n_heads)
        
        batch_size, seq_len = 2, 10
        query = torch.randn(batch_size, seq_len, d_model)
        key = torch.randn(batch_size, seq_len, d_model)
        value = torch.randn(batch_size, seq_len, d_model)
        
        output, attention_weights = attention(query, key, value)
        
        assert output.shape == (batch_size, seq_len, d_model)
        assert attention_weights.shape == (batch_size, n_heads, seq_len, seq_len)
```

#### 2. 集成测试

```python
# tests/test_integration.py

import pytest
import torch
from vivtransformer import VIVTransformer
from vivtransformer.data import VIVDataset
from vivtransformer.losses import SVDEnhancedLoss

class TestVIVTransformerIntegration:
    """VIVTransformer集成测试"""
    
    @pytest.fixture
    def model_config(self):
        """模型配置"""
        return {
            'input_dim': 64,
            'output_dim': 2,
            'd_model': 256,
            'n_heads': 4,
            'n_layers': 2,
            'attention_type': 'external',
            'max_seq_len': 32
        }
    
    @pytest.fixture
    def sample_data(self):
        """样本数据"""
        batch_size, seq_len, input_dim = 4, 32, 64
        output_dim = 2
        
        return {
            'input': torch.randn(batch_size, seq_len, input_dim),
            'target': torch.randn(batch_size, seq_len, output_dim)
        }
    
    def test_end_to_end_training(self, model_config, sample_data):
        """端到端训练测试"""
        # 创建模型
        model = VIVTransformer(**model_config)
        
        # 创建损失函数和优化器
        criterion = SVDEnhancedLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        # 训练几个步骤
        model.train()
        initial_loss = None
        
        for step in range(5):
            optimizer.zero_grad()
            
            output = model(sample_data['input'])
            loss = criterion(output, sample_data['target'])
            
            if initial_loss is None:
                initial_loss = loss.item()
            
            loss.backward()
            optimizer.step()
        
        # 检查损失是否下降
        final_loss = loss.item()
        assert final_loss < initial_loss, "训练过程中损失应该下降"
    
    def test_inference_mode(self, model_config, sample_data):
        """推理模式测试"""
        model = VIVTransformer(**model_config)
        model.eval()
        
        with torch.no_grad():
            output = model(sample_data['input'])
        
        # 检查输出形状
        expected_shape = (sample_data['input'].shape[0], sample_data['input'].shape[1], model_config['output_dim'])
        assert output.shape == expected_shape
        
        # 检查输出是否为有限值
        assert torch.isfinite(output).all(), "输出应该是有限值"
        
        # 检查输出不包含NaN
        assert not torch.isnan(output).any(), "输出不应包含NaN"
    
    def test_model_serialization(self, model_config):
        """模型序列化测试"""
        model = VIVTransformer(**model_config)
        
        # 保存模型
        torch.save(model.state_dict(), 'test_model.pth')
        
        # 加载模型
        new_model = VIVTransformer(**model_config)
        new_model.load_state_dict(torch.load('test_model.pth'))
        
        # 比较参数
        for p1, p2 in zip(model.parameters(), new_model.parameters()):
            assert torch.allclose(p1, p2), "加载的模型参数应该与原模型相同"
```

---

## 🏆 贡献者认可

### 贡献者等级

#### 1. 贡献者分类

| 等级 | 要求 | 权限 | 徽章 |
|------|------|------|------|
| **新手贡献者** | 1-5个PR | 提交PR、参与讨论 | 🌱 |
| **活跃贡献者** | 10+个PR，持续3个月 | 审查PR、标记Issues | 🌟 |
| **核心贡献者** | 50+个PR，重要功能开发 | 合并PR、发布管理 | 💎 |
| **维护者** | 长期贡献，技术领导 | 完全权限、项目决策 | 👑 |

### 认可机制

#### 1. 月度贡献者

每月评选优秀贡献者，标准包括：
- 代码质量和创新性
- 文档贡献
- 社区参与度
- 帮助其他贡献者

#### 2. 年度奖项

- **最佳新人奖**: 新加入的优秀贡献者
- **技术创新奖**: 重要技术突破
- **社区建设奖**: 社区发展贡献
- **文档贡献奖**: 文档质量提升

---

## 📞 联系我们

### 沟通渠道

#### 1. 官方渠道

- **GitHub Issues**: [项目问题和功能请求](https://github.com/your-repo/VIVTransformer/issues)
- **GitHub Discussions**: [社区讨论](https://github.com/your-repo/VIVTransformer/discussions)
- **Pull Requests**: [代码贡献](https://github.com/your-repo/VIVTransformer/pulls)

#### 2. 社交媒体

- **Twitter**: [@VIVTransformer](https://twitter.com/VIVTransformer)
- **LinkedIn**: [VIVTransformer项目](https://linkedin.com/company/vivtransformer)
- **知乎**: [VIVTransformer专栏](https://zhuanlan.zhihu.com/vivtransformer)

### 行为准则

我们致力于为所有人提供友好、安全和欢迎的环境。请遵守以下准则：

#### 积极行为
- ✅ 使用友好和包容的语言
- ✅ 尊重不同的观点和经验
- ✅ 优雅地接受建设性批评
- ✅ 关注对社区最有利的事情
- ✅ 对其他社区成员表现出同理心

#### 不当行为
- ❌ 使用性化的语言或图像
- ❌ 恶意评论、人身攻击或政治攻击
- ❌ 公开或私下骚扰
- ❌ 未经明确许可发布他人的私人信息
- ❌ 在专业环境中被认为不当的其他行为

### 获得帮助

如果您需要帮助或有任何问题，请通过以下方式联系我们：

1. **技术问题**: 在GitHub Issues中创建问题
2. **使用疑问**: 查看文档或在Discussions中提问
3. **合作机会**: 发送邮件至 collaboration@vivtransformer.org
4. **媒体咨询**: 发送邮件至 media@vivtransformer.org

---

*感谢您对VIVTransformer项目的关注和贡献！让我们一起构建更好的涡激振动分析工具。* 🚀