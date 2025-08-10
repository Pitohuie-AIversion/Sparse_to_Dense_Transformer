# 📚 教程与示例

> 从入门到精通的VIVTransformer使用教程和代码示例

---

## 📋 目录

- [🚀 快速入门教程](#-快速入门教程)
- [🔧 基础配置示例](#-基础配置示例)
- [🧠 注意力机制使用](#-注意力机制使用)
- [📊 损失函数配置](#-损失函数配置)
- [🏋️ 训练流程示例](#️-训练流程示例)
- [📈 评估与可视化](#-评估与可视化)
- [🔬 高级应用案例](#-高级应用案例)
- [🌊 涡激振动专业应用](#-涡激振动专业应用)
- [🚀 部署与推理](#-部署与推理)
- [🛠️ 自定义扩展](#️-自定义扩展)

---

## 🚀 快速入门教程

### 第一个VIVTransformer模型

```python
# quick_start.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# 假设这些是VIVTransformer的核心组件
from vivtransformer import VIVTransformer, VIVConfig
from vivtransformer.data import VIVDataset
from vivtransformer.training import VIVTrainer
from vivtransformer.losses import SVDLoss

def quick_start_example():
    """快速入门示例"""
    
    print("🚀 VIVTransformer 快速入门示例")
    
    # 1. 创建配置
    config = VIVConfig(
        d_model=256,           # 模型维度
        n_heads=8,             # 注意力头数
        n_layers=6,            # 层数
        attention_type='scaled_dot_product',  # 注意力类型
        max_seq_length=512,    # 最大序列长度
        dropout=0.1,           # Dropout率
        use_svd_loss=True      # 使用SVD损失
    )
    
    print(f"✅ 配置创建完成: {config}")
    
    # 2. 创建模型
    model = VIVTransformer(config)
    print(f"✅ 模型创建完成，参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 3. 准备数据
    # 生成示例数据（实际使用时替换为真实数据）
    batch_size = 32
    seq_length = 128
    input_dim = config.d_model
    
    # 输入数据：[batch_size, seq_length, input_dim]
    x = torch.randn(batch_size, seq_length, input_dim)
    # 目标数据：[batch_size, seq_length, output_dim]
    y = torch.randn(batch_size, seq_length, input_dim)
    
    print(f"✅ 数据准备完成，输入形状: {x.shape}, 目标形状: {y.shape}")
    
    # 4. 前向传播
    model.eval()
    with torch.no_grad():
        output = model(x)
        print(f"✅ 前向传播完成，输出形状: {output.shape}")
    
    # 5. 计算损失
    criterion = SVDLoss(alpha=0.5, beta=0.3)
    loss = criterion(output, y)
    print(f"✅ 损失计算完成，损失值: {loss.item():.6f}")
    
    # 6. 简单训练步骤
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    optimizer.zero_grad()
    output = model(x)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()
    
    print(f"✅ 训练步骤完成，损失值: {loss.item():.6f}")
    print("🎉 快速入门示例完成！")

if __name__ == "__main__":
    quick_start_example()
```

### 环境设置

```bash
# 安装依赖
pip install torch torchvision torchaudio
pip install numpy pandas matplotlib seaborn
pip install tensorboard wandb  # 可选：实验跟踪
pip install plotly dash       # 可选：交互式可视化

# 克隆项目
git clone https://github.com/your-repo/VIVTransformer.git
cd VIVTransformer

# 安装项目
pip install -e .

# 验证安装
python -c "import vivtransformer; print('安装成功！')"
```

---

## 🔧 基础配置示例

### YAML配置文件

```yaml
# config/basic_config.yaml
model:
  name: "VIVTransformer"
  d_model: 512
  n_heads: 8
  n_layers: 6
  d_ff: 2048
  dropout: 0.1
  max_seq_length: 1024
  
  # 注意力机制配置
  attention:
    type: "scaled_dot_product"  # 可选: multi_head, external, se_attention等
    use_relative_position: true
    max_relative_position: 32
    
  # 位置编码
  position_encoding:
    type: "sinusoidal"  # 可选: learned, rotary
    max_length: 1024

# 训练配置
training:
  batch_size: 32
  learning_rate: 1e-4
  num_epochs: 100
  warmup_steps: 1000
  gradient_clip_norm: 1.0
  
  # 优化器
  optimizer:
    type: "adamw"
    weight_decay: 0.01
    betas: [0.9, 0.999]
    
  # 学习率调度
  scheduler:
    type: "cosine_annealing"
    T_max: 100
    eta_min: 1e-6

# 损失函数配置
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

# 数据配置
data:
  train_path: "data/train"
  val_path: "data/val"
  test_path: "data/test"
  
  # 数据预处理
  preprocessing:
    normalize: true
    standardize: true
    augmentation:
      noise_std: 0.01
      time_shift_max: 5
      amplitude_scale_range: [0.9, 1.1]

# 实验配置
experiment:
  name: "basic_experiment"
  save_dir: "experiments"
  log_interval: 100
  save_interval: 1000
  
  # 监控指标
  metrics:
    - "mse"
    - "mae"
    - "r2_score"
    - "attention_entropy"

# 硬件配置
hardware:
  device: "auto"  # auto, cpu, cuda, cuda:0
  mixed_precision: true
  compile_model: false  # PyTorch 2.0+
```

### Python配置类

```python
# config_example.py
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union
import yaml
from pathlib import Path

@dataclass
class AttentionConfig:
    """注意力机制配置"""
    type: str = "scaled_dot_product"
    use_relative_position: bool = True
    max_relative_position: int = 32
    dropout: float = 0.1

@dataclass
class ModelConfig:
    """模型配置"""
    name: str = "VIVTransformer"
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 2048
    dropout: float = 0.1
    max_seq_length: int = 1024
    attention: AttentionConfig = field(default_factory=AttentionConfig)
    
    def __post_init__(self):
        """配置验证"""
        if self.d_model % self.n_heads != 0:
            raise ValueError(f"d_model ({self.d_model}) 必须能被 n_heads ({self.n_heads}) 整除")
        
        if self.dropout < 0 or self.dropout > 1:
            raise ValueError(f"dropout 必须在 [0, 1] 范围内，当前值: {self.dropout}")

@dataclass
class TrainingConfig:
    """训练配置"""
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
    """损失函数配置"""
    primary_type: str = "svd_loss"
    primary_params: Dict = field(default_factory=lambda: {"alpha": 0.5, "beta": 0.3, "gamma": 0.2})
    auxiliary_losses: List[Dict] = field(default_factory=list)

@dataclass
class VIVConfig:
    """VIVTransformer完整配置"""
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    
    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path]) -> 'VIVConfig':
        """从YAML文件加载配置"""
        with open(yaml_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_dict(cls, config_dict: Dict) -> 'VIVConfig':
        """从字典创建配置"""
        model_config = ModelConfig(**config_dict.get('model', {}))
        training_config = TrainingConfig(**config_dict.get('training', {}))
        loss_config = LossConfig(**config_dict.get('loss', {}))
        
        return cls(
            model=model_config,
            training=training_config,
            loss=loss_config
        )
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            'model': self.model.__dict__,
            'training': self.training.__dict__,
            'loss': self.loss.__dict__
        }
    
    def save_yaml(self, yaml_path: Union[str, Path]):
        """保存为YAML文件"""
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, allow_unicode=True)

# 使用示例
if __name__ == "__main__":
    # 1. 创建默认配置
    config = VIVConfig()
    print("默认配置:")
    print(config)
    
    # 2. 从YAML加载配置
    # config = VIVConfig.from_yaml('config/basic_config.yaml')
    
    # 3. 自定义配置
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
    
    print("\n自定义配置:")
    print(custom_config)
    
    # 4. 保存配置
    custom_config.save_yaml('config/custom_config.yaml')
    print("\n配置已保存到 config/custom_config.yaml")
```

---

## 🧠 注意力机制使用

### 多种注意力机制对比

```python
# attention_comparison.py
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, List, Tuple
import time

# 假设这些是项目中的注意力机制
from vivtransformer.attention import (
    ScaledDotProductAttention,
    ExternalAttention,
    SEAttention,
    CBAMAttention,
    ECAAttention,
    CoordinateAttention
)

class AttentionBenchmark:
    """注意力机制基准测试"""
    
    def __init__(self, d_model: int = 512, seq_length: int = 128, batch_size: int = 32):
        self.d_model = d_model
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 创建测试数据
        self.test_data = torch.randn(batch_size, seq_length, d_model).to(self.device)
        
        # 注意力机制字典
        self.attention_modules = {
            'Scaled Dot-Product': ScaledDotProductAttention(d_model, n_heads=8),
            'External Attention': ExternalAttention(d_model),
            'SE Attention': SEAttention(d_model),
            'CBAM': CBAMAttention(d_model),
            'ECA': ECAAttention(d_model),
            'Coordinate Attention': CoordinateAttention(d_model)
        }
        
        # 移动到设备
        for name, module in self.attention_modules.items():
            module.to(self.device)
    
    def benchmark_performance(self, num_runs: int = 100) -> Dict[str, Dict[str, float]]:
        """性能基准测试"""
        
        results = {}
        
        for name, module in self.attention_modules.items():
            print(f"测试 {name}...")
            
            # 预热
            for _ in range(10):
                with torch.no_grad():
                    _ = module(self.test_data)
            
            # 计时测试
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            start_time = time.time()
            
            for _ in range(num_runs):
                with torch.no_grad():
                    output = module(self.test_data)
            
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            end_time = time.time()
            
            # 计算指标
            avg_time = (end_time - start_time) / num_runs * 1000  # ms
            throughput = (self.batch_size * num_runs) / (end_time - start_time)  # samples/sec
            
            # 内存使用（近似）
            param_count = sum(p.numel() for p in module.parameters())
            memory_mb = param_count * 4 / (1024 * 1024)  # 假设float32
            
            results[name] = {
                'avg_time_ms': avg_time,
                'throughput_samples_per_sec': throughput,
                'parameters': param_count,
                'memory_mb': memory_mb
            }
        
        return results
    
    def analyze_attention_patterns(self) -> Dict[str, torch.Tensor]:
        """分析注意力模式"""
        
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
        """可视化结果"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. 平均推理时间
        names = list(results.keys())
        times = [results[name]['avg_time_ms'] for name in names]
        
        axes[0, 0].bar(names, times, color='skyblue')
        axes[0, 0].set_title('平均推理时间 (ms)')
        axes[0, 0].set_ylabel('时间 (ms)')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. 吞吐量
        throughputs = [results[name]['throughput_samples_per_sec'] for name in names]
        
        axes[0, 1].bar(names, throughputs, color='lightgreen')
        axes[0, 1].set_title('吞吐量 (samples/sec)')
        axes[0, 1].set_ylabel('样本数/秒')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. 参数数量
        param_counts = [results[name]['parameters'] / 1000 for name in names]  # K parameters
        
        axes[1, 0].bar(names, param_counts, color='orange')
        axes[1, 0].set_title('参数数量 (K)')
        axes[1, 0].set_ylabel('参数数量 (千)')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 4. 内存使用
        memory_usage = [results[name]['memory_mb'] for name in names]
        
        axes[1, 1].bar(names, memory_usage, color='salmon')
        axes[1, 1].set_title('内存使用 (MB)')
        axes[1, 1].set_ylabel('内存 (MB)')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('attention_benchmark_results.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def visualize_attention_weights(self, attention_weights: Dict[str, torch.Tensor]):
        """可视化注意力权重"""
        
        num_attentions = len(attention_weights)
        if num_attentions == 0:
            print("没有可用的注意力权重")
            return
        
        fig, axes = plt.subplots(1, num_attentions, figsize=(5 * num_attentions, 5))
        if num_attentions == 1:
            axes = [axes]
        
        for idx, (name, weights) in enumerate(attention_weights.items()):
            # 取第一个样本的第一个头的注意力权重
            if weights.dim() == 4:  # [batch, heads, seq, seq]
                attn_map = weights[0, 0].numpy()
            elif weights.dim() == 3:  # [batch, seq, seq]
                attn_map = weights[0].numpy()
            else:
                continue
            
            # 只显示前32x32的区域（如果序列太长）
            display_size = min(32, attn_map.shape[0])
            attn_map = attn_map[:display_size, :display_size]
            
            im = axes[idx].imshow(attn_map, cmap='Blues', aspect='auto')
            axes[idx].set_title(f'{name}\n注意力权重')
            axes[idx].set_xlabel('Key Position')
            axes[idx].set_ylabel('Query Position')
            
            # 添加颜色条
            plt.colorbar(im, ax=axes[idx])
        
        plt.tight_layout()
        plt.savefig('attention_weights_visualization.png', dpi=300, bbox_inches='tight')
        plt.show()

# 使用示例
def run_attention_comparison():
    """运行注意力机制对比"""
    
    print("🧠 注意力机制对比分析")
    
    # 创建基准测试
    benchmark = AttentionBenchmark(d_model=512, seq_length=128, batch_size=32)
    
    # 性能测试
    print("\n📊 性能基准测试...")
    results = benchmark.benchmark_performance(num_runs=50)
    
    # 打印结果
    print("\n📈 性能测试结果:")
    print("-" * 80)
    print(f"{'注意力机制':<20} {'时间(ms)':<12} {'吞吐量':<15} {'参数数':<12} {'内存(MB)':<10}")
    print("-" * 80)
    
    for name, metrics in results.items():
        print(f"{name:<20} {metrics['avg_time_ms']:<12.2f} "
              f"{metrics['throughput_samples_per_sec']:<15.1f} "
              f"{metrics['parameters']:<12,} {metrics['memory_mb']:<10.2f}")
    
    # 可视化性能结果
    benchmark.visualize_results(results)
    
    # 注意力模式分析
    print("\n🔍 注意力模式分析...")
    attention_weights = benchmark.analyze_attention_patterns()
    
    if attention_weights:
        benchmark.visualize_attention_weights(attention_weights)
        print(f"✅ 分析了 {len(attention_weights)} 种注意力机制的权重模式")
    else:
        print("⚠️ 没有可用的注意力权重进行分析")
    
    print("\n🎉 注意力机制对比完成！")
    return results, attention_weights

if __name__ == "__main__":
    results, attention_weights = run_attention_comparison()
```

### 自定义注意力机制

```python
# custom_attention.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

class AdaptiveAttention(nn.Module):
    """自适应注意力机制
    
    根据输入动态调整注意力计算方式
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
        
        # 线性变换层
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model)
        
        # 自适应门控
        self.adaptive_gate = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.ReLU(),
            nn.Linear(d_model // 4, 1),
            nn.Sigmoid()
        )
        
        # 局部注意力卷积
        self.local_conv = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, groups=d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
        
        # 存储注意力权重用于可视化
        self.attention_weights = None
    
    def forward(self, 
                query: torch.Tensor, 
                key: torch.Tensor, 
                value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """前向传播
        
        Args:
            query: [batch_size, seq_len, d_model]
            key: [batch_size, seq_len, d_model]
            value: [batch_size, seq_len, d_model]
            mask: [batch_size, seq_len, seq_len] 或 None
            
        Returns:
            output: [batch_size, seq_len, d_model]
        """
        
        batch_size, seq_len, d_model = query.shape
        
        # 计算自适应权重
        adaptive_weights = self.adaptive_gate(query.mean(dim=1))  # [batch_size, 1]
        
        # 全局注意力
        global_output = self._global_attention(query, key, value, mask)
        
        # 局部注意力
        local_output = self._local_attention(query)
        
        # 自适应融合
        output = adaptive_weights.unsqueeze(1) * global_output + \
                (1 - adaptive_weights.unsqueeze(1)) * local_output
        
        return output
    
    def _global_attention(self, 
                         query: torch.Tensor, 
                         key: torch.Tensor, 
                         value: torch.Tensor,
                         mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """全局注意力计算"""
        
        batch_size, seq_len, d_model = query.shape
        
        # 线性变换
        Q = self.w_q(query).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        # 应用掩码
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1) == 0, -1e9)
        
        # Softmax归一化
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # 保存注意力权重
        self.attention_weights = attention_weights.detach()
        
        # 应用注意力
        context = torch.matmul(attention_weights, V)
        
        # 重塑输出
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len, d_model
        )
        
        return self.w_o(context)
    
    def _local_attention(self, x: torch.Tensor) -> torch.Tensor:
        """局部注意力计算"""
        
        # 转置用于卷积 [batch_size, d_model, seq_len]
        x_conv = x.transpose(1, 2)
        
        # 局部卷积
        local_features = self.local_conv(x_conv)
        
        # 转回原始形状
        local_output = local_features.transpose(1, 2)
        
        return local_output
    
    def get_attention_weights(self) -> Optional[torch.Tensor]:
        """获取注意力权重"""
        return self.attention_weights

class HierarchicalAttention(nn.Module):
    """层次化注意力机制
    
    在不同层次上计算注意力
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
        
        # 多层次注意力
        self.level_attentions = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=n_heads,
                dropout=dropout,
                batch_first=True
            ) for _ in range(n_levels)
        ])
        
        # 层次融合权重
        self.level_weights = nn.Parameter(torch.ones(n_levels) / n_levels)
        
        # 下采样和上采样
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
        """前向传播"""
        
        batch_size, seq_len, d_model = query.shape
        level_outputs = []
        
        # 第0层：原始分辨率
        output_0, _ = self.level_attentions[0](query, key, value, attn_mask=mask)
        level_outputs.append(output_0)
        
        # 其他层：不同分辨率
        current_q, current_k, current_v = query, key, value
        
        for level in range(1, self.n_levels):
            # 下采样
            current_q = self._downsample_sequence(current_q, level - 1)
            current_k = self._downsample_sequence(current_k, level - 1)
            current_v = self._downsample_sequence(current_v, level - 1)
            
            # 注意力计算
            level_output, _ = self.level_attentions[level](current_q, current_k, current_v)
            
            # 上采样回原始分辨率
            level_output = self._upsample_sequence(level_output, level - 1, seq_len)
            level_outputs.append(level_output)
        
        # 加权融合
        weights = F.softmax(self.level_weights, dim=0)
        final_output = sum(w * output for w, output in zip(weights, level_outputs))
        
        return final_output
    
    def _downsample_sequence(self, x: torch.Tensor, level: int) -> torch.Tensor:
        """下采样序列"""
        # [batch_size, seq_len, d_model] -> [batch_size, d_model, seq_len]
        x = x.transpose(1, 2)
        x = self.downsample[level](x)
        # [batch_size, d_model, new_seq_len] -> [batch_size, new_seq_len, d_model]
        x = x.transpose(1, 2)
        return x
    
    def _upsample_sequence(self, x: torch.Tensor, level: int, target_len: int) -> torch.Tensor:
        """上采样序列"""
        # [batch_size, seq_len, d_model] -> [batch_size, d_model, seq_len]
        x = x.transpose(1, 2)
        x = self.upsample[level](x)
        
        # 调整到目标长度
        current_len = x.shape[-1]
        if current_len != target_len:
            x = F.interpolate(x, size=target_len, mode='linear', align_corners=False)
        
        # [batch_size, d_model, target_len] -> [batch_size, target_len, d_model]
        x = x.transpose(1, 2)
        return x

# 使用示例
def test_custom_attention():
    """测试自定义注意力机制"""
    
    print("🧠 测试自定义注意力机制")
    
    # 创建测试数据
    batch_size, seq_len, d_model = 4, 64, 256
    x = torch.randn(batch_size, seq_len, d_model)
    
    # 测试自适应注意力
    print("\n🔄 测试自适应注意力...")
    adaptive_attn = AdaptiveAttention(d_model, n_heads=8)
    adaptive_output = adaptive_attn(x, x, x)
    print(f"输入形状: {x.shape}")
    print(f"自适应注意力输出形状: {adaptive_output.shape}")
    
    # 获取注意力权重
    attn_weights = adaptive_attn.get_attention_weights()
    if attn_weights is not None:
        print(f"注意力权重形状: {attn_weights.shape}")
    
    # 测试层次化注意力
    print("\n🏗️ 测试层次化注意力...")
    hierarchical_attn = HierarchicalAttention(d_model, n_heads=8, n_levels=3)
    hierarchical_output = hierarchical_attn(x, x, x)
    print(f"层次化注意力输出形状: {hierarchical_output.shape}")
    
    # 参数统计
    adaptive_params = sum(p.numel() for p in adaptive_attn.parameters())
    hierarchical_params = sum(p.numel() for p in hierarchical_attn.parameters())
    
    print(f"\n📊 参数统计:")
    print(f"自适应注意力参数数量: {adaptive_params:,}")
    print(f"层次化注意力参数数量: {hierarchical_params:,}")
    
    print("\n✅ 自定义注意力机制测试完成！")

if __name__ == "__main__":
    test_custom_attention()
```

---

## 📊 损失函数配置

### SVD损失函数详解

```python
# svd_loss_tutorial.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional, Dict, Any

class SVDLoss(nn.Module):
    """SVD损失函数
    
    基于奇异值分解的损失函数，用于保持数据的低秩结构
    """
    
    def __init__(self, 
                 alpha: float = 0.5, 
                 beta: float = 0.3, 
                 gamma: float = 0.2,
                 rank_penalty: bool = True,
                 adaptive_weights: bool = True):
        """
        Args:
            alpha: 重构损失权重
            beta: 奇异值损失权重
            gamma: 正交性损失权重
            rank_penalty: 是否使用秩惩罚
            adaptive_weights: 是否使用自适应权重
        """
        super().__init__()
        
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.rank_penalty = rank_penalty
        self.adaptive_weights = adaptive_weights
        
        # 自适应权重参数
        if adaptive_weights:
            self.weight_net = nn.Sequential(
                nn.Linear(3, 16),  # 3个损失分量
                nn.ReLU(),
                nn.Linear(16, 3),
                nn.Softmax(dim=-1)
            )
    
    def forward(self, 
                pred: torch.Tensor, 
                target: torch.Tensor,
                return_components: bool = False) -> torch.Tensor:
        """
        计算SVD损失
        
        Args:
            pred: 预测值 [batch_size, seq_len, features]
            target: 目标值 [batch_size, seq_len, features]
            return_components: 是否返回损失分量
            
        Returns:
            loss: 总损失
            components (可选): 损失分量字典
        """
        
        # 1. 重构损失（MSE）
        reconstruction_loss = F.mse_loss(pred, target)
        
        # 2. SVD分解
        pred_svd = self._compute_svd_loss(pred, target)
        target_svd = self._compute_svd_loss(target, target)
        
        # 3. 奇异值损失
        singular_value_loss = F.mse_loss(pred_svd['singular_values'], target_svd['singular_values'])
        
        # 4. 正交性损失
        orthogonality_loss = self._compute_orthogonality_loss(pred_svd['U'], pred_svd['V'])
        
        # 5. 秩惩罚（可选）
        rank_loss = 0.0
        if self.rank_penalty:
            rank_loss = self._compute_rank_penalty(pred_svd['singular_values'])
        
        # 损失分量
        components = {
            'reconstruction': reconstruction_loss,
            'singular_value': singular_value_loss,
            'orthogonality': orthogonality_loss,
            'rank_penalty': rank_loss
        }
        
        # 计算权重
        if self.adaptive_weights:
            # 使用神经网络自适应调整权重
            loss_values = torch.stack([
                reconstruction_loss.detach(),
                singular_value_loss.detach(),
                orthogonality_loss.detach()
            ])
            weights = self.weight_net(loss_values.unsqueeze(0)).squeeze(0)
            alpha, beta, gamma = weights[0], weights[1], weights[2]
        else:
            alpha, beta, gamma = self.alpha, self.beta, self.gamma
        
        # 总损失
        total_loss = (
            alpha * reconstruction_loss +
            beta * singular_value_loss +
            gamma * orthogonality_loss +
            0.1 * rank_loss  # 固定权重用于秩惩罚
        )
        
        if return_components:
            components['weights'] = {'alpha': alpha, 'beta': beta, 'gamma': gamma}
            components['total'] = total_loss
            return total_loss, components
        
        return total_loss
    
    def _compute_svd_loss(self, x: torch.Tensor, reference: torch.Tensor) -> Dict[str, torch.Tensor]:
        """计算SVD分解"""
        
        batch_size, seq_len, features = x.shape
        
        # 重塑为矩阵形式
        x_matrix = x.view(batch_size, seq_len * features)
        
        # SVD分解
        try:
            U, S, V = torch.svd(x_matrix)
        except RuntimeError:
            # 如果SVD失败，使用备用方法
            U, S, V = torch.svd(x_matrix + 1e-8 * torch.randn_like(x_matrix))
        
        return {
            'U': U,
            'singular_values': S,
            'V': V
        }
    
    def _compute_orthogonality_loss(self, U: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        """计算正交性损失"""
        
        # U的正交性
        U_orth = torch.matmul(U.transpose(-2, -1), U)
        I_U = torch.eye(U_orth.shape[-1], device=U.device, dtype=U.dtype)
        U_loss = F.mse_loss(U_orth, I_U.expand_as(U_orth))
        
        # V的正交性
        V_orth = torch.matmul(V.transpose(-2, -1), V)
        I_V = torch.eye(V_orth.shape[-1], device=V.device, dtype=V.dtype)
        V_loss = F.mse_loss(V_orth, I_V.expand_as(V_orth))
        
        return (U_loss + V_loss) / 2
    
    def _compute_rank_penalty(self, singular_values: torch.Tensor) -> torch.Tensor:
        """计算秩惩罚"""
        
        # 使用奇异值的L1范数作为秩的近似
        rank_penalty = torch.sum(singular_values, dim=-1).mean()
        
        return rank_penalty

class CompositeLoss(nn.Module):
    """复合损失函数
    
    组合多种损失函数
    """
    
    def __init__(self, loss_configs: Dict[str, Dict[str, Any]]):
        """
        Args:
            loss_configs: 损失函数配置字典
                例如: {
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
            
            # 创建损失函数
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
                raise ValueError(f"不支持的损失函数类型: {loss_type}")
            
            self.loss_functions[name] = loss_fn
            self.loss_weights[name] = weight
    
    def forward(self, 
                pred: torch.Tensor, 
                target: torch.Tensor,
                return_components: bool = False) -> torch.Tensor:
        """计算复合损失"""
        
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
    """演示SVD损失函数"""
    
    print("📊 SVD损失函数演示")
    
    # 创建测试数据
    batch_size, seq_len, features = 8, 64, 32
    
    # 创建低秩目标数据
    rank = 10
    U_true = torch.randn(batch_size, seq_len * features, rank)
    S_true = torch.abs(torch.randn(batch_size, rank)) + 0.1
    V_true = torch.randn(batch_size, rank, seq_len * features)
    
    target_matrix = torch.bmm(torch.bmm(U_true, torch.diag_embed(S_true)), V_true)
    target = target_matrix.view(batch_size, seq_len, features)
    
    # 创建预测数据（添加噪声）
    noise = 0.1 * torch.randn_like(target)
    pred = target + noise
    
    print(f"数据形状: {target.shape}")
    print(f"目标数据的真实秩: {rank}")
    
    # 测试不同的损失函数
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
        print(f"\n测试 {name}...")
        
        if config['type'] == 'SVDLoss':
            loss_fn = SVDLoss(**config['params'])
            loss_value, components = loss_fn(pred, target, return_components=True)
            
            print(f"  总损失: {loss_value.item():.6f}")
            print(f"  重构损失: {components['reconstruction'].item():.6f}")
            print(f"  奇异值损失: {components['singular_value'].item():.6f}")
            print(f"  正交性损失: {components['orthogonality'].item():.6f}")
            
            if 'weights' in components:
                weights = components['weights']
                print(f"  自适应权重: α={weights['alpha']:.3f}, β={weights['beta']:.3f}, γ={weights['gamma']:.3f}")
        
        else:
            loss_fn = nn.MSELoss()
            loss_value = loss_fn(pred, target)
            print(f"  损失值: {loss_value.item():.6f}")
        
        results[name] = loss_value.item()
    
    # 可视化结果
    plt.figure(figsize=(10, 6))
    
    names = list(results.keys())
    values = list(results.values())
    
    plt.bar(names, values, color=['skyblue', 'lightgreen', 'salmon'])
    plt.title('不同损失函数的损失值对比')
    plt.ylabel('损失值')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('loss_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n✅ SVD损失函数演示完成！")

def loss_function_tutorial():
    """损失函数使用教程"""
    
    print("📚 损失函数使用教程")
    
    # 1. 基本SVD损失
    print("\n1️⃣ 基本SVD损失使用")
    
    svd_loss = SVDLoss(alpha=0.5, beta=0.3, gamma=0.2)
    
    # 示例数据
    pred = torch.randn(4, 32, 16)
    target = torch.randn(4, 32, 16)
    
    loss = svd_loss(pred, target)
    print(f"SVD损失值: {loss.item():.6f}")
    
    # 2. 复合损失
    print("\n2️⃣ 复合损失使用")
    
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
    
    print(f"复合损失总值: {total_loss.item():.6f}")
    print("各分量损失:")
    for name, component in components.items():
        if name != 'total':
            if isinstance(component, dict) and 'total' in component:
                print(f"  {name}: {component['total'].item():.6f}")
    
    # 3. 训练中的使用
    print("\n3️⃣ 训练中的使用示例")
    
    # 创建简单模型
    model = nn.Linear(16, 16)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # 训练循环示例
    for epoch in range(5):
        optimizer.zero_grad()
        
        output = model(pred.view(-1, 16)).view(4, 32, 16)
        loss = svd_loss(output, target)
        
        loss.backward()
        optimizer.step()
        
        print(f"Epoch {epoch+1}, Loss: {loss.item():.6f}")
    
    print("\n✅ 损失函数教程完成！")

if __name__ == "__main__":
    demonstrate_svd_loss()
    print("\n" + "="*50 + "\n")
    loss_function_tutorial()
```

---

*本教程与示例页面提供了详细的使用指导和代码示例。更多高级功能请参考[API文档]({{ site.baseurl }}/pages/api-reference)和[最佳实践]({{ site.baseurl }}/pages/best-practices)。*