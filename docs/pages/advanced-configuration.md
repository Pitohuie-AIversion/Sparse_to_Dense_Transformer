---
layout: default
title: 高级配置指南
nav_order: 14
permalink: /pages/advanced-configuration/
description: "VIVTransformer高级配置和定制化指南"
---

# 高级配置指南 ⚙️

本指南提供VIVTransformer的高级配置选项和定制化方案，帮助您充分发挥系统的潜力。

## 📋 目录

- [配置系统架构](#配置系统架构)
- [模型架构配置](#模型架构配置)
- [注意力机制定制](#注意力机制定制)
- [损失函数配置](#损失函数配置)
- [训练策略配置](#训练策略配置)
- [数据处理配置](#数据处理配置)
- [性能优化配置](#性能优化配置)
- [分布式训练配置](#分布式训练配置)
- [实验管理配置](#实验管理配置)
- [部署配置](#部署配置)

---

## 🏗️ 配置系统架构

### 配置文件层次结构

VIVTransformer采用分层配置系统，支持灵活的配置管理：

```yaml
# config/base_config.yaml - 基础配置
model:
  name: "VIVTransformer"
  version: "1.0.0"
  
training:
  epochs: 100
  batch_size: 32
  learning_rate: 1e-4
  
data:
  sequence_length: 100
  input_dim: 512
  
logging:
  level: "INFO"
  save_dir: "./logs"
```

```yaml
# config/attention_config.yaml - 注意力配置
attention:
  type: "MultiHeadAttention"
  n_heads: 8
  dropout: 0.1
  
  # 高级选项
  use_bias: true
  scale_factor: null  # 自动计算
  attention_dropout: 0.1
  
  # 位置编码
  positional_encoding:
    type: "sinusoidal"  # sinusoidal, learned, rotary
    max_length: 1000
```

```yaml
# config/loss_config.yaml - 损失函数配置
loss:
  primary:
    type: "MSELoss"
    weight: 1.0
    
  auxiliary:
    - type: "SVDLoss"
      weight: 0.1
      rank_penalty: 0.01
    - type: "PhysicsLoss"
      weight: 0.05
      constraints: ["continuity", "momentum"]
```

### 配置加载和合并

```python
# config_manager.py
import yaml
from pathlib import Path
from typing import Dict, Any, List
from dataclasses import dataclass, field
from omegaconf import OmegaConf, DictConfig

@dataclass
class VIVConfig:
    """VIVTransformer配置类"""
    
    # 模型配置
    model: Dict[str, Any] = field(default_factory=dict)
    
    # 训练配置
    training: Dict[str, Any] = field(default_factory=dict)
    
    # 数据配置
    data: Dict[str, Any] = field(default_factory=dict)
    
    # 注意力配置
    attention: Dict[str, Any] = field(default_factory=dict)
    
    # 损失配置
    loss: Dict[str, Any] = field(default_factory=dict)
    
    # 优化器配置
    optimizer: Dict[str, Any] = field(default_factory=dict)
    
    # 调度器配置
    scheduler: Dict[str, Any] = field(default_factory=dict)
    
    # 日志配置
    logging: Dict[str, Any] = field(default_factory=dict)
    
    # 实验配置
    experiment: Dict[str, Any] = field(default_factory=dict)

class ConfigManager:
    """配置管理器"""
    
    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self.config_cache = {}
    
    def load_config(self, config_files: List[str], overrides: Dict[str, Any] = None) -> DictConfig:
        """加载和合并配置文件"""
        
        # 加载基础配置
        merged_config = {}
        
        for config_file in config_files:
            config_path = self.config_dir / config_file
            
            if config_path.exists():
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                    merged_config = self._deep_merge(merged_config, config)
            else:
                print(f"⚠️  配置文件不存在: {config_path}")
        
        # 应用覆盖配置
        if overrides:
            merged_config = self._deep_merge(merged_config, overrides)
        
        # 转换为OmegaConf对象
        omega_config = OmegaConf.create(merged_config)
        
        # 验证配置
        self._validate_config(omega_config)
        
        return omega_config
    
    def _deep_merge(self, base: Dict, update: Dict) -> Dict:
        """深度合并字典"""
        result = base.copy()
        
        for key, value in update.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def _validate_config(self, config: DictConfig):
        """验证配置有效性"""
        
        # 检查必需字段
        required_fields = ['model', 'training', 'data']
        for field in required_fields:
            if field not in config:
                raise ValueError(f"缺少必需配置字段: {field}")
        
        # 检查数值范围
        if 'training' in config:
            training = config.training
            
            if 'learning_rate' in training and training.learning_rate <= 0:
                raise ValueError("学习率必须大于0")
            
            if 'batch_size' in training and training.batch_size <= 0:
                raise ValueError("批次大小必须大于0")
        
        # 检查注意力配置
        if 'attention' in config:
            attention = config.attention
            
            if 'n_heads' in attention and 'd_model' in config.model:
                if config.model.d_model % attention.n_heads != 0:
                    raise ValueError("d_model必须能被n_heads整除")
    
    def save_config(self, config: DictConfig, output_path: str):
        """保存配置到文件"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            OmegaConf.save(config, f)
    
    def create_experiment_config(self, base_config: str, experiment_name: str, 
                               modifications: Dict[str, Any]) -> DictConfig:
        """创建实验配置"""
        
        # 加载基础配置
        base = self.load_config([base_config])
        
        # 应用实验修改
        experiment_config = self._deep_merge(OmegaConf.to_container(base), modifications)
        
        # 添加实验信息
        experiment_config['experiment'] = {
            'name': experiment_name,
            'base_config': base_config,
            'modifications': modifications,
            'created_at': str(pd.Timestamp.now())
        }
        
        return OmegaConf.create(experiment_config)

# 使用示例
config_manager = ConfigManager()

# 加载多个配置文件
config = config_manager.load_config([
    'base_config.yaml',
    'attention_config.yaml',
    'loss_config.yaml'
], overrides={
    'training': {'learning_rate': 5e-5},
    'model': {'d_model': 768}
})

# 创建实验配置
experiment_config = config_manager.create_experiment_config(
    'base_config.yaml',
    'attention_comparison',
    {
        'attention': {'type': 'LinearAttention'},
        'training': {'epochs': 50}
    }
)
```

---

## 🧠 模型架构配置

### 基础架构参数

```yaml
# 模型核心参数
model:
  # 基础维度
  d_model: 512          # 模型维度
  d_ff: 2048           # 前馈网络维度
  n_layers: 6          # Transformer层数
  n_heads: 8           # 注意力头数
  
  # Dropout配置
  dropout: 0.1         # 主dropout率
  attention_dropout: 0.1  # 注意力dropout
  ff_dropout: 0.1      # 前馈网络dropout
  
  # 激活函数
  activation: "gelu"    # relu, gelu, swish, mish
  
  # 层归一化
  layer_norm:
    type: "pre"         # pre, post, both
    eps: 1e-6
  
  # 残差连接
  residual:
    scale: 1.0
    dropout: 0.1
```

### 高级架构选项

```python
# advanced_model_config.py
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

@dataclass
class AdvancedModelConfig:
    """高级模型配置"""
    
    # 架构变体
    architecture_type: str = "standard"  # standard, lightweight, heavy
    
    # 层配置
    layer_configs: Optional[List[Dict]] = None  # 每层独立配置
    
    # 跳跃连接
    skip_connections: bool = True
    skip_layers: List[int] = None  # 指定跳跃层
    
    # 权重共享
    weight_sharing: Dict[str, Any] = None
    
    # 渐进式训练
    progressive_training: Dict[str, Any] = None
    
    # 知识蒸馏
    distillation: Dict[str, Any] = None

class ModelArchitectureBuilder:
    """模型架构构建器"""
    
    def __init__(self, config: AdvancedModelConfig):
        self.config = config
    
    def build_progressive_model(self):
        """构建渐进式模型"""
        
        if not self.config.progressive_training:
            return self.build_standard_model()
        
        stages = self.config.progressive_training.get('stages', [])
        
        class ProgressiveVIVTransformer(nn.Module):
            def __init__(self, stages_config):
                super().__init__()
                self.stages = nn.ModuleList()
                self.current_stage = 0
                
                for stage_config in stages_config:
                    stage_model = self._build_stage(stage_config)
                    self.stages.append(stage_model)
            
            def forward(self, x, stage=None):
                if stage is None:
                    stage = self.current_stage
                
                return self.stages[stage](x)
            
            def advance_stage(self):
                if self.current_stage < len(self.stages) - 1:
                    self.current_stage += 1
                    # 复制权重到新阶段
                    self._transfer_weights()
            
            def _transfer_weights(self):
                # 权重迁移逻辑
                prev_stage = self.stages[self.current_stage - 1]
                curr_stage = self.stages[self.current_stage]
                
                # 复制共同层的权重
                for name, param in prev_stage.named_parameters():
                    if hasattr(curr_stage, name):
                        getattr(curr_stage, name).data.copy_(param.data)
        
        return ProgressiveVIVTransformer(stages)
    
    def build_lightweight_model(self):
        """构建轻量级模型"""
        
        lightweight_config = {
            'd_model': 256,
            'n_layers': 4,
            'n_heads': 4,
            'd_ff': 1024,
            'attention_type': 'LinearAttention',
            'use_depthwise_conv': True,
            'parameter_sharing': True
        }
        
        return self._build_efficient_model(lightweight_config)
    
    def build_heavy_model(self):
        """构建重型模型"""
        
        heavy_config = {
            'd_model': 1024,
            'n_layers': 12,
            'n_heads': 16,
            'd_ff': 4096,
            'attention_type': 'MultiHeadAttention',
            'use_expert_layers': True,
            'gradient_checkpointing': True
        }
        
        return self._build_expert_model(heavy_config)
    
    def _build_efficient_model(self, config):
        """构建高效模型"""
        
        class EfficientVIVTransformer(nn.Module):
            def __init__(self, config):
                super().__init__()
                
                # 使用深度可分离卷积
                if config.get('use_depthwise_conv', False):
                    self.input_conv = DepthwiseSeparableConv1d(
                        config['input_dim'], config['d_model']
                    )
                
                # 参数共享的Transformer层
                if config.get('parameter_sharing', False):
                    shared_layer = TransformerLayer(config)
                    self.layers = nn.ModuleList([shared_layer] * config['n_layers'])
                else:
                    self.layers = nn.ModuleList([
                        TransformerLayer(config) for _ in range(config['n_layers'])
                    ])
            
            def forward(self, x):
                if hasattr(self, 'input_conv'):
                    x = self.input_conv(x)
                
                for layer in self.layers:
                    x = layer(x)
                
                return x
        
        return EfficientVIVTransformer(config)
```

---

## 🎯 注意力机制定制

### 注意力类型配置

```yaml
# 标准多头注意力
attention:
  type: "MultiHeadAttention"
  n_heads: 8
  d_k: 64  # 键维度
  d_v: 64  # 值维度
  dropout: 0.1
  
  # 缩放因子
  scale_factor: null  # 自动计算 (1/sqrt(d_k))
  
  # 偏置项
  use_qkv_bias: true
  use_output_bias: true
```

```yaml
# 线性注意力
attention:
  type: "LinearAttention"
  feature_map: "elu"  # elu, relu, softmax
  eps: 1e-6
  
  # 核函数配置
  kernel:
    type: "polynomial"  # polynomial, rbf, linear
    degree: 2  # 多项式度数
    gamma: 1.0  # RBF参数
```

```yaml
# 稀疏注意力
attention:
  type: "SparseAttention"
  sparsity_pattern: "local"  # local, strided, random
  
  # 局部注意力窗口
  local_window_size: 64
  
  # 步长注意力
  stride: 8
  
  # 随机注意力
  random_ratio: 0.1
```

### 自定义注意力机制

```python
# custom_attention.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

class PhysicsInformedAttention(nn.Module):
    """物理信息注意力机制"""
    
    def __init__(self, d_model: int, n_heads: int, physics_constraints: dict):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # 标准注意力组件
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        # 物理约束参数
        self.physics_constraints = physics_constraints
        
        # 物理先验权重
        self.physics_weight = nn.Parameter(torch.tensor(0.1))
        
        # 距离编码
        if 'spatial_decay' in physics_constraints:
            self.spatial_decay = nn.Parameter(torch.tensor(1.0))
        
        # 时间衰减
        if 'temporal_decay' in physics_constraints:
            self.temporal_decay = nn.Parameter(torch.tensor(1.0))
    
    def forward(self, x: torch.Tensor, 
                spatial_coords: Optional[torch.Tensor] = None,
                temporal_coords: Optional[torch.Tensor] = None) -> torch.Tensor:
        
        batch_size, seq_len, _ = x.shape
        
        # 计算Q, K, V
        Q = self.w_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # 标准注意力分数
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)
        
        # 应用物理约束
        if 'spatial_decay' in self.physics_constraints and spatial_coords is not None:
            spatial_mask = self._compute_spatial_mask(spatial_coords)
            attention_scores = attention_scores + self.physics_weight * spatial_mask
        
        if 'temporal_decay' in self.physics_constraints and temporal_coords is not None:
            temporal_mask = self._compute_temporal_mask(temporal_coords)
            attention_scores = attention_scores + self.physics_weight * temporal_mask
        
        # 连续性约束
        if 'continuity' in self.physics_constraints:
            continuity_mask = self._compute_continuity_mask(x)
            attention_scores = attention_scores + self.physics_weight * continuity_mask
        
        # Softmax归一化
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        # 应用注意力
        output = torch.matmul(attention_weights, V)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        return self.w_o(output)
    
    def _compute_spatial_mask(self, spatial_coords: torch.Tensor) -> torch.Tensor:
        """计算空间衰减掩码"""
        # spatial_coords: [batch_size, seq_len, spatial_dim]
        
        # 计算空间距离
        coords_expanded_i = spatial_coords.unsqueeze(2)  # [B, L, 1, D]
        coords_expanded_j = spatial_coords.unsqueeze(1)  # [B, 1, L, D]
        
        spatial_distances = torch.norm(coords_expanded_i - coords_expanded_j, dim=-1)
        
        # 应用指数衰减
        spatial_mask = -self.spatial_decay * spatial_distances
        
        return spatial_mask.unsqueeze(1)  # [B, 1, L, L] for broadcasting
    
    def _compute_temporal_mask(self, temporal_coords: torch.Tensor) -> torch.Tensor:
        """计算时间衰减掩码"""
        # temporal_coords: [batch_size, seq_len, 1]
        
        time_diff = temporal_coords.unsqueeze(2) - temporal_coords.unsqueeze(1)
        temporal_mask = -self.temporal_decay * torch.abs(time_diff).squeeze(-1)
        
        return temporal_mask.unsqueeze(1)  # [B, 1, L, L]
    
    def _compute_continuity_mask(self, x: torch.Tensor) -> torch.Tensor:
        """计算连续性约束掩码"""
        # 基于特征相似性的连续性
        x_norm = F.normalize(x, dim=-1)
        similarity = torch.matmul(x_norm, x_norm.transpose(-2, -1))
        
        # 增强相邻位置的注意力
        continuity_bonus = torch.zeros_like(similarity)
        for i in range(similarity.size(-1) - 1):
            continuity_bonus[:, i, i+1] = similarity[:, i, i+1]
            continuity_bonus[:, i+1, i] = similarity[:, i+1, i]
        
        return continuity_bonus.unsqueeze(1)

class AdaptiveAttention(nn.Module):
    """自适应注意力机制"""
    
    def __init__(self, d_model: int, n_heads: int, adaptation_config: dict):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        
        # 多种注意力机制
        self.attention_types = nn.ModuleDict({
            'standard': MultiHeadAttention(d_model, n_heads),
            'linear': LinearAttention(d_model, n_heads),
            'sparse': SparseAttention(d_model, n_heads)
        })
        
        # 注意力选择网络
        self.attention_selector = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.ReLU(),
            nn.Linear(d_model // 4, len(self.attention_types)),
            nn.Softmax(dim=-1)
        )
        
        # 自适应参数
        self.adaptation_config = adaptation_config
        
        # 温度参数
        self.temperature = nn.Parameter(torch.tensor(1.0))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, d_model = x.shape
        
        # 计算注意力选择权重
        # 使用全局平均池化获取序列表示
        global_repr = x.mean(dim=1)  # [B, D]
        attention_weights = self.attention_selector(global_repr)  # [B, num_attention_types]
        
        # 计算每种注意力的输出
        attention_outputs = {}
        for name, attention_module in self.attention_types.items():
            attention_outputs[name] = attention_module(x)
        
        # 加权组合
        output = torch.zeros_like(x)
        for i, (name, attention_output) in enumerate(attention_outputs.items()):
            weight = attention_weights[:, i].unsqueeze(1).unsqueeze(2)  # [B, 1, 1]
            output += weight * attention_output
        
        return output

# 注册自定义注意力
from vivtransformer.attention import AttentionFactory

AttentionFactory.register_attention('PhysicsInformed', PhysicsInformedAttention)
AttentionFactory.register_attention('Adaptive', AdaptiveAttention)
```

---

## 📊 损失函数配置

### 多损失函数组合

```yaml
loss:
  # 主损失函数
  primary:
    type: "MSELoss"
    weight: 1.0
    reduction: "mean"
  
  # 辅助损失函数
  auxiliary:
    - type: "SVDLoss"
      weight: 0.1
      rank_penalty: 0.01
      target_rank: 10
      
    - type: "PhysicsLoss"
      weight: 0.05
      constraints:
        - "continuity"
        - "momentum_conservation"
        - "energy_conservation"
      
    - type: "RegularizationLoss"
      weight: 0.01
      l1_weight: 0.001
      l2_weight: 0.01
      
    - type: "PerceptualLoss"
      weight: 0.02
      feature_layers: [2, 4, 6]
      
  # 动态权重调整
  dynamic_weighting:
    enabled: true
    method: "uncertainty"  # uncertainty, gradient_norm, adaptive
    update_frequency: 100  # 每100步更新一次
    
  # 课程学习
  curriculum:
    enabled: true
    schedule: "linear"  # linear, exponential, cosine
    start_weights: [1.0, 0.0, 0.0, 0.0, 0.0]
    end_weights: [1.0, 0.1, 0.05, 0.01, 0.02]
    total_steps: 10000
```

### 自定义损失函数

```python
# custom_losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional

class VIVPhysicsLoss(nn.Module):
    """涡激振动物理损失函数"""
    
    def __init__(self, constraints: List[str], weights: Dict[str, float] = None):
        super().__init__()
        self.constraints = constraints
        self.weights = weights or {constraint: 1.0 for constraint in constraints}
        
        # 物理常数
        self.reynolds_number = 100.0
        self.strouhal_number = 0.2
    
    def forward(self, prediction: torch.Tensor, target: torch.Tensor, 
                velocity_field: torch.Tensor, pressure_field: torch.Tensor) -> torch.Tensor:
        
        total_loss = 0.0
        
        # 连续性方程约束
        if 'continuity' in self.constraints:
            continuity_loss = self._continuity_constraint(velocity_field)
            total_loss += self.weights['continuity'] * continuity_loss
        
        # 动量守恒约束
        if 'momentum_conservation' in self.constraints:
            momentum_loss = self._momentum_constraint(velocity_field, pressure_field)
            total_loss += self.weights['momentum_conservation'] * momentum_loss
        
        # 能量守恒约束
        if 'energy_conservation' in self.constraints:
            energy_loss = self._energy_constraint(velocity_field)
            total_loss += self.weights['energy_conservation'] * energy_loss
        
        # 边界条件约束
        if 'boundary_conditions' in self.constraints:
            boundary_loss = self._boundary_constraint(velocity_field)
            total_loss += self.weights['boundary_conditions'] * boundary_loss
        
        return total_loss
    
    def _continuity_constraint(self, velocity_field: torch.Tensor) -> torch.Tensor:
        """连续性方程: ∇·u = 0"""
        # velocity_field: [B, T, H, W, 2] (u, v components)
        
        u = velocity_field[..., 0]  # x方向速度
        v = velocity_field[..., 1]  # y方向速度
        
        # 计算散度
        du_dx = torch.gradient(u, dim=-1)[0]
        dv_dy = torch.gradient(v, dim=-2)[0]
        
        divergence = du_dx + dv_dy
        
        # 最小化散度的L2范数
        continuity_loss = torch.mean(divergence ** 2)
        
        return continuity_loss
    
    def _momentum_constraint(self, velocity_field: torch.Tensor, 
                           pressure_field: torch.Tensor) -> torch.Tensor:
        """动量方程约束"""
        
        u = velocity_field[..., 0]
        v = velocity_field[..., 1]
        p = pressure_field
        
        # 时间导数
        du_dt = torch.gradient(u, dim=1)[0]
        dv_dt = torch.gradient(v, dim=1)[0]
        
        # 空间导数
        du_dx = torch.gradient(u, dim=-1)[0]
        du_dy = torch.gradient(u, dim=-2)[0]
        dv_dx = torch.gradient(v, dim=-1)[0]
        dv_dy = torch.gradient(v, dim=-2)[0]
        
        # 压力梯度
        dp_dx = torch.gradient(p, dim=-1)[0]
        dp_dy = torch.gradient(p, dim=-2)[0]
        
        # 拉普拉斯算子
        d2u_dx2 = torch.gradient(du_dx, dim=-1)[0]
        d2u_dy2 = torch.gradient(du_dy, dim=-2)[0]
        d2v_dx2 = torch.gradient(dv_dx, dim=-1)[0]
        d2v_dy2 = torch.gradient(dv_dy, dim=-2)[0]
        
        laplacian_u = d2u_dx2 + d2u_dy2
        laplacian_v = d2v_dx2 + d2v_dy2
        
        # Navier-Stokes方程残差
        momentum_x = du_dt + u * du_dx + v * du_dy + dp_dx - (1/self.reynolds_number) * laplacian_u
        momentum_y = dv_dt + u * dv_dx + v * dv_dy + dp_dy - (1/self.reynolds_number) * laplacian_v
        
        momentum_loss = torch.mean(momentum_x ** 2) + torch.mean(momentum_y ** 2)
        
        return momentum_loss
    
    def _energy_constraint(self, velocity_field: torch.Tensor) -> torch.Tensor:
        """能量守恒约束"""
        
        # 计算动能
        kinetic_energy = 0.5 * torch.sum(velocity_field ** 2, dim=-1)
        
        # 时间导数
        dE_dt = torch.gradient(kinetic_energy, dim=1)[0]
        
        # 能量耗散（粘性耗散）
        u = velocity_field[..., 0]
        v = velocity_field[..., 1]
        
        du_dx = torch.gradient(u, dim=-1)[0]
        du_dy = torch.gradient(u, dim=-2)[0]
        dv_dx = torch.gradient(v, dim=-1)[0]
        dv_dy = torch.gradient(v, dim=-2)[0]
        
        # 应变率张量
        strain_rate = 0.5 * ((du_dx + du_dx) ** 2 + (dv_dy + dv_dy) ** 2 + 
                            (du_dy + dv_dx) ** 2)
        
        viscous_dissipation = (2 / self.reynolds_number) * strain_rate
        
        # 能量平衡
        energy_balance = dE_dt + viscous_dissipation
        energy_loss = torch.mean(energy_balance ** 2)
        
        return energy_loss
    
    def _boundary_constraint(self, velocity_field: torch.Tensor) -> torch.Tensor:
        """边界条件约束"""
        
        # 假设边界在网格边缘
        boundary_loss = 0.0
        
        # 上下边界（自由滑移）
        top_boundary = velocity_field[:, :, 0, :, 1]  # v = 0 at top
        bottom_boundary = velocity_field[:, :, -1, :, 1]  # v = 0 at bottom
        
        boundary_loss += torch.mean(top_boundary ** 2) + torch.mean(bottom_boundary ** 2)
        
        # 左右边界（入口/出口条件）
        # 这里可以根据具体问题设置
        
        return boundary_loss

class AdaptiveLossWeighting(nn.Module):
    """自适应损失权重调整"""
    
    def __init__(self, num_losses: int, method: str = "uncertainty"):
        super().__init__()
        self.num_losses = num_losses
        self.method = method
        
        if method == "uncertainty":
            # 学习不确定性权重
            self.log_vars = nn.Parameter(torch.zeros(num_losses))
        elif method == "gradient_norm":
            # 基于梯度范数的权重
            self.gradient_weights = torch.ones(num_losses)
        
        self.loss_history = []
    
    def forward(self, losses: List[torch.Tensor], 
                model: Optional[nn.Module] = None) -> torch.Tensor:
        
        if self.method == "uncertainty":
            return self._uncertainty_weighting(losses)
        elif self.method == "gradient_norm":
            return self._gradient_norm_weighting(losses, model)
        elif self.method == "adaptive":
            return self._adaptive_weighting(losses)
        else:
            # 简单加权
            return sum(losses)
    
    def _uncertainty_weighting(self, losses: List[torch.Tensor]) -> torch.Tensor:
        """基于不确定性的权重调整"""
        
        weighted_loss = 0.0
        
        for i, loss in enumerate(losses):
            # 权重 = 1 / (2 * σ²), 其中 σ² = exp(log_var)
            precision = torch.exp(-self.log_vars[i])
            weighted_loss += precision * loss + self.log_vars[i]
        
        return weighted_loss
    
    def _gradient_norm_weighting(self, losses: List[torch.Tensor], 
                                model: nn.Module) -> torch.Tensor:
        """基于梯度范数的权重调整"""
        
        if model is None:
            return sum(losses)
        
        # 计算每个损失的梯度范数
        gradient_norms = []
        
        for loss in losses:
            # 计算梯度
            grads = torch.autograd.grad(loss, model.parameters(), 
                                      retain_graph=True, create_graph=False)
            
            # 计算梯度范数
            grad_norm = torch.sqrt(sum(torch.sum(g ** 2) for g in grads))
            gradient_norms.append(grad_norm)
        
        # 归一化权重
        total_norm = sum(gradient_norms)
        weights = [norm / total_norm for norm in gradient_norms]
        
        # 加权损失
        weighted_loss = sum(w * loss for w, loss in zip(weights, losses))
        
        return weighted_loss
    
    def _adaptive_weighting(self, losses: List[torch.Tensor]) -> torch.Tensor:
        """自适应权重调整"""
        
        # 记录损失历史
        current_losses = [loss.item() for loss in losses]
        self.loss_history.append(current_losses)
        
        # 保持最近100步的历史
        if len(self.loss_history) > 100:
            self.loss_history.pop(0)
        
        if len(self.loss_history) < 10:
            # 初期使用等权重
            return sum(losses)
        
        # 计算损失变化率
        recent_losses = torch.tensor(self.loss_history[-10:])
        loss_trends = torch.mean(recent_losses, dim=0)
        
        # 根据损失趋势调整权重
        # 损失下降慢的给更高权重
        weights = 1.0 / (loss_trends + 1e-8)
        weights = weights / weights.sum()  # 归一化
        
        weighted_loss = sum(w * loss for w, loss in zip(weights, losses))
        
        return weighted_loss

# 损失函数工厂
class LossFactory:
    """损失函数工厂"""
    
    _losses = {
        'MSELoss': nn.MSELoss,
        'L1Loss': nn.L1Loss,
        'SmoothL1Loss': nn.SmoothL1Loss,
        'VIVPhysicsLoss': VIVPhysicsLoss,
        'AdaptiveLossWeighting': AdaptiveLossWeighting
    }
    
    @classmethod
    def create_loss(cls, loss_config: Dict) -> nn.Module:
        """创建损失函数"""
        
        loss_type = loss_config['type']
        
        if loss_type not in cls._losses:
            raise ValueError(f"不支持的损失函数类型: {loss_type}")
        
        loss_class = cls._losses[loss_type]
        
        # 移除type参数
        params = {k: v for k, v in loss_config.items() if k != 'type'}
        
        return loss_class(**params)
    
    @classmethod
    def register_loss(cls, name: str, loss_class: type):
        """注册新的损失函数"""
        cls._losses[name] = loss_class
```

---

## 🚀 训练策略配置

### 优化器配置

```yaml
optimizer:
  type: "AdamW"  # Adam, AdamW, SGD, RMSprop, Lion
  
  # 基础参数
  lr: 1e-4
  weight_decay: 0.01
  
  # Adam/AdamW特定参数
  betas: [0.9, 0.999]
  eps: 1e-8
  amsgrad: false
  
  # 分层学习率
  layer_wise_lr:
    enabled: true
    decay_factor: 0.9  # 每层递减因子
    
  # 参数组配置
  param_groups:
    - name: "embeddings"
      lr_multiplier: 0.1
      weight_decay: 0.0
      
    - name: "attention"
      lr_multiplier: 1.0
      weight_decay: 0.01
      
    - name: "feedforward"
      lr_multiplier: 1.0
      weight_decay: 0.01
```

### 学习率调度

```yaml
scheduler:
  type: "CosineAnnealingWarmRestarts"  # StepLR, ExponentialLR, ReduceLROnPlateau
  
  # 余弦退火参数
  T_0: 1000  # 初始重启周期
  T_mult: 2  # 周期倍增因子
  eta_min: 1e-6  # 最小学习率
  
  # 预热配置
  warmup:
    enabled: true
    steps: 1000
    start_lr: 1e-6
    
  # 学习率查找
  lr_finder:
    enabled: false
    start_lr: 1e-8
    end_lr: 1e-1
    num_steps: 1000
```

### 高级训练策略

```python
# advanced_training.py
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import _LRScheduler
from typing import Dict, List, Optional, Callable

class CurriculumLearning:
    """课程学习策略"""
    
    def __init__(self, curriculum_config: Dict):
        self.config = curriculum_config
        self.current_stage = 0
        self.stage_progress = 0
        
        self.stages = curriculum_config.get('stages', [])
        self.transition_criteria = curriculum_config.get('transition_criteria', 'steps')
        
    def get_current_config(self, step: int, metrics: Dict = None) -> Dict:
        """获取当前阶段的配置"""
        
        if self.current_stage >= len(self.stages):
            return self.stages[-1]  # 返回最后阶段配置
        
        current_stage_config = self.stages[self.current_stage]
        
        # 检查是否需要进入下一阶段
        if self._should_advance_stage(step, metrics):
            self.current_stage += 1
            self.stage_progress = 0
            
            if self.current_stage < len(self.stages):
                current_stage_config = self.stages[self.current_stage]
        
        return current_stage_config
    
    def _should_advance_stage(self, step: int, metrics: Dict = None) -> bool:
        """判断是否应该进入下一阶段"""
        
        if self.current_stage >= len(self.stages) - 1:
            return False
        
        current_stage = self.stages[self.current_stage]
        
        if self.transition_criteria == 'steps':
            return step >= current_stage.get('duration', 1000)
        
        elif self.transition_criteria == 'loss' and metrics:
            target_loss = current_stage.get('target_loss', 0.1)
            current_loss = metrics.get('loss', float('inf'))
            return current_loss <= target_loss
        
        elif self.transition_criteria == 'accuracy' and metrics:
            target_accuracy = current_stage.get('target_accuracy', 0.9)
            current_accuracy = metrics.get('accuracy', 0.0)
            return current_accuracy >= target_accuracy
        
        return False

class GradientAccumulation:
    """梯度累积策略"""
    
    def __init__(self, accumulation_steps: int, 
                 dynamic_accumulation: bool = False):
        self.accumulation_steps = accumulation_steps
        self.dynamic_accumulation = dynamic_accumulation
        self.current_step = 0
        self.accumulated_loss = 0.0
        
    def accumulate_gradients(self, loss: torch.Tensor, 
                           model: nn.Module, 
                           optimizer: torch.optim.Optimizer,
                           memory_usage: float = None) -> bool:
        """累积梯度并返回是否应该更新参数"""
        
        # 动态调整累积步数
        if self.dynamic_accumulation and memory_usage:
            self.accumulation_steps = self._adjust_accumulation_steps(memory_usage)
        
        # 缩放损失
        scaled_loss = loss / self.accumulation_steps
        scaled_loss.backward()
        
        self.accumulated_loss += scaled_loss.item()
        self.current_step += 1
        
        # 检查是否应该更新参数
        if self.current_step % self.accumulation_steps == 0:
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # 更新参数
            optimizer.step()
            optimizer.zero_grad()
            
            # 重置累积状态
            avg_loss = self.accumulated_loss
            self.accumulated_loss = 0.0
            
            return True, avg_loss
        
        return False, self.accumulated_loss
    
    def _adjust_accumulation_steps(self, memory_usage: float) -> int:
        """根据内存使用情况动态调整累积步数"""
        
        if memory_usage > 0.9:  # 内存使用超过90%
            return min(self.accumulation_steps * 2, 32)
        elif memory_usage < 0.5:  # 内存使用低于50%
            return max(self.accumulation_steps // 2, 1)
        else:
            return self.accumulation_steps

class MixedPrecisionTraining:
    """混合精度训练"""
    
    def __init__(self, enabled: bool = True, 
                 loss_scale: str = "dynamic",
                 init_scale: float = 2**16):
        self.enabled = enabled
        
        if enabled:
            from torch.cuda.amp import GradScaler, autocast
            self.scaler = GradScaler(
                init_scale=init_scale,
                enabled=(loss_scale == "dynamic")
            )
            self.autocast = autocast
        else:
            self.scaler = None
            self.autocast = None
    
    def forward_pass(self, model: nn.Module, inputs: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        
        if self.enabled and self.autocast:
            with self.autocast():
                return model(inputs)
        else:
            return model(inputs)
    
    def backward_pass(self, loss: torch.Tensor, 
                     model: nn.Module, 
                     optimizer: torch.optim.Optimizer) -> bool:
        """反向传播"""
        
        if self.enabled and self.scaler:
            # 缩放损失
            self.scaler.scale(loss).backward()
            
            # 梯度裁剪
            self.scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # 更新参数
            self.scaler.step(optimizer)
            self.scaler.update()
            
            return True
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            return True

class EarlyStopping:
    """早停策略"""
    
    def __init__(self, patience: int = 10, 
                 min_delta: float = 1e-4,
                 monitor: str = "val_loss",
                 mode: str = "min"):
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor
        self.mode = mode
        
        self.best_score = None
        self.counter = 0
        self.early_stop = False
        
        self.is_better = self._get_comparison_function()
    
    def __call__(self, metrics: Dict[str, float]) -> bool:
        """检查是否应该早停"""
        
        if self.monitor not in metrics:
            return False
        
        current_score = metrics[self.monitor]
        
        if self.best_score is None:
            self.best_score = current_score
        elif self.is_better(current_score, self.best_score):
            self.best_score = current_score
            self.counter = 0
        else:
            self.counter += 1
            
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop
    
    def _get_comparison_function(self) -> Callable:
        """获取比较函数"""
        
        if self.mode == "min":
            return lambda current, best: current < best - self.min_delta
        elif self.mode == "max":
            return lambda current, best: current > best + self.min_delta
        else:
            raise ValueError(f"不支持的模式: {self.mode}")

class ModelEMA:
    """模型指数移动平均"""
    
    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        # 初始化影子参数
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self, model: nn.Module):
        """更新EMA参数"""
        
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name] = (self.decay * self.shadow[name] + 
                                   (1 - self.decay) * param.data)
    
    def apply_shadow(self, model: nn.Module):
        """应用EMA参数到模型"""
        
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])
    
    def restore(self, model: nn.Module):
        """恢复原始参数"""
        
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data.copy_(self.backup[name])
        
        self.backup.clear()
```

---

## 📈 性能优化配置

### 内存优化

```yaml
performance:
  # 内存优化
  memory:
    gradient_checkpointing: true
    cpu_offload: false
    pin_memory: true
    
    # 动态批次大小
    dynamic_batch_size:
      enabled: true
      min_batch_size: 8
      max_batch_size: 128
      memory_threshold: 0.9
    
    # 内存映射
    memory_mapping:
      enabled: true
      cache_size: "1GB"
  
  # 计算优化
  compute:
    mixed_precision: true
    compile_model: true  # PyTorch 2.0+
    
    # 融合操作
    fused_ops:
      attention: true
      layer_norm: true
      activation: true
    
    # 并行策略
    parallelism:
      data_parallel: true
      model_parallel: false
      pipeline_parallel: false
  
  # I/O优化
  io:
    num_workers: 8
    prefetch_factor: 2
    persistent_workers: true
    
    # 数据预加载
    preload_data: true
    async_loading: true
```

### 分布式训练配置

```yaml
distributed:
  # 分布式策略
  strategy: "ddp"  # ddp, fsdp, deepspeed
  
  # DDP配置
  ddp:
    find_unused_parameters: false
    gradient_as_bucket_view: true
    static_graph: true
  
  # FSDP配置
  fsdp:
    sharding_strategy: "FULL_SHARD"
    cpu_offload: false
    mixed_precision: true
    
  # DeepSpeed配置
  deepspeed:
    config_file: "deepspeed_config.json"
    zero_stage: 2
    
  # 通信优化
  communication:
    backend: "nccl"  # nccl, gloo, mpi
    timeout: 1800
    
    # 梯度压缩
    gradient_compression:
      enabled: true
      algorithm: "topk"  # topk, randomk, threshold
      compression_ratio: 0.1
```

---

## 🔬 实验管理配置

### 实验跟踪

```yaml
experiment:
  # 基础信息
  name: "viv_transformer_experiment"
  description: "涡激振动Transformer模型实验"
  tags: ["viv", "transformer", "attention"]
  
  # 版本控制
  version_control:
    enabled: true
    track_code: true
    track_data: true
    track_config: true
  
  # 日志记录
  logging:
    # TensorBoard
    tensorboard:
      enabled: true
      log_dir: "./runs"
      log_graph: true
      log_images: true
      
    # Weights & Biases
    wandb:
      enabled: false
      project: "viv-transformer"
      entity: "your-team"
      
    # MLflow
    mlflow:
      enabled: false
      tracking_uri: "http://localhost:5000"
      experiment_name: "VIVTransformer"
  
  # 检查点管理
  checkpointing:
    save_frequency: 1000  # 每1000步保存一次
    keep_last_n: 5  # 保留最近5个检查点
    save_best: true  # 保存最佳模型
    
    # 自动恢复
    auto_resume: true
    resume_from_latest: true
  
  # 超参数搜索
  hyperparameter_search:
    enabled: false
    method: "optuna"  # optuna, ray_tune, hyperopt
    
    # 搜索空间
    search_space:
      learning_rate:
        type: "loguniform"
        low: 1e-5
        high: 1e-3
      
      batch_size:
        type: "choice"
        choices: [16, 32, 64, 128]
      
      attention_type:
        type: "categorical"
        choices: ["MultiHeadAttention", "LinearAttention", "SparseAttention"]
    
    # 优化目标
    objective:
      metric: "val_loss"
      direction: "minimize"
    
    # 搜索配置
    n_trials: 100
    timeout: 3600  # 1小时
```

### 自动化实验

```python
# experiment_automation.py
import optuna
import mlflow
import wandb
from typing import Dict, Any, List
from pathlib import Path
import json
import yaml

class ExperimentManager:
    """实验管理器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.experiment_dir = Path(config.get('experiment_dir', './experiments'))
        self.experiment_dir.mkdir(exist_ok=True)
        
        # 初始化日志记录器
        self._init_loggers()
    
    def _init_loggers(self):
        """初始化日志记录器"""
        
        self.loggers = {}
        
        # TensorBoard
        if self.config.get('logging', {}).get('tensorboard', {}).get('enabled', False):
            from torch.utils.tensorboard import SummaryWriter
            tb_config = self.config['logging']['tensorboard']
            self.loggers['tensorboard'] = SummaryWriter(
                log_dir=tb_config.get('log_dir', './runs')
            )
        
        # Weights & Biases
        if self.config.get('logging', {}).get('wandb', {}).get('enabled', False):
            wandb_config = self.config['logging']['wandb']
            wandb.init(
                project=wandb_config.get('project', 'viv-transformer'),
                entity=wandb_config.get('entity'),
                config=self.config
            )
            self.loggers['wandb'] = wandb
        
        # MLflow
        if self.config.get('logging', {}).get('mlflow', {}).get('enabled', False):
            mlflow_config = self.config['logging']['mlflow']
            mlflow.set_tracking_uri(mlflow_config.get('tracking_uri'))
            mlflow.set_experiment(mlflow_config.get('experiment_name', 'VIVTransformer'))
            self.loggers['mlflow'] = mlflow
    
    def log_metrics(self, metrics: Dict[str, float], step: int):
        """记录指标"""
        
        # TensorBoard
        if 'tensorboard' in self.loggers:
            for name, value in metrics.items():
                self.loggers['tensorboard'].add_scalar(name, value, step)
        
        # Weights & Biases
        if 'wandb' in self.loggers:
            self.loggers['wandb'].log(metrics, step=step)
        
        # MLflow
        if 'mlflow' in self.loggers:
            for name, value in metrics.items():
                mlflow.log_metric(name, value, step=step)
    
    def save_experiment_config(self, experiment_name: str):
        """保存实验配置"""
        
        experiment_path = self.experiment_dir / experiment_name
        experiment_path.mkdir(exist_ok=True)
        
        # 保存配置
        config_path = experiment_path / 'config.yaml'
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.config, f, default_flow_style=False)
        
        # 保存实验元数据
        metadata = {
            'experiment_name': experiment_name,
            'created_at': str(pd.Timestamp.now()),
            'config_file': str(config_path)
        }
        
        metadata_path = experiment_path / 'metadata.json'
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)

class HyperparameterOptimizer:
    """超参数优化器"""
    
    def __init__(self, search_config: Dict[str, Any]):
        self.search_config = search_config
        self.method = search_config.get('method', 'optuna')
        
        if self.method == 'optuna':
            self.study = optuna.create_study(
                direction=search_config['objective']['direction']
            )
    
    def optimize(self, objective_function: callable, n_trials: int = 100):
        """执行超参数优化"""
        
        if self.method == 'optuna':
            self.study.optimize(objective_function, n_trials=n_trials)
            return self.study.best_params
        else:
            raise NotImplementedError(f"不支持的优化方法: {self.method}")
    
    def suggest_hyperparameters(self, trial) -> Dict[str, Any]:
        """建议超参数"""
        
        params = {}
        search_space = self.search_config.get('search_space', {})
        
        for param_name, param_config in search_space.items():
            param_type = param_config['type']
            
            if param_type == 'loguniform':
                params[param_name] = trial.suggest_loguniform(
                    param_name, param_config['low'], param_config['high']
                )
            elif param_type == 'uniform':
                params[param_name] = trial.suggest_uniform(
                    param_name, param_config['low'], param_config['high']
                )
            elif param_type == 'int':
                params[param_name] = trial.suggest_int(
                    param_name, param_config['low'], param_config['high']
                )
            elif param_type == 'choice':
                params[param_name] = trial.suggest_categorical(
                    param_name, param_config['choices']
                )
            elif param_type == 'categorical':
                params[param_name] = trial.suggest_categorical(
                    param_name, param_config['choices']
                )
        
        return params

---

## 🚀 部署配置

### 生产环境配置

```yaml
deployment:
  # 环境配置
  environment: "production"  # development, staging, production
  
  # 服务配置
  service:
    host: "0.0.0.0"
    port: 8000
    workers: 4
    
    # 负载均衡
    load_balancer:
      enabled: true
      algorithm: "round_robin"  # round_robin, least_connections, ip_hash
      health_check:
        enabled: true
        interval: 30
        timeout: 10
  
  # 模型服务
  model_serving:
    # 模型格式
    format: "torchscript"  # pytorch, torchscript, onnx, tensorrt
    
    # 批处理
    batching:
      enabled: true
      max_batch_size: 32
      timeout_ms: 100
      
    # 缓存
    caching:
      enabled: true
      cache_size: "1GB"
      ttl: 3600  # 1小时
    
    # 模型版本管理
    versioning:
      enabled: true
      strategy: "blue_green"  # blue_green, canary, rolling
      
  # 监控配置
  monitoring:
    # 指标收集
    metrics:
      enabled: true
      endpoint: "/metrics"
      
    # 日志记录
    logging:
      level: "INFO"
      format: "json"
      
    # 健康检查
    health_check:
      endpoint: "/health"
      
  # 安全配置
  security:
    # API密钥
    api_key:
      enabled: true
      header_name: "X-API-Key"
      
    # 速率限制
    rate_limiting:
      enabled: true
      requests_per_minute: 1000
      
    # CORS
    cors:
      enabled: true
      allowed_origins: ["*"]
```

### Docker配置

```dockerfile
# Dockerfile
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

# 设置工作目录
WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 复制requirements文件
COPY requirements.txt .

# 安装Python依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY . .

# 设置环境变量
ENV PYTHONPATH=/app
ENV CUDA_VISIBLE_DEVICES=0

# 暴露端口
EXPOSE 8000

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# 启动命令
CMD ["python", "serve.py"]
```

```yaml
# docker-compose.yml
version: '3.8'

services:
  viv-transformer:
    build: .
    ports:
      - "8000:8000"
    environment:
      - CUDA_VISIBLE_DEVICES=0
      - MODEL_PATH=/app/models/best_model.pt
    volumes:
      - ./models:/app/models
      - ./logs:/app/logs
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    restart: unless-stopped
    
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - viv-transformer
    restart: unless-stopped
    
  prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
    restart: unless-stopped
    
  grafana:
    image: grafana/grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana-storage:/var/lib/grafana
    restart: unless-stopped

volumes:
  grafana-storage:
```

### Kubernetes配置

```yaml
# k8s-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: viv-transformer
  labels:
    app: viv-transformer
spec:
  replicas: 3
  selector:
    matchLabels:
      app: viv-transformer
  template:
    metadata:
      labels:
        app: viv-transformer
    spec:
      containers:
      - name: viv-transformer
        image: viv-transformer:latest
        ports:
        - containerPort: 8000
        env:
        - name: MODEL_PATH
          value: "/app/models/best_model.pt"
        resources:
          requests:
            memory: "2Gi"
            cpu: "1"
            nvidia.com/gpu: 1
          limits:
            memory: "4Gi"
            cpu: "2"
            nvidia.com/gpu: 1
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 60
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        volumeMounts:
        - name: model-storage
          mountPath: /app/models
      volumes:
      - name: model-storage
        persistentVolumeClaim:
          claimName: model-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: viv-transformer-service
spec:
  selector:
    app: viv-transformer
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer
```

---

## 📝 配置最佳实践

### 1. 配置文件组织

```
config/
├── base/
│   ├── model.yaml
│   ├── training.yaml
│   └── data.yaml
├── environments/
│   ├── development.yaml
│   ├── staging.yaml
│   └── production.yaml
├── experiments/
│   ├── attention_comparison.yaml
│   ├── loss_ablation.yaml
│   └── architecture_search.yaml
└── deployment/
    ├── docker.yaml
    ├── kubernetes.yaml
    └── monitoring.yaml
```

### 2. 环境变量管理

```bash
# .env文件
CUDA_VISIBLE_DEVICES=0,1
MODEL_PATH=/app/models/best_model.pt
LOG_LEVEL=INFO
API_KEY=your-secret-api-key
DATABASE_URL=postgresql://user:pass@localhost/db
```

### 3. 配置验证

```python
# config_validation.py
from pydantic import BaseModel, validator
from typing import List, Optional, Dict, Any

class ModelConfig(BaseModel):
    d_model: int
    n_heads: int
    n_layers: int
    dropout: float = 0.1
    
    @validator('d_model')
    def d_model_must_be_positive(cls, v):
        if v <= 0:
            raise ValueError('d_model must be positive')
        return v
    
    @validator('n_heads')
    def n_heads_must_divide_d_model(cls, v, values):
        if 'd_model' in values and values['d_model'] % v != 0:
            raise ValueError('d_model must be divisible by n_heads')
        return v

class TrainingConfig(BaseModel):
    epochs: int
    batch_size: int
    learning_rate: float
    
    @validator('learning_rate')
    def lr_must_be_positive(cls, v):
        if v <= 0:
            raise ValueError('learning_rate must be positive')
        return v

class VIVTransformerConfig(BaseModel):
    model: ModelConfig
    training: TrainingConfig
    
    class Config:
        extra = 'forbid'  # 禁止额外字段
```

### 4. 配置模板

```yaml
# config_template.yaml
# VIVTransformer配置模板
# 复制此文件并根据需要修改参数

# 模型配置
model:
  d_model: ${MODEL_DIM:512}  # 支持环境变量
  n_heads: ${N_HEADS:8}
  n_layers: ${N_LAYERS:6}
  dropout: ${DROPOUT:0.1}
  
# 训练配置
training:
  epochs: ${EPOCHS:100}
  batch_size: ${BATCH_SIZE:32}
  learning_rate: ${LEARNING_RATE:1e-4}
  
# 数据配置
data:
  sequence_length: ${SEQ_LENGTH:100}
  input_dim: ${INPUT_DIM:512}
  
# 实验配置
experiment:
  name: ${EXPERIMENT_NAME:"default_experiment"}
  description: ${EXPERIMENT_DESC:"Default VIVTransformer experiment"}
  tags: ${EXPERIMENT_TAGS:["viv", "transformer"]}
```

---

## 🔧 故障排除

### 常见配置问题

1. **配置文件格式错误**
   ```bash
   # 验证YAML语法
   python -c "import yaml; yaml.safe_load(open('config.yaml'))"
   ```

2. **环境变量未设置**
   ```bash
   # 检查环境变量
   echo $CUDA_VISIBLE_DEVICES
   env | grep VIV
   ```

3. **权限问题**
   ```bash
   # 检查文件权限
   ls -la config/
   chmod 644 config/*.yaml
   ```

4. **路径问题**
   ```python
   # 验证路径存在
   from pathlib import Path
   config_path = Path('config/base_config.yaml')
   assert config_path.exists(), f"配置文件不存在: {config_path}"
   ```

---

## 📚 相关资源

- 📖 [基础配置教程](/pages/quick-start-tutorial/)
- 🔧 [故障排除指南](/pages/troubleshooting/)
- 📊 [性能优化指南](/pages/performance-optimization/)
- 🚀 [部署指南](/pages/deployment-guide/)
- 💡 [最佳实践](/pages/best-practices/)

---

*本指南涵盖了VIVTransformer的高级配置选项。如需更多帮助，请参考[完整文档](/)或联系技术支持。*