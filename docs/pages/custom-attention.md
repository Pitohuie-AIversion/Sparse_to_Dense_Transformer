---
layout: doc
title: Custom Attention
description: 自定义注意力机制的实现指南
permalink: /pages/custom-attention/
---

# 自定义注意力机制 {#自定义注意力机制}

本文档详细介绍如何在VIVTransformer框架中创建和集成自定义注意力机制，包括设计原则、实现步骤、测试方法和最佳实践。

## 📋 目录 {#目录}

- [设计原则](#设计原则)
- [基础接口](#基础接口)
- [实现步骤](#实现步骤)
- [示例实现](#示例实现)
- [注册机制](#注册机制)
- [测试验证](#测试验证)
- [性能优化](#性能优化)
- [最佳实践](#最佳实践)

## 设计原则 {#设计原则}

### 🎯 核心原则 {#核心原则}

1. **统一接口**: 所有注意力机制必须遵循统一的接口规范
2. **模块化设计**: 每个注意力机制应该是独立的、可替换的模块
3. **配置驱动**: 通过配置文件控制注意力机制的选择和参数
4. **性能优先**: 优化计算效率和内存使用
5. **可扩展性**: 易于添加新的注意力变体

### 📐 设计约束 {#设计约束}

```python
# 输入输出约束 {#输入输出约束}
class AttentionConstraints:
    """
    注意力机制设计约束
    
    输入:
        - hidden_states: [batch_size, seq_len, d_model]
        - attention_mask: [batch_size, 1, 1, seq_len] (可选)
        - position_ids: [batch_size, seq_len] (可选)
    
    输出:
        - output: [batch_size, seq_len, d_model]
        - attention_weights: [batch_size, num_heads, seq_len, seq_len] (可选)
    
    约束:
        - 输入输出维度必须保持一致
        - 支持变长序列（通过attention_mask）
        - 内存使用应该可控
        - 计算复杂度应该明确
    """
    pass
```

## 基础接口 {#基础接口}

### 🔧 抽象基类 {#抽象基类}

```python
from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any

class BaseAttention(nn.Module, ABC):
    """注意力机制抽象基类
    
    所有自定义注意力机制都应该继承此类并实现必要的方法。
    """
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1, **kwargs):
        super(BaseAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = d_model // num_heads
        
        # 验证参数
        assert d_model % num_heads == 0, f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
        
        # 通用组件
        self.dropout_layer = nn.Dropout(dropout)
        
        # 子类特定初始化
        self._init_parameters(**kwargs)
    
    @abstractmethod
    def _init_parameters(self, **kwargs):
        """初始化特定参数
        
        子类应该在此方法中初始化自己的参数，如线性层、卷积层等。
        
        Args:
            **kwargs: 额外的初始化参数
        """
        pass
    
    @abstractmethod
    def compute_attention(self, 
                         query: torch.Tensor, 
                         key: torch.Tensor, 
                         value: torch.Tensor,
                         attention_mask: Optional[torch.Tensor] = None,
                         **kwargs) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """计算注意力
        
        Args:
            query: 查询张量 [batch_size, num_heads, seq_len, head_dim]
            key: 键张量 [batch_size, num_heads, seq_len, head_dim]
            value: 值张量 [batch_size, num_heads, seq_len, head_dim]
            attention_mask: 注意力掩码 [batch_size, 1, 1, seq_len]
            **kwargs: 额外参数
        
        Returns:
            output: 注意力输出 [batch_size, num_heads, seq_len, head_dim]
            attention_weights: 注意力权重 [batch_size, num_heads, seq_len, seq_len] (可选)
        """
        pass
    
    def forward(self, 
                hidden_states: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.Tensor] = None,
                return_attention_weights: bool = False,
                **kwargs) -> torch.Tensor:
        """前向传播
        
        Args:
            hidden_states: 输入隐藏状态 [batch_size, seq_len, d_model]
            attention_mask: 注意力掩码 [batch_size, 1, 1, seq_len]
            position_ids: 位置ID [batch_size, seq_len]
            return_attention_weights: 是否返回注意力权重
            **kwargs: 额外参数
        
        Returns:
            输出张量 [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, d_model = hidden_states.shape
        
        # 生成Q、K、V
        query, key, value = self.generate_qkv(hidden_states, position_ids, **kwargs)
        
        # 重塑为多头格式
        query = query.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 计算注意力
        attention_output, attention_weights = self.compute_attention(
            query, key, value, attention_mask, **kwargs
        )
        
        # 重塑回原始格式
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, d_model
        )
        
        # 输出投影
        output = self.output_projection(attention_output)
        
        if return_attention_weights:
            return output, attention_weights
        return output
    
    @abstractmethod
    def generate_qkv(self, 
                     hidden_states: torch.Tensor,
                     position_ids: Optional[torch.Tensor] = None,
                     **kwargs) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """生成查询、键、值张量
        
        Args:
            hidden_states: 输入隐藏状态 [batch_size, seq_len, d_model]
            position_ids: 位置ID [batch_size, seq_len]
            **kwargs: 额外参数
        
        Returns:
            query: 查询张量 [batch_size, seq_len, d_model]
            key: 键张量 [batch_size, seq_len, d_model]
            value: 值张量 [batch_size, seq_len, d_model]
        """
        pass
    
    @abstractmethod
    def output_projection(self, attention_output: torch.Tensor) -> torch.Tensor:
        """输出投影
        
        Args:
            attention_output: 注意力输出 [batch_size, seq_len, d_model]
        
        Returns:
            投影后的输出 [batch_size, seq_len, d_model]
        """
        pass
    
    def get_attention_info(self) -> Dict[str, Any]:
        """获取注意力机制信息
        
        Returns:
            包含注意力机制信息的字典
        """
        return {
            'type': self.__class__.__name__,
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'head_dim': self.head_dim,
            'dropout': self.dropout,
            'parameters': sum(p.numel() for p in self.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.parameters() if p.requires_grad)
        }
```

## 实现步骤 {#实现步骤}

### 📝 步骤1: 继承基类 {#步骤1-继承基类}

```python
class CustomAttention(BaseAttention):
    """自定义注意力机制示例
    
    这是一个示例实现，展示如何创建自定义注意力机制。
    """
    
    def _init_parameters(self, **kwargs):
        """初始化参数"""
        # 线性投影层
        self.q_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.k_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.v_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.out_proj = nn.Linear(self.d_model, self.d_model)
        
        # 自定义参数
        self.temperature = kwargs.get('temperature', 1.0)
        self.use_bias = kwargs.get('use_bias', True)
        
        if self.use_bias:
            self.bias = nn.Parameter(torch.zeros(self.num_heads, 1, 1))
        
        # 缩放因子
        self.scale = self.head_dim ** -0.5
```

### 📝 步骤2: 实现QKV生成 {#步骤2-实现qkv生成}

```python
    def generate_qkv(self, hidden_states, position_ids=None, **kwargs):
        """生成Q、K、V张量"""
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)
        
        # 可以在这里添加位置编码或其他变换
        if position_ids is not None:
            # 示例：添加位置相关的变换
            pos_encoding = self._get_position_encoding(position_ids)
            query = query + pos_encoding
            key = key + pos_encoding
        
        return query, key, value
    
    def _get_position_encoding(self, position_ids):
        """获取位置编码（示例实现）"""
        # 这里可以实现各种位置编码方案
        # 例如：正弦位置编码、学习位置编码、旋转位置编码等
        batch_size, seq_len = position_ids.shape
        pos_encoding = torch.zeros(batch_size, seq_len, self.d_model, device=position_ids.device)
        
        # 简单的正弦位置编码示例
        for pos in range(seq_len):
            for i in range(0, self.d_model, 2):
                pos_encoding[:, pos, i] = torch.sin(position_ids[:, pos] / (10000 ** (i / self.d_model)))
                if i + 1 < self.d_model:
                    pos_encoding[:, pos, i + 1] = torch.cos(position_ids[:, pos] / (10000 ** (i / self.d_model)))
        
        return pos_encoding
```

### 📝 步骤3: 实现注意力计算 {#步骤3-实现注意力计算}

```python
    def compute_attention(self, query, key, value, attention_mask=None, **kwargs):
        """计算自定义注意力"""
        # 计算注意力分数
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        
        # 应用温度参数
        attention_scores = attention_scores / self.temperature
        
        # 添加偏置（如果使用）
        if self.use_bias:
            attention_scores = attention_scores + self.bias
        
        # 应用注意力掩码
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        
        # 自定义注意力计算逻辑
        # 例如：使用不同的激活函数、添加噪声、应用稀疏化等
        attention_scores = self._apply_custom_logic(attention_scores, **kwargs)
        
        # Softmax归一化
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout_layer(attention_probs)
        
        # 应用注意力权重
        context = torch.matmul(attention_probs, value)
        
        return context, attention_probs
    
    def _apply_custom_logic(self, attention_scores, **kwargs):
        """应用自定义逻辑"""
        # 示例1：添加高斯噪声
        if kwargs.get('add_noise', False):
            noise = torch.randn_like(attention_scores) * kwargs.get('noise_std', 0.1)
            attention_scores = attention_scores + noise
        
        # 示例2：应用稀疏化
        if kwargs.get('apply_sparsity', False):
            sparsity_ratio = kwargs.get('sparsity_ratio', 0.1)
            k = int(attention_scores.size(-1) * (1 - sparsity_ratio))
            topk_values, topk_indices = torch.topk(attention_scores, k, dim=-1)
            sparse_scores = torch.full_like(attention_scores, float('-inf'))
            sparse_scores.scatter_(-1, topk_indices, topk_values)
            attention_scores = sparse_scores
        
        # 示例3：应用局部注意力窗口
        if kwargs.get('local_window', False):
            window_size = kwargs.get('window_size', 64)
            attention_scores = self._apply_local_window(attention_scores, window_size)
        
        return attention_scores
    
    def _apply_local_window(self, attention_scores, window_size):
        """应用局部注意力窗口"""
        seq_len = attention_scores.size(-1)
        mask = torch.zeros_like(attention_scores)
        
        for i in range(seq_len):
            start = max(0, i - window_size // 2)
            end = min(seq_len, i + window_size // 2 + 1)
            mask[..., i, start:end] = 1
        
        attention_scores = attention_scores.masked_fill(mask == 0, float('-inf'))
        return attention_scores
```

### 📝 步骤4: 实现输出投影 {#步骤4-实现输出投影}

```python
    def output_projection(self, attention_output):
        """输出投影"""
        return self.out_proj(attention_output)
```

## 示例实现 {#示例实现}

### 🌟 示例1: 多尺度注意力 {#示例1-多尺度注意力}

```python
@AttentionFactory.register("multiscale")
class MultiScaleAttention(BaseAttention):
    """多尺度注意力机制
    
    在不同尺度上计算注意力，然后融合结果。
    """
    
    def _init_parameters(self, scales=[1, 2, 4], **kwargs):
        self.scales = scales
        self.num_scales = len(scales)
        
        # 为每个尺度创建投影层
        self.scale_projections = nn.ModuleList([
            nn.ModuleDict({
                'q_proj': nn.Linear(self.d_model, self.d_model // self.num_scales, bias=False),
                'k_proj': nn.Linear(self.d_model, self.d_model // self.num_scales, bias=False),
                'v_proj': nn.Linear(self.d_model, self.d_model // self.num_scales, bias=False)
            }) for _ in self.scales
        ])
        
        # 融合层
        self.fusion = nn.Linear(self.d_model, self.d_model)
        self.out_proj = nn.Linear(self.d_model, self.d_model)
        
        self.scale = (self.d_model // self.num_scales // self.num_heads) ** -0.5
    
    def generate_qkv(self, hidden_states, position_ids=None, **kwargs):
        batch_size, seq_len, d_model = hidden_states.shape
        
        all_queries, all_keys, all_values = [], [], []
        
        for scale_idx, scale in enumerate(self.scales):
            # 下采样（如果scale > 1）
            if scale > 1:
                # 使用平均池化进行下采样
                pooled_states = F.avg_pool1d(
                    hidden_states.transpose(1, 2), 
                    kernel_size=scale, 
                    stride=scale
                ).transpose(1, 2)
            else:
                pooled_states = hidden_states
            
            # 生成Q、K、V
            proj = self.scale_projections[scale_idx]
            q = proj['q_proj'](pooled_states)
            k = proj['k_proj'](pooled_states)
            v = proj['v_proj'](pooled_states)
            
            # 上采样回原始长度（如果需要）
            if scale > 1:
                q = F.interpolate(q.transpose(1, 2), size=seq_len, mode='linear', align_corners=False).transpose(1, 2)
                k = F.interpolate(k.transpose(1, 2), size=seq_len, mode='linear', align_corners=False).transpose(1, 2)
                v = F.interpolate(v.transpose(1, 2), size=seq_len, mode='linear', align_corners=False).transpose(1, 2)
            
            all_queries.append(q)
            all_keys.append(k)
            all_values.append(v)
        
        # 拼接所有尺度
        query = torch.cat(all_queries, dim=-1)
        key = torch.cat(all_keys, dim=-1)
        value = torch.cat(all_values, dim=-1)
        
        return query, key, value
    
    def compute_attention(self, query, key, value, attention_mask=None, **kwargs):
        # 标准注意力计算
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout_layer(attention_probs)
        
        context = torch.matmul(attention_probs, value)
        
        return context, attention_probs
    
    def output_projection(self, attention_output):
        # 先通过融合层，再通过输出投影
        fused = self.fusion(attention_output)
        return self.out_proj(fused)
```

### 🌟 示例2: 自适应注意力 {#示例2-自适应注意力}

```python
@AttentionFactory.register("adaptive")
class AdaptiveAttention(BaseAttention):
    """自适应注意力机制
    
    根据输入动态调整注意力模式。
    """
    
    def _init_parameters(self, **kwargs):
        # 标准投影层
        self.q_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.k_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.v_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.out_proj = nn.Linear(self.d_model, self.d_model)
        
        # 自适应控制器
        self.controller = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 4),
            nn.ReLU(),
            nn.Linear(self.d_model // 4, self.num_heads),
            nn.Sigmoid()
        )
        
        # 多种注意力模式
        self.attention_modes = nn.ModuleDict({
            'global': GlobalAttentionMode(self.d_model, self.num_heads),
            'local': LocalAttentionMode(self.d_model, self.num_heads),
            'sparse': SparseAttentionMode(self.d_model, self.num_heads)
        })
        
        self.scale = self.head_dim ** -0.5
    
    def generate_qkv(self, hidden_states, position_ids=None, **kwargs):
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)
        return query, key, value
    
    def compute_attention(self, query, key, value, attention_mask=None, **kwargs):
        batch_size, num_heads, seq_len, head_dim = query.shape
        
        # 计算自适应权重
        # 使用全局平均池化获取序列表示
        global_repr = query.mean(dim=2)  # [batch_size, num_heads, head_dim]
        adaptive_weights = self.controller(global_repr.view(batch_size, -1))  # [batch_size, num_heads]
        adaptive_weights = adaptive_weights.unsqueeze(-1).unsqueeze(-1)  # [batch_size, num_heads, 1, 1]
        
        # 计算不同模式的注意力
        attention_outputs = {}
        attention_weights = {}
        
        for mode_name, mode in self.attention_modes.items():
            output, weights = mode(query, key, value, attention_mask)
            attention_outputs[mode_name] = output
            attention_weights[mode_name] = weights
        
        # 自适应融合
        # 这里简化为三种模式的加权平均
        mode_weights = F.softmax(adaptive_weights.squeeze(-1).squeeze(-1), dim=-1)  # [batch_size, num_heads]
        
        final_output = torch.zeros_like(attention_outputs['global'])
        final_weights = torch.zeros_like(attention_weights['global'])
        
        for i, (mode_name, output) in enumerate(attention_outputs.items()):
            weight = mode_weights[:, :, None, None] if i == 0 else mode_weights[:, :, None, None]
            final_output += weight * output
            final_weights += weight * attention_weights[mode_name]
        
        return final_output, final_weights
    
    def output_projection(self, attention_output):
        return self.out_proj(attention_output)

# 辅助类：不同的注意力模式 {#辅助类-不同的注意力模式}
class GlobalAttentionMode(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.scale = (d_model // num_heads) ** -0.5
    
    def forward(self, query, key, value, attention_mask=None):
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        attention_probs = F.softmax(attention_scores, dim=-1)
        context = torch.matmul(attention_probs, value)
        return context, attention_probs

class LocalAttentionMode(nn.Module):
    def __init__(self, d_model, num_heads, window_size=64):
        super().__init__()
        self.scale = (d_model // num_heads) ** -0.5
        self.window_size = window_size
    
    def forward(self, query, key, value, attention_mask=None):
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        
        # 应用局部窗口掩码
        seq_len = attention_scores.size(-1)
        local_mask = torch.zeros_like(attention_scores)
        for i in range(seq_len):
            start = max(0, i - self.window_size // 2)
            end = min(seq_len, i + self.window_size // 2 + 1)
            local_mask[..., i, start:end] = 1
        
        attention_scores = attention_scores.masked_fill(local_mask == 0, float('-inf'))
        
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        
        attention_probs = F.softmax(attention_scores, dim=-1)
        context = torch.matmul(attention_probs, value)
        return context, attention_probs

class SparseAttentionMode(nn.Module):
    def __init__(self, d_model, num_heads, sparsity_ratio=0.1):
        super().__init__()
        self.scale = (d_model // num_heads) ** -0.5
        self.sparsity_ratio = sparsity_ratio
    
    def forward(self, query, key, value, attention_mask=None):
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        
        # 应用稀疏化
        seq_len = attention_scores.size(-1)
        k = int(seq_len * (1 - self.sparsity_ratio))
        topk_values, topk_indices = torch.topk(attention_scores, k, dim=-1)
        sparse_scores = torch.full_like(attention_scores, float('-inf'))
        sparse_scores.scatter_(-1, topk_indices, topk_values)
        
        if attention_mask is not None:
            sparse_scores = sparse_scores + attention_mask
        
        attention_probs = F.softmax(sparse_scores, dim=-1)
        context = torch.matmul(attention_probs, value)
        return context, attention_probs
```

## 注册机制 {#注册机制}

### 🔧 注册自定义注意力 {#注册自定义注意力}

```python
# 方法1: 使用装饰器注册 {#方法1-使用装饰器注册}
@AttentionFactory.register("my_custom_attention")
class MyCustomAttention(BaseAttention):
    # 实现细节...
    pass

# 方法2: 手动注册 {#方法2-手动注册}
class AnotherCustomAttention(BaseAttention):
    # 实现细节...
    pass

AttentionFactory.register_class("another_custom", AnotherCustomAttention)

# 方法3: 批量注册 {#方法3-批量注册}
custom_attentions = {
    "attention_a": AttentionA,
    "attention_b": AttentionB,
    "attention_c": AttentionC
}

for name, cls in custom_attentions.items():
    AttentionFactory.register_class(name, cls)
```

### ⚙️ 配置文件集成 {#配置文件集成}

```yaml
# config.yaml {#config-yaml}
model:
  attention_config:
    # 使用自定义注意力
    attention_types: ["my_custom_attention", "multiscale", "adaptive"]
    
    # 自定义注意力参数
    attention_kwargs:
      my_custom_attention:
        temperature: 1.5
        use_bias: true
        add_noise: false
        noise_std: 0.1
      
      multiscale:
        scales: [1, 2, 4, 8]
      
      adaptive:
        controller_hidden_dim: 256
        num_modes: 3
```

```python
# 在代码中使用 {#在代码中使用}
config = load_config("config.yaml")
model = VIVTransformer(config)

# 自动使用配置中指定的注意力机制 {#自动使用配置中指定的注意力机制}
```

## 测试验证 {#测试验证}

### 🧪 单元测试 {#单元测试}

```python
import unittest
import torch

class TestCustomAttention(unittest.TestCase):
    def setUp(self):
        self.batch_size = 2
        self.seq_len = 128
        self.d_model = 512
        self.num_heads = 8
        
        self.attention = CustomAttention(
            d_model=self.d_model,
            num_heads=self.num_heads,
            dropout=0.1
        )
        
        self.hidden_states = torch.randn(self.batch_size, self.seq_len, self.d_model)
        self.attention_mask = torch.ones(self.batch_size, 1, 1, self.seq_len)
    
    def test_output_shape(self):
        """测试输出形状"""
        output = self.attention(self.hidden_states, self.attention_mask)
        expected_shape = (self.batch_size, self.seq_len, self.d_model)
        self.assertEqual(output.shape, expected_shape)
    
    def test_attention_weights_shape(self):
        """测试注意力权重形状"""
        output, weights = self.attention(
            self.hidden_states, 
            self.attention_mask, 
            return_attention_weights=True
        )
        expected_weights_shape = (self.batch_size, self.num_heads, self.seq_len, self.seq_len)
        self.assertEqual(weights.shape, expected_weights_shape)
    
    def test_attention_weights_sum(self):
        """测试注意力权重和为1"""
        _, weights = self.attention(
            self.hidden_states, 
            self.attention_mask, 
            return_attention_weights=True
        )
        weights_sum = weights.sum(dim=-1)
        expected_sum = torch.ones_like(weights_sum)
        torch.testing.assert_close(weights_sum, expected_sum, atol=1e-6, rtol=1e-6)
    
    def test_gradient_flow(self):
        """测试梯度流"""
        self.hidden_states.requires_grad_(True)
        output = self.attention(self.hidden_states, self.attention_mask)
        loss = output.sum()
        loss.backward()
        
        # 检查梯度是否存在
        self.assertIsNotNone(self.hidden_states.grad)
        self.assertFalse(torch.isnan(self.hidden_states.grad).any())
    
    def test_mask_effectiveness(self):
        """测试掩码有效性"""
        # 创建部分掩码
        masked_attention_mask = self.attention_mask.clone()
        masked_attention_mask[:, :, :, self.seq_len//2:] = 0  # 掩盖后半部分
        
        _, weights = self.attention(
            self.hidden_states, 
            masked_attention_mask, 
            return_attention_weights=True
        )
        
        # 检查被掩盖部分的注意力权重是否接近0
        masked_weights = weights[:, :, :, self.seq_len//2:]
        self.assertTrue((masked_weights < 1e-6).all())
    
    def test_parameter_count(self):
        """测试参数数量"""
        info = self.attention.get_attention_info()
        self.assertGreater(info['parameters'], 0)
        self.assertGreater(info['trainable_parameters'], 0)
    
    def test_different_sequence_lengths(self):
        """测试不同序列长度"""
        for seq_len in [32, 64, 256, 512]:
            hidden_states = torch.randn(self.batch_size, seq_len, self.d_model)
            attention_mask = torch.ones(self.batch_size, 1, 1, seq_len)
            
            output = self.attention(hidden_states, attention_mask)
            expected_shape = (self.batch_size, seq_len, self.d_model)
            self.assertEqual(output.shape, expected_shape)

if __name__ == '__main__':
    unittest.main()
```

### 📊 性能测试 {#性能测试}

```python
class AttentionBenchmark:
    """注意力机制性能测试"""
    
    def __init__(self, attention_types, test_configs):
        self.attention_types = attention_types
        self.test_configs = test_configs
        self.results = {}
    
    def benchmark_memory(self, attention_type, config):
        """内存使用测试"""
        attention = AttentionFactory.create(attention_type, **config)
        
        batch_size, seq_len, d_model = config['batch_size'], config['seq_len'], config['d_model']
        hidden_states = torch.randn(batch_size, seq_len, d_model, device='cuda')
        
        torch.cuda.reset_peak_memory_stats()
        
        with torch.no_grad():
            _ = attention(hidden_states)
        
        peak_memory = torch.cuda.max_memory_allocated() / 1024**2  # MB
        return peak_memory
    
    def benchmark_speed(self, attention_type, config, num_runs=100):
        """速度测试"""
        attention = AttentionFactory.create(attention_type, **config)
        attention = attention.cuda()
        
        batch_size, seq_len, d_model = config['batch_size'], config['seq_len'], config['d_model']
        hidden_states = torch.randn(batch_size, seq_len, d_model, device='cuda')
        
        # 预热
        for _ in range(10):
            with torch.no_grad():
                _ = attention(hidden_states)
        
        torch.cuda.synchronize()
        start_time = time.time()
        
        for _ in range(num_runs):
            with torch.no_grad():
                _ = attention(hidden_states)
        
        torch.cuda.synchronize()
        end_time = time.time()
        
        avg_time = (end_time - start_time) / num_runs
        throughput = batch_size / avg_time
        
        return avg_time, throughput
    
    def run_benchmark(self):
        """运行完整基准测试"""
        for attention_type in self.attention_types:
            self.results[attention_type] = {}
            
            for config_name, config in self.test_configs.items():
                print(f"Testing {attention_type} with {config_name}...")
                
                try:
                    # 内存测试
                    memory_usage = self.benchmark_memory(attention_type, config)
                    
                    # 速度测试
                    avg_time, throughput = self.benchmark_speed(attention_type, config)
                    
                    self.results[attention_type][config_name] = {
                        'memory_mb': memory_usage,
                        'avg_time_ms': avg_time * 1000,
                        'throughput': throughput
                    }
                    
                except Exception as e:
                    print(f"Error testing {attention_type} with {config_name}: {e}")
                    self.results[attention_type][config_name] = {'error': str(e)}
        
        return self.results
    
    def print_results(self):
        """打印测试结果"""
        for attention_type, configs in self.results.items():
            print(f"\n{attention_type}:")
            for config_name, metrics in configs.items():
                if 'error' in metrics:
                    print(f"  {config_name}: ERROR - {metrics['error']}")
                else:
                    print(f"  {config_name}:")
                    print(f"    Memory: {metrics['memory_mb']:.2f} MB")
                    print(f"    Time: {metrics['avg_time_ms']:.2f} ms")
                    print(f"    Throughput: {metrics['throughput']:.2f} samples/sec")

# 使用示例 {#使用示例}
test_configs = {
    'small': {'d_model': 256, 'num_heads': 8, 'batch_size': 4, 'seq_len': 128},
    'medium': {'d_model': 512, 'num_heads': 8, 'batch_size': 4, 'seq_len': 256},
    'large': {'d_model': 768, 'num_heads': 12, 'batch_size': 2, 'seq_len': 512}
}

benchmark = AttentionBenchmark(
    attention_types=['standard', 'linear', 'sparse', 'my_custom_attention'],
    test_configs=test_configs
)

results = benchmark.run_benchmark()
benchmark.print_results()
```

## 性能优化 {#性能优化}

### ⚡ 计算优化 {#计算优化}

```python
class OptimizedAttention(BaseAttention):
    """优化的注意力实现"""
    
    def _init_parameters(self, use_flash_attention=True, **kwargs):
        # 标准参数初始化
        self.q_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.k_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.v_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.out_proj = nn.Linear(self.d_model, self.d_model)
        
        # 优化选项
        self.use_flash_attention = use_flash_attention
        self.chunk_size = kwargs.get('chunk_size', 1024)
        
        # 尝试导入Flash Attention
        if self.use_flash_attention:
            try:
                from flash_attn import flash_attn_func
                self.flash_attn_func = flash_attn_func
            except ImportError:
                print("Flash Attention not available, falling back to standard implementation")
                self.use_flash_attention = False
        
        self.scale = self.head_dim ** -0.5
    
    def compute_attention(self, query, key, value, attention_mask=None, **kwargs):
        if self.use_flash_attention and attention_mask is None:
            # 使用Flash Attention（更高效）
            return self._flash_attention(query, key, value)
        elif query.size(-2) > self.chunk_size:
            # 使用分块计算（节省内存）
            return self._chunked_attention(query, key, value, attention_mask)
        else:
            # 标准计算
            return self._standard_attention(query, key, value, attention_mask)
    
    def _flash_attention(self, query, key, value):
        """Flash Attention实现"""
        # 重排维度以匹配Flash Attention API
        q = query.transpose(1, 2)  # [B, L, H, D]
        k = key.transpose(1, 2)
        v = value.transpose(1, 2)
        
        # 调用Flash Attention
        output = self.flash_attn_func(q, k, v, dropout_p=self.dropout if self.training else 0.0)
        
        # 重排回原始格式
        output = output.transpose(1, 2)  # [B, H, L, D]
        
        return output, None  # Flash Attention不返回权重
    
    def _chunked_attention(self, query, key, value, attention_mask=None):
        """分块注意力计算"""
        batch_size, num_heads, seq_len, head_dim = query.shape
        chunk_size = self.chunk_size
        
        outputs = []
        attention_weights = []
        
        for i in range(0, seq_len, chunk_size):
            end_i = min(i + chunk_size, seq_len)
            q_chunk = query[:, :, i:end_i, :]
            
            # 计算当前块的注意力
            chunk_scores = torch.matmul(q_chunk, key.transpose(-2, -1)) * self.scale
            
            if attention_mask is not None:
                chunk_mask = attention_mask[:, :, i:end_i, :]
                chunk_scores = chunk_scores + chunk_mask
            
            chunk_probs = F.softmax(chunk_scores, dim=-1)
            chunk_probs = self.dropout_layer(chunk_probs)
            
            chunk_output = torch.matmul(chunk_probs, value)
            
            outputs.append(chunk_output)
            attention_weights.append(chunk_probs)
        
        # 拼接所有块
        output = torch.cat(outputs, dim=2)
        weights = torch.cat(attention_weights, dim=2)
        
        return output, weights
    
    def _standard_attention(self, query, key, value, attention_mask=None):
        """标准注意力计算"""
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout_layer(attention_probs)
        
        context = torch.matmul(attention_probs, value)
        
        return context, attention_probs
```

### 🧠 内存优化 {#内存优化}

```python
class MemoryEfficientAttention(BaseAttention):
    """内存高效的注意力实现"""
    
    def _init_parameters(self, **kwargs):
        # 使用更少的参数
        self.qkv_proj = nn.Linear(self.d_model, self.d_model * 3, bias=False)
        self.out_proj = nn.Linear(self.d_model, self.d_model)
        
        # 梯度检查点
        self.use_checkpoint = kwargs.get('use_checkpoint', True)
        
        self.scale = self.head_dim ** -0.5
    
    def generate_qkv(self, hidden_states, position_ids=None, **kwargs):
        # 一次性生成Q、K、V
        qkv = self.qkv_proj(hidden_states)
        query, key, value = qkv.chunk(3, dim=-1)
        return query, key, value
    
    def compute_attention(self, query, key, value, attention_mask=None, **kwargs):
        if self.use_checkpoint and self.training:
            # 使用梯度检查点
            return checkpoint(self._attention_forward, query, key, value, attention_mask)
        else:
            return self._attention_forward(query, key, value, attention_mask)
    
    def _attention_forward(self, query, key, value, attention_mask=None):
        """注意力前向传播"""
        # 使用更节省内存的实现
        batch_size, num_heads, seq_len, head_dim = query.shape
        
        # 分块计算以节省内存
        chunk_size = min(512, seq_len)
        outputs = []
        
        for i in range(0, seq_len, chunk_size):
            end_i = min(i + chunk_size, seq_len)
            q_chunk = query[:, :, i:end_i, :]
            
            # 计算注意力分数
            scores = torch.matmul(q_chunk, key.transpose(-2, -1)) * self.scale
            
            if attention_mask is not None:
                mask_chunk = attention_mask[:, :, i:end_i, :]
                scores = scores + mask_chunk
            
            # 使用数值稳定的softmax
            max_scores = scores.max(dim=-1, keepdim=True)[0]
            scores = scores - max_scores
            exp_scores = torch.exp(scores)
            sum_exp_scores = exp_scores.sum(dim=-1, keepdim=True)
            probs = exp_scores / sum_exp_scores
            
            # 应用dropout
            probs = self.dropout_layer(probs)
            
            # 计算输出
            chunk_output = torch.matmul(probs, value)
            outputs.append(chunk_output)
        
        output = torch.cat(outputs, dim=2)
        return output, None  # 不返回注意力权重以节省内存
    
    def output_projection(self, attention_output):
        return self.out_proj(attention_output)
```

## 最佳实践 {#最佳实践}

### ✅ 设计建议 {#设计建议}

1. **接口一致性**: 始终遵循BaseAttention接口
2. **参数验证**: 在初始化时验证参数的有效性
3. **错误处理**: 优雅地处理异常情况
4. **文档完整**: 提供详细的文档字符串
5. **测试覆盖**: 编写全面的单元测试

### 🚀 性能建议 {#性能建议}

1. **内存优化**: 使用梯度检查点和分块计算
2. **计算优化**: 利用Flash Attention等高效实现
3. **数值稳定**: 使用数值稳定的算法
4. **缓存友好**: 优化内存访问模式
5. **并行化**: 充分利用GPU并行计算能力

### 🔧 调试技巧 {#调试技巧}

```python
class DebuggableAttention(BaseAttention):
    """可调试的注意力实现"""
    
    def __init__(self, *args, debug=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.debug = debug
        self.debug_info = {}
    
    def compute_attention(self, query, key, value, attention_mask=None, **kwargs):
        if self.debug:
            # 记录调试信息
            self.debug_info['input_shapes'] = {
                'query': query.shape,
                'key': key.shape,
                'value': value.shape
            }
            
            # 检查数值稳定性
            self._check_numerical_stability(query, key, value)
        
        # 正常计算
        output, weights = self._standard_attention(query, key, value, attention_mask)
        
        if self.debug:
            self.debug_info['output_shape'] = output.shape
            if weights is not None:
                self.debug_info['attention_stats'] = {
                    'min': weights.min().item(),
                    'max': weights.max().item(),
                    'mean': weights.mean().item(),
                    'std': weights.std().item()
                }
        
        return output, weights
    
    def _check_numerical_stability(self, query, key, value):
        """检查数值稳定性"""
        for name, tensor in [('query', query), ('key', key), ('value', value)]:
            if torch.isnan(tensor).any():
                print(f"Warning: NaN detected in {name}")
            if torch.isinf(tensor).any():
                print(f"Warning: Inf detected in {name}")
            if tensor.abs().max() > 1e6:
                print(f"Warning: Large values detected in {name}: max={tensor.abs().max()}")
    
    def get_debug_info(self):
        """获取调试信息"""
        return self.debug_info
```

---

## 📚 总结 {#总结}

通过本文档，您已经学会了如何在VIVTransformer框架中创建自定义注意力机制。关键要点包括：

### 🎯 核心要素 {#核心要素}

1. **继承BaseAttention**: 确保接口一致性
2. **实现必要方法**: `_init_parameters`, `generate_qkv`, `compute_attention`, `output_projection`
3. **注册机制**: 使用AttentionFactory进行注册
4. **配置集成**: 通过配置文件控制参数
5. **测试验证**: 编写全面的测试用例

### 🚀 优化策略 {#优化策略}

- **计算优化**: Flash Attention、分块计算
- **内存优化**: 梯度检查点、参数共享
- **数值稳定**: 稳定的softmax、梯度裁剪
- **调试支持**: 详细的调试信息和检查

### 📈 扩展方向 {#扩展方向}

- **新的注意力模式**: 局部、稀疏、自适应
- **多模态注意力**: 跨模态交互
- **动态注意力**: 根据输入调整行为
- **高效实现**: 利用最新的硬件特性

遵循这些指导原则，您可以创建高效、可靠、易于维护的自定义注意力机制，为VIVTransformer项目贡献新的功能。

---

*需要帮助？查看 [FAQ](faq) 或 [故障排除](troubleshooting) 页面。*
