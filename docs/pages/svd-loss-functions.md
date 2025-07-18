---
layout: default
title: SVD Loss Functions
parent: Training & Optimization
nav_order: 3
description: "SVD损失函数的理论和实现"
permalink: /pages/svd-loss-functions/
---

# SVD损失函数 {#svd损失函数}

本文档详细介绍VIVTransformer项目中SVD（奇异值分解）损失函数的设计原理、数学基础、实现细节和应用策略。

## 📋 目录 {#目录}

- [理论基础](#理论基础)
- [数学原理](#数学原理)
- [实现架构](#实现架构)
- [损失函数变体](#损失函数变体)
- [自适应权重](#自适应权重)
- [性能优化](#性能优化)
- [实验分析](#实验分析)
- [使用指南](#使用指南)

## 理论基础 {#理论基础}

### 🎯 设计动机 {#设计动机}

SVD损失函数的引入基于以下观察和理论：

1. **低秩结构**: 神经网络权重矩阵通常具有低秩或近似低秩的结构
2. **正则化效果**: SVD正则化可以防止过拟合，提高模型泛化能力
3. **计算效率**: 低秩分解可以减少计算复杂度和存储需求
4. **特征学习**: SVD有助于学习更有意义的特征表示

### 📊 SVD基础 {#svd基础}

对于任意矩阵 $W \in \mathbb{R}^{m \times n}$，SVD分解为：

$$W = U\Sigma V^T$$

其中：
- $U \in \mathbb{R}^{m \times m}$ 是左奇异向量矩阵
- $\Sigma \in \mathbb{R}^{m \times n}$ 是奇异值对角矩阵
- $V \in \mathbb{R}^{n \times n}$ 是右奇异向量矩阵

### 🔍 正则化原理 {#正则化原理}

SVD正则化通过以下方式工作：

1. **核范数正则化**: $\|W\|_* = \sum_{i} \sigma_i$
2. **秩约束**: 限制有效奇异值的数量
3. **谱正则化**: 控制最大奇异值
4. **低秩近似**: 保留主要的奇异值

## 数学原理 {#数学原理}

### 📐 基础SVD损失 {#基础svd损失}

最基本的SVD损失函数定义为：

$$\mathcal{L}_{SVD} = \lambda \sum_{l} \|W_l\|_*$$

其中：
- $\lambda$ 是正则化权重
- $W_l$ 是第$l$层的权重矩阵
- $\|\cdot\|_*$ 是核范数（奇异值之和）

### 🎯 加权SVD损失 {#加权svd损失}

考虑不同层的重要性，引入加权版本：

$$\mathcal{L}_{SVD} = \sum_{l} \lambda_l \|W_l\|_*$$

其中 $\lambda_l$ 是第$l$层的特定权重。

### 🔄 自适应SVD损失 {#自适应svd损失}

动态调整正则化强度：

$$\mathcal{L}_{SVD} = \sum_{l} \lambda_l(t) \cdot f(\sigma_l) \cdot \|W_l\|_*$$

其中：
- $\lambda_l(t)$ 是时间相关的权重
- $f(\sigma_l)$ 是基于奇异值的自适应函数

### 📊 多项式SVD损失 {#多项式svd损失}

使用奇异值的不同幂次：

$$\mathcal{L}_{SVD} = \sum_{l} \lambda_l \sum_{i} \sigma_{l,i}^p$$

其中 $p$ 控制正则化的强度（$p=1$ 为核范数，$p=2$ 为Frobenius范数）。

## 实现架构 {#实现架构}

### 🏗️ 核心实现 {#核心实现}

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Union, Callable
import numpy as np
from abc import ABC, abstractmethod

class BaseSVDLoss(nn.Module, ABC):
    """SVD损失函数基类
    
    提供SVD损失函数的通用接口和基础功能。
    """
    
    def __init__(self, 
                 weight: float = 0.01,
                 target_layers: Optional[List[str]] = None,
                 layer_weights: Optional[Dict[str, float]] = None):
        super(BaseSVDLoss, self).__init__()
        self.weight = weight
        self.target_layers = target_layers or ['attention', 'feed_forward']
        self.layer_weights = layer_weights or {}
        
        # 统计信息
        self.stats = {
            'total_calls': 0,
            'avg_svd_loss': 0.0,
            'layer_contributions': {}
        }
    
    @abstractmethod
    def compute_svd_regularization(self, 
                                  weight_matrix: torch.Tensor,
                                  layer_name: str = None) -> torch.Tensor:
        """计算单个权重矩阵的SVD正则化项
        
        Args:
            weight_matrix: 权重矩阵 [out_features, in_features]
            layer_name: 层名称（用于自适应权重）
        
        Returns:
            SVD正则化损失值
        """
        pass
    
    def forward(self, model: nn.Module) -> Dict[str, torch.Tensor]:
        """计算模型的总SVD损失
        
        Args:
            model: 目标模型
        
        Returns:
            包含损失信息的字典
        """
        total_svd_loss = 0.0
        layer_losses = {}
        processed_layers = 0
        
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and self._should_regularize(name):
                try:
                    # 计算SVD正则化
                    svd_loss = self.compute_svd_regularization(
                        module.weight, layer_name=name
                    )
                    
                    # 应用层特定权重
                    layer_weight = self._get_layer_weight(name)
                    weighted_loss = layer_weight * svd_loss
                    
                    total_svd_loss += weighted_loss
                    layer_losses[name] = weighted_loss.item()
                    processed_layers += 1
                    
                except RuntimeError as e:
                    # SVD可能失败，记录但继续
                    print(f"SVD failed for layer {name}: {e}")
                    continue
        
        # 平均化
        if processed_layers > 0:
            total_svd_loss = total_svd_loss / processed_layers
        
        # 应用全局权重
        final_loss = self.weight * total_svd_loss
        
        # 更新统计信息
        self._update_stats(final_loss.item(), layer_losses)
        
        return {
            'svd_loss': final_loss,
            'layer_losses': layer_losses,
            'processed_layers': processed_layers
        }
    
    def _should_regularize(self, layer_name: str) -> bool:
        """判断是否应该对该层进行正则化"""
        return any(target in layer_name for target in self.target_layers)
    
    def _get_layer_weight(self, layer_name: str) -> float:
        """获取层特定权重"""
        for pattern, weight in self.layer_weights.items():
            if pattern in layer_name:
                return weight
        return 1.0
    
    def _update_stats(self, total_loss: float, layer_losses: Dict[str, float]):
        """更新统计信息"""
        self.stats['total_calls'] += 1
        
        # 更新平均损失
        alpha = 0.1  # 指数移动平均
        self.stats['avg_svd_loss'] = (
            alpha * total_loss + 
            (1 - alpha) * self.stats['avg_svd_loss']
        )
        
        # 更新层贡献
        for layer_name, loss in layer_losses.items():
            if layer_name not in self.stats['layer_contributions']:
                self.stats['layer_contributions'][layer_name] = 0.0
            self.stats['layer_contributions'][layer_name] = (
                alpha * loss + 
                (1 - alpha) * self.stats['layer_contributions'][layer_name]
            )
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        return self.stats.copy()
    
    def reset_stats(self):
        """重置统计信息"""
        self.stats = {
            'total_calls': 0,
            'avg_svd_loss': 0.0,
            'layer_contributions': {}
        }
```

### 🎯 核范数SVD损失 {#核范数svd损失}

```python
class NuclearNormSVDLoss(BaseSVDLoss):
    """核范数SVD损失函数
    
    使用奇异值之和作为正则化项：||W||_* = Σσᵢ
    """
    
    def __init__(self, **kwargs):
        super(NuclearNormSVDLoss, self).__init__(**kwargs)
        self.eps = 1e-8  # 数值稳定性
    
    def compute_svd_regularization(self, 
                                  weight_matrix: torch.Tensor,
                                  layer_name: str = None) -> torch.Tensor:
        """计算核范数正则化"""
        try:
            # 计算SVD
            U, S, V = torch.svd(weight_matrix)
            
            # 核范数 = 奇异值之和
            nuclear_norm = torch.sum(S)
            
            return nuclear_norm
            
        except RuntimeError:
            # SVD失败时的备选方案
            return torch.norm(weight_matrix, p='fro')

class TruncatedSVDLoss(BaseSVDLoss):
    """截断SVD损失函数
    
    只考虑前k个最大的奇异值。
    """
    
    def __init__(self, rank_ratio: float = 0.5, **kwargs):
        super(TruncatedSVDLoss, self).__init__(**kwargs)
        self.rank_ratio = rank_ratio  # 保留的奇异值比例
    
    def compute_svd_regularization(self, 
                                  weight_matrix: torch.Tensor,
                                  layer_name: str = None) -> torch.Tensor:
        """计算截断SVD正则化"""
        try:
            U, S, V = torch.svd(weight_matrix)
            
            # 计算截断秩
            total_rank = S.size(0)
            truncated_rank = max(1, int(total_rank * self.rank_ratio))
            
            # 只使用前k个奇异值
            truncated_S = S[:truncated_rank]
            truncated_norm = torch.sum(truncated_S)
            
            return truncated_norm
            
        except RuntimeError:
            return torch.norm(weight_matrix, p='fro')

class WeightedSVDLoss(BaseSVDLoss):
    """加权SVD损失函数
    
    对不同的奇异值应用不同的权重。
    """
    
    def __init__(self, 
                 weighting_scheme: str = 'exponential',
                 decay_factor: float = 0.9,
                 **kwargs):
        super(WeightedSVDLoss, self).__init__(**kwargs)
        self.weighting_scheme = weighting_scheme
        self.decay_factor = decay_factor
    
    def compute_svd_regularization(self, 
                                  weight_matrix: torch.Tensor,
                                  layer_name: str = None) -> torch.Tensor:
        """计算加权SVD正则化"""
        try:
            U, S, V = torch.svd(weight_matrix)
            
            # 生成权重
            weights = self._generate_weights(S.size(0), S.device)
            
            # 加权奇异值和
            weighted_sum = torch.sum(weights * S)
            
            return weighted_sum
            
        except RuntimeError:
            return torch.norm(weight_matrix, p='fro')
    
    def _generate_weights(self, num_singular_values: int, device: torch.device) -> torch.Tensor:
        """生成奇异值权重"""
        if self.weighting_scheme == 'exponential':
            # 指数衰减权重
            indices = torch.arange(num_singular_values, device=device, dtype=torch.float)
            weights = self.decay_factor ** indices
        elif self.weighting_scheme == 'linear':
            # 线性衰减权重
            weights = torch.linspace(1.0, 0.1, num_singular_values, device=device)
        elif self.weighting_scheme == 'inverse':
            # 逆序权重（小奇异值权重更大）
            indices = torch.arange(num_singular_values, device=device, dtype=torch.float)
            weights = 1.0 / (indices + 1.0)
        else:
            # 均匀权重
            weights = torch.ones(num_singular_values, device=device)
        
        return weights
```

## 损失函数变体 {#损失函数变体}

### 🔄 自适应SVD损失 {#自适应svd损失}

```python
class AdaptiveSVDLoss(BaseSVDLoss):
    """自适应SVD损失函数
    
    根据训练进度和模型状态动态调整正则化强度。
    """
    
    def __init__(self, 
                 initial_weight: float = 0.01,
                 min_weight: float = 0.001,
                 max_weight: float = 0.1,
                 adaptation_strategy: str = 'loss_based',
                 **kwargs):
        super(AdaptiveSVDLoss, self).__init__(weight=initial_weight, **kwargs)
        self.initial_weight = initial_weight
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.adaptation_strategy = adaptation_strategy
        
        # 自适应状态
        self.current_epoch = 0
        self.loss_history = []
        self.rank_history = []
    
    def update_epoch(self, epoch: int, validation_loss: float = None):
        """更新训练轮次和相关信息"""
        self.current_epoch = epoch
        if validation_loss is not None:
            self.loss_history.append(validation_loss)
    
    def compute_svd_regularization(self, 
                                  weight_matrix: torch.Tensor,
                                  layer_name: str = None) -> torch.Tensor:
        """计算自适应SVD正则化"""
        try:
            U, S, V = torch.svd(weight_matrix)
            
            # 计算有效秩
            effective_rank = self._compute_effective_rank(S)
            self.rank_history.append(effective_rank)
            
            # 自适应调整权重
            adaptive_weight = self._compute_adaptive_weight(S, effective_rank)
            
            # 计算正则化项
            nuclear_norm = torch.sum(S)
            
            return adaptive_weight * nuclear_norm
            
        except RuntimeError:
            return self.weight * torch.norm(weight_matrix, p='fro')
    
    def _compute_effective_rank(self, singular_values: torch.Tensor) -> float:
        """计算有效秩"""
        # 使用Shannon熵定义的有效秩
        normalized_s = singular_values / torch.sum(singular_values)
        entropy = -torch.sum(normalized_s * torch.log(normalized_s + 1e-8))
        effective_rank = torch.exp(entropy).item()
        return effective_rank
    
    def _compute_adaptive_weight(self, 
                               singular_values: torch.Tensor, 
                               effective_rank: float) -> float:
        """计算自适应权重"""
        if self.adaptation_strategy == 'rank_based':
            # 基于秩的自适应
            max_rank = singular_values.size(0)
            rank_ratio = effective_rank / max_rank
            # 秩越高，正则化越强
            weight = self.initial_weight * (1 + rank_ratio)
            
        elif self.adaptation_strategy == 'loss_based':
            # 基于损失的自适应
            if len(self.loss_history) >= 2:
                loss_trend = self.loss_history[-1] - self.loss_history[-2]
                if loss_trend > 0:  # 损失增加，增强正则化
                    weight = min(self.max_weight, self.weight * 1.1)
                else:  # 损失减少，减弱正则化
                    weight = max(self.min_weight, self.weight * 0.95)
            else:
                weight = self.initial_weight
                
        elif self.adaptation_strategy == 'epoch_based':
            # 基于训练轮次的自适应
            decay_factor = 0.95 ** (self.current_epoch // 10)
            weight = self.initial_weight * decay_factor
            
        else:
            weight = self.initial_weight
        
        # 更新当前权重
        self.weight = max(self.min_weight, min(self.max_weight, weight))
        return self.weight

class SpectralSVDLoss(BaseSVDLoss):
    """谱SVD损失函数
    
    专注于控制最大奇异值（谱范数）。
    """
    
    def __init__(self, spectral_weight: float = 1.0, **kwargs):
        super(SpectralSVDLoss, self).__init__(**kwargs)
        self.spectral_weight = spectral_weight
    
    def compute_svd_regularization(self, 
                                  weight_matrix: torch.Tensor,
                                  layer_name: str = None) -> torch.Tensor:
        """计算谱正则化"""
        try:
            U, S, V = torch.svd(weight_matrix)
            
            # 谱范数 = 最大奇异值
            spectral_norm = S[0]
            
            # 核范数
            nuclear_norm = torch.sum(S)
            
            # 组合损失
            total_loss = nuclear_norm + self.spectral_weight * spectral_norm
            
            return total_loss
            
        except RuntimeError:
            return torch.norm(weight_matrix, p='fro')

class LowRankSVDLoss(BaseSVDLoss):
    """低秩SVD损失函数
    
    鼓励权重矩阵具有低秩结构。
    """
    
    def __init__(self, 
                 target_rank: Optional[int] = None,
                 rank_penalty: float = 1.0,
                 **kwargs):
        super(LowRankSVDLoss, self).__init__(**kwargs)
        self.target_rank = target_rank
        self.rank_penalty = rank_penalty
    
    def compute_svd_regularization(self, 
                                  weight_matrix: torch.Tensor,
                                  layer_name: str = None) -> torch.Tensor:
        """计算低秩正则化"""
        try:
            U, S, V = torch.svd(weight_matrix)
            
            # 如果没有指定目标秩，使用矩阵维度的一半
            if self.target_rank is None:
                target_rank = min(weight_matrix.shape) // 2
            else:
                target_rank = self.target_rank
            
            # 计算超出目标秩的奇异值惩罚
            if S.size(0) > target_rank:
                excess_singular_values = S[target_rank:]
                rank_penalty_loss = self.rank_penalty * torch.sum(excess_singular_values)
            else:
                rank_penalty_loss = torch.tensor(0.0, device=S.device)
            
            # 基础核范数
            nuclear_norm = torch.sum(S[:target_rank])
            
            total_loss = nuclear_norm + rank_penalty_loss
            
            return total_loss
            
        except RuntimeError:
            return torch.norm(weight_matrix, p='fro')
```

## 自适应权重 {#自适应权重}

### 🎛️ 动态权重调整 {#动态权重调整}

```python
class DynamicSVDWeightScheduler:
    """动态SVD权重调度器
    
    根据训练状态动态调整SVD损失的权重。
    """
    
    def __init__(self, 
                 initial_weight: float = 0.01,
                 schedule_type: str = 'cosine',
                 total_epochs: int = 100,
                 warmup_epochs: int = 10,
                 min_weight: float = 0.001,
                 max_weight: float = 0.1):
        self.initial_weight = initial_weight
        self.schedule_type = schedule_type
        self.total_epochs = total_epochs
        self.warmup_epochs = warmup_epochs
        self.min_weight = min_weight
        self.max_weight = max_weight
        
        self.current_epoch = 0
        self.current_weight = initial_weight
    
    def step(self, epoch: int, metrics: Dict[str, float] = None) -> float:
        """更新权重
        
        Args:
            epoch: 当前训练轮次
            metrics: 训练指标（如验证损失、准确率等）
        
        Returns:
            更新后的SVD权重
        """
        self.current_epoch = epoch
        
        if self.schedule_type == 'cosine':
            self.current_weight = self._cosine_schedule(epoch)
        elif self.schedule_type == 'linear':
            self.current_weight = self._linear_schedule(epoch)
        elif self.schedule_type == 'exponential':
            self.current_weight = self._exponential_schedule(epoch)
        elif self.schedule_type == 'adaptive':
            self.current_weight = self._adaptive_schedule(epoch, metrics)
        else:
            self.current_weight = self.initial_weight
        
        return self.current_weight
    
    def _cosine_schedule(self, epoch: int) -> float:
        """余弦调度"""
        if epoch < self.warmup_epochs:
            # 预热阶段
            return self.initial_weight * (epoch / self.warmup_epochs)
        else:
            # 余弦衰减
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            cosine_factor = 0.5 * (1 + np.cos(np.pi * progress))
            return self.min_weight + (self.initial_weight - self.min_weight) * cosine_factor
    
    def _linear_schedule(self, epoch: int) -> float:
        """线性调度"""
        if epoch < self.warmup_epochs:
            return self.initial_weight * (epoch / self.warmup_epochs)
        else:
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            return self.initial_weight * (1 - progress) + self.min_weight * progress
    
    def _exponential_schedule(self, epoch: int) -> float:
        """指数调度"""
        decay_rate = 0.95
        return max(self.min_weight, self.initial_weight * (decay_rate ** epoch))
    
    def _adaptive_schedule(self, epoch: int, metrics: Dict[str, float]) -> float:
        """自适应调度"""
        if metrics is None:
            return self.current_weight
        
        # 基于验证损失的自适应调整
        val_loss = metrics.get('val_loss', None)
        if val_loss is not None:
            # 如果验证损失停止改善，增加正则化
            if hasattr(self, 'best_val_loss'):
                if val_loss > self.best_val_loss:
                    self.current_weight = min(self.max_weight, self.current_weight * 1.05)
                else:
                    self.current_weight = max(self.min_weight, self.current_weight * 0.98)
                    self.best_val_loss = val_loss
            else:
                self.best_val_loss = val_loss
        
        return self.current_weight
    
    def get_current_weight(self) -> float:
        """获取当前权重"""
        return self.current_weight

class LayerWiseSVDWeightManager:
    """分层SVD权重管理器
    
    为不同层分配不同的SVD权重。
    """
    
    def __init__(self, 
                 base_weight: float = 0.01,
                 layer_config: Dict[str, Dict] = None):
        self.base_weight = base_weight
        self.layer_config = layer_config or {}
        
        # 默认配置
        self.default_config = {
            'attention': {'weight_multiplier': 1.0, 'decay_factor': 0.95},
            'feed_forward': {'weight_multiplier': 0.8, 'decay_factor': 0.9},
            'embedding': {'weight_multiplier': 1.2, 'decay_factor': 0.98},
            'output': {'weight_multiplier': 1.5, 'decay_factor': 0.92}
        }
        
        # 合并配置
        for layer_type, config in self.default_config.items():
            if layer_type not in self.layer_config:
                self.layer_config[layer_type] = config
    
    def get_layer_weight(self, layer_name: str, epoch: int = 0) -> float:
        """获取特定层的SVD权重
        
        Args:
            layer_name: 层名称
            epoch: 当前训练轮次
        
        Returns:
            该层的SVD权重
        """
        # 确定层类型
        layer_type = self._identify_layer_type(layer_name)
        
        # 获取配置
        config = self.layer_config.get(layer_type, {'weight_multiplier': 1.0, 'decay_factor': 1.0})
        
        # 计算权重
        multiplier = config['weight_multiplier']
        decay_factor = config['decay_factor']
        
        # 应用时间衰减
        time_decay = decay_factor ** (epoch // 10)
        
        final_weight = self.base_weight * multiplier * time_decay
        
        return final_weight
    
    def _identify_layer_type(self, layer_name: str) -> str:
        """识别层类型"""
        layer_name_lower = layer_name.lower()
        
        if 'attention' in layer_name_lower or 'attn' in layer_name_lower:
            return 'attention'
        elif 'feed_forward' in layer_name_lower or 'ffn' in layer_name_lower or 'mlp' in layer_name_lower:
            return 'feed_forward'
        elif 'embedding' in layer_name_lower or 'embed' in layer_name_lower:
            return 'embedding'
        elif 'output' in layer_name_lower or 'classifier' in layer_name_lower:
            return 'output'
        else:
            return 'default'
    
    def update_config(self, layer_type: str, config: Dict):
        """更新层配置"""
        if layer_type in self.layer_config:
            self.layer_config[layer_type].update(config)
        else:
            self.layer_config[layer_type] = config
    
    def get_all_weights(self, model: nn.Module, epoch: int = 0) -> Dict[str, float]:
        """获取模型所有层的权重"""
        weights = {}
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                weights[name] = self.get_layer_weight(name, epoch)
        return weights
```

## 性能优化 {#性能优化}

### ⚡ 高效SVD计算 {#高效svd计算}

```python
class EfficientSVDLoss(BaseSVDLoss):
    """高效SVD损失实现
    
    使用各种优化技术提高SVD计算效率。
    """
    
    def __init__(self, 
                 use_power_iteration: bool = True,
                 power_iterations: int = 5,
                 use_randomized_svd: bool = False,
                 svd_rank: Optional[int] = None,
                 cache_svd: bool = True,
                 **kwargs):
        super(EfficientSVDLoss, self).__init__(**kwargs)
        self.use_power_iteration = use_power_iteration
        self.power_iterations = power_iterations
        self.use_randomized_svd = use_randomized_svd
        self.svd_rank = svd_rank
        self.cache_svd = cache_svd
        
        # SVD缓存
        self.svd_cache = {} if cache_svd else None
        self.cache_hits = 0
        self.cache_misses = 0
    
    def compute_svd_regularization(self, 
                                  weight_matrix: torch.Tensor,
                                  layer_name: str = None) -> torch.Tensor:
        """高效SVD正则化计算"""
        # 检查缓存
        if self.cache_svd and layer_name is not None:
            cache_key = self._get_cache_key(weight_matrix, layer_name)
            if cache_key in self.svd_cache:
                self.cache_hits += 1
                return self.svd_cache[cache_key]
            else:
                self.cache_misses += 1
        
        try:
            if self.use_randomized_svd and weight_matrix.numel() > 10000:
                # 使用随机化SVD（适用于大矩阵）
                singular_values = self._randomized_svd(weight_matrix)
            elif self.use_power_iteration:
                # 使用幂迭代法估计最大奇异值
                singular_values = self._power_iteration_svd(weight_matrix)
            else:
                # 标准SVD
                _, singular_values, _ = torch.svd(weight_matrix)
            
            # 计算正则化项
            nuclear_norm = torch.sum(singular_values)
            
            # 缓存结果
            if self.cache_svd and layer_name is not None:
                self.svd_cache[cache_key] = nuclear_norm
            
            return nuclear_norm
            
        except RuntimeError:
            # 备选方案
            return torch.norm(weight_matrix, p='fro')
    
    def _randomized_svd(self, matrix: torch.Tensor) -> torch.Tensor:
        """随机化SVD实现"""
        m, n = matrix.shape
        rank = self.svd_rank or min(m, n, 50)  # 默认秩为50
        
        # 生成随机矩阵
        omega = torch.randn(n, rank, device=matrix.device, dtype=matrix.dtype)
        
        # 计算Y = A * Omega
        Y = torch.matmul(matrix, omega)
        
        # QR分解
        Q, _ = torch.qr(Y)
        
        # 计算B = Q^T * A
        B = torch.matmul(Q.t(), matrix)
        
        # 对B进行SVD
        _, S, _ = torch.svd(B)
        
        return S
    
    def _power_iteration_svd(self, matrix: torch.Tensor) -> torch.Tensor:
        """幂迭代法估计奇异值"""
        m, n = matrix.shape
        
        # 估计最大奇异值
        v = torch.randn(n, 1, device=matrix.device, dtype=matrix.dtype)
        v = v / torch.norm(v)
        
        for _ in range(self.power_iterations):
            # v = A^T * A * v
            v = torch.matmul(matrix.t(), torch.matmul(matrix, v))
            v = v / torch.norm(v)
        
        # 计算最大奇异值
        Av = torch.matmul(matrix, v)
        max_singular_value = torch.norm(Av)
        
        # 近似核范数（这里简化为最大奇异值的倍数）
        # 更精确的方法需要估计所有奇异值
        estimated_rank = min(m, n)
        estimated_nuclear_norm = max_singular_value * estimated_rank * 0.5
        
        return torch.tensor([estimated_nuclear_norm], device=matrix.device)
    
    def _get_cache_key(self, weight_matrix: torch.Tensor, layer_name: str) -> str:
        """生成缓存键"""
        # 使用权重矩阵的哈希和层名称作为键
        weight_hash = hash(weight_matrix.data_ptr())
        return f"{layer_name}_{weight_hash}_{weight_matrix.shape}"
    
    def clear_cache(self):
        """清理缓存"""
        if self.svd_cache is not None:
            self.svd_cache.clear()
            self.cache_hits = 0
            self.cache_misses = 0
    
    def get_cache_stats(self) -> Dict[str, int]:
        """获取缓存统计"""
        return {
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate': self.cache_hits / max(1, self.cache_hits + self.cache_misses)
        }
```

### 🔧 批量SVD处理 {#批量svd处理}

```python
class BatchSVDLoss(BaseSVDLoss):
    """批量SVD损失处理
    
    同时处理多个权重矩阵以提高效率。
    """
    
    def __init__(self, batch_size: int = 8, **kwargs):
        super(BatchSVDLoss, self).__init__(**kwargs)
        self.batch_size = batch_size
    
    def forward(self, model: nn.Module) -> Dict[str, torch.Tensor]:
        """批量处理SVD损失"""
        # 收集所有需要处理的权重矩阵
        weight_matrices = []
        layer_names = []
        layer_weights = []
        
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and self._should_regularize(name):
                weight_matrices.append(module.weight)
                layer_names.append(name)
                layer_weights.append(self._get_layer_weight(name))
        
        if not weight_matrices:
            return {
                'svd_loss': torch.tensor(0.0),
                'layer_losses': {},
                'processed_layers': 0
            }
        
        # 批量处理
        total_loss = 0.0
        layer_losses = {}
        
        for i in range(0, len(weight_matrices), self.batch_size):
            batch_end = min(i + self.batch_size, len(weight_matrices))
            batch_matrices = weight_matrices[i:batch_end]
            batch_names = layer_names[i:batch_end]
            batch_weights = layer_weights[i:batch_end]
            
            # 批量计算SVD
            batch_losses = self._batch_compute_svd(batch_matrices, batch_names)
            
            # 累积损失
            for j, (name, loss) in enumerate(zip(batch_names, batch_losses)):
                weighted_loss = batch_weights[j] * loss
                total_loss += weighted_loss
                layer_losses[name] = weighted_loss.item()
        
        # 平均化和应用全局权重
        if len(weight_matrices) > 0:
            total_loss = total_loss / len(weight_matrices)
        final_loss = self.weight * total_loss
        
        return {
            'svd_loss': final_loss,
            'layer_losses': layer_losses,
            'processed_layers': len(weight_matrices)
        }
    
    def _batch_compute_svd(self, 
                          weight_matrices: List[torch.Tensor],
                          layer_names: List[str]) -> List[torch.Tensor]:
        """批量计算SVD正则化"""
        losses = []
        
        for weight_matrix, layer_name in zip(weight_matrices, layer_names):
            try:
                loss = self.compute_svd_regularization(weight_matrix, layer_name)
                losses.append(loss)
            except RuntimeError:
                # SVD失败时的备选方案
                losses.append(torch.norm(weight_matrix, p='fro'))
        
        return losses
    
    def compute_svd_regularization(self, 
                                  weight_matrix: torch.Tensor,
                                  layer_name: str = None) -> torch.Tensor:
        """计算SVD正则化（基础实现）"""
        try:
            _, S, _ = torch.svd(weight_matrix)
            return torch.sum(S)
        except RuntimeError:
            return torch.norm(weight_matrix, p='fro')
```

## 实验分析 {#实验分析}

### 📊 SVD损失效果分析 {#svd损失效果分析}

```python
class SVDLossAnalyzer:
    """SVD损失效果分析器"""
    
    def __init__(self, model: nn.Module, svd_loss: BaseSVDLoss):
        self.model = model
        self.svd_loss = svd_loss
        self.analysis_history = []
    
    def analyze_model_ranks(self) -> Dict[str, Dict]:
        """分析模型各层的秩信息"""
        rank_info = {}
        
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                weight_matrix = module.weight.detach()
                
                try:
                    U, S, V = torch.svd(weight_matrix)
                    
                    # 计算各种秩度量
                    full_rank = S.size(0)
                    effective_rank = self._compute_effective_rank(S)
                    stable_rank = self._compute_stable_rank(S)
                    numerical_rank = self._compute_numerical_rank(S)
                    
                    rank_info[name] = {
                        'full_rank': full_rank,
                        'effective_rank': effective_rank,
                        'stable_rank': stable_rank,
                        'numerical_rank': numerical_rank,
                        'singular_values': S.cpu().numpy(),
                        'condition_number': (S[0] / S[-1]).item() if S[-1] > 1e-8 else float('inf')
                    }
                    
                except RuntimeError:
                    rank_info[name] = {'error': 'SVD computation failed'}
        
        return rank_info
    
    def _compute_effective_rank(self, singular_values: torch.Tensor) -> float:
        """计算有效秩（基于Shannon熵）"""
        normalized_s = singular_values / torch.sum(singular_values)
        entropy = -torch.sum(normalized_s * torch.log(normalized_s + 1e-8))
        return torch.exp(entropy).item()
    
    def _compute_stable_rank(self, singular_values: torch.Tensor) -> float:
        """计算稳定秩"""
        frobenius_norm_sq = torch.sum(singular_values ** 2)
        spectral_norm_sq = singular_values[0] ** 2
        return (frobenius_norm_sq / spectral_norm_sq).item()
    
    def _compute_numerical_rank(self, singular_values: torch.Tensor, tol: float = 1e-6) -> int:
        """计算数值秩"""
        return torch.sum(singular_values > tol * singular_values[0]).item()
    
    def track_training_progress(self, epoch: int, metrics: Dict[str, float]):
        """跟踪训练进度"""
        rank_info = self.analyze_model_ranks()
        svd_stats = self.svd_loss.get_stats()
        
        analysis_entry = {
            'epoch': epoch,
            'metrics': metrics.copy(),
            'rank_info': rank_info,
            'svd_stats': svd_stats
        }
        
        self.analysis_history.append(analysis_entry)
    
    def generate_report(self) -> Dict:
        """生成分析报告"""
        if not self.analysis_history:
            return {'error': 'No analysis data available'}
        
        # 提取趋势信息
        epochs = [entry['epoch'] for entry in self.analysis_history]
        
        # 计算平均有效秩变化
        avg_effective_ranks = []
        for entry in self.analysis_history:
            ranks = [info.get('effective_rank', 0) for info in entry['rank_info'].values() 
                    if 'effective_rank' in info]
            avg_effective_ranks.append(np.mean(ranks) if ranks else 0)
        
        # 计算SVD损失变化
        svd_losses = [entry['svd_stats'].get('avg_svd_loss', 0) for entry in self.analysis_history]
        
        # 生成报告
        report = {
            'training_epochs': len(epochs),
            'rank_evolution': {
                'epochs': epochs,
                'avg_effective_rank': avg_effective_ranks,
                'rank_trend': 'decreasing' if avg_effective_ranks[-1] < avg_effective_ranks[0] else 'increasing'
            },
            'svd_loss_evolution': {
                'epochs': epochs,
                'svd_losses': svd_losses,
                'loss_trend': 'decreasing' if svd_losses[-1] < svd_losses[0] else 'increasing'
            },
            'final_rank_analysis': self.analysis_history[-1]['rank_info'],
            'recommendations': self._generate_recommendations()
        }
        
        return report
    
    def _generate_recommendations(self) -> List[str]:
        """生成优化建议"""
        recommendations = []
        
        if not self.analysis_history:
            return recommendations
        
        latest_analysis = self.analysis_history[-1]
        
        # 检查过拟合迹象
        if len(self.analysis_history) >= 2:
            recent_metrics = [entry['metrics'] for entry in self.analysis_history[-5:]]
            if 'val_loss' in recent_metrics[0]:
                val_losses = [m['val_loss'] for m in recent_metrics]
                if len(val_losses) >= 3 and all(val_losses[i] >= val_losses[i-1] for i in range(1, len(val_losses))):
                    recommendations.append("考虑增加SVD正则化权重以防止过拟合")
        
        # 检查秩退化
        rank_info = latest_analysis['rank_info']
        low_rank_layers = []
        for layer_name, info in rank_info.items():
            if 'effective_rank' in info and 'full_rank' in info:
                rank_ratio = info['effective_rank'] / info['full_rank']
                if rank_ratio < 0.1:
                    low_rank_layers.append(layer_name)
        
        if low_rank_layers:
            recommendations.append(f"以下层的有效秩过低，可能需要调整正则化: {', '.join(low_rank_layers)}")
        
        # 检查条件数
        high_condition_layers = []
        for layer_name, info in rank_info.items():
            if 'condition_number' in info and info['condition_number'] > 1000:
                high_condition_layers.append(layer_name)
        
        if high_condition_layers:
            recommendations.append(f"以下层的条件数过高，建议增加正则化: {', '.join(high_condition_layers)}")
        
        return recommendations
    
    def visualize_singular_values(self, layer_name: str = None) -> Dict:
        """可视化奇异值分布"""
        if not self.analysis_history:
            return {'error': 'No analysis data available'}
        
        latest_analysis = self.analysis_history[-1]
        rank_info = latest_analysis['rank_info']
        
        if layer_name is None:
            # 选择第一个可用的层
            layer_name = next(iter(rank_info.keys()))
        
        if layer_name not in rank_info or 'singular_values' not in rank_info[layer_name]:
            return {'error': f'No singular value data for layer {layer_name}'}
        
        singular_values = rank_info[layer_name]['singular_values']
        
        return {
            'layer_name': layer_name,
            'singular_values': singular_values.tolist(),
            'num_singular_values': len(singular_values),
            'max_singular_value': float(singular_values[0]),
            'min_singular_value': float(singular_values[-1]),
            'singular_value_ratio': float(singular_values[0] / singular_values[-1]) if singular_values[-1] > 1e-8 else float('inf')
        }
```

## 使用指南 {#使用指南}

### 🚀 快速开始 {#快速开始}

```python
# 基础使用 {#基础使用}
from vivtransformer.losses import NuclearNormSVDLoss

# 创建SVD损失函数 {#创建svd损失函数}
svd_loss = NuclearNormSVDLoss(
    weight=0.01,
    target_layers=['attention', 'feed_forward']
)

# 在训练循环中使用 {#在训练循环中使用}
for batch in dataloader:
    # 前向传播
    outputs = model(batch['input'])
    
    # 计算主要损失
    main_loss = criterion(outputs, batch['target'])
    
    # 计算SVD损失
    svd_result = svd_loss(model)
    
    # 总损失
    total_loss = main_loss + svd_result['svd_loss']
    
    # 反向传播
    total_loss.backward()
    optimizer.step()
```

### ⚙️ 高级配置 {#高级配置}

```python
# 自适应SVD损失 {#自适应svd损失}
from vivtransformer.losses import AdaptiveSVDLoss
from vivtransformer.losses import DynamicSVDWeightScheduler

# 创建自适应SVD损失 {#创建自适应svd损失}
svd_loss = AdaptiveSVDLoss(
    initial_weight=0.01,
    min_weight=0.001,
    max_weight=0.1,
    adaptation_strategy='loss_based'
)

# 创建权重调度器 {#创建权重调度器}
weight_scheduler = DynamicSVDWeightScheduler(
    initial_weight=0.01,
    schedule_type='cosine',
    total_epochs=100
)

# 训练循环 {#训练循环}
for epoch in range(num_epochs):
    # 更新SVD权重
    current_weight = weight_scheduler.step(epoch, {'val_loss': val_loss})
    svd_loss.weight = current_weight
    
    for batch in dataloader:
        # 训练步骤...
        pass
    
    # 更新自适应参数
    svd_loss.update_epoch(epoch, val_loss)
```

### 📊 配置文件示例 {#配置文件示例}

```yaml
# config.yaml {#config-yaml}
loss:
  svd_loss:
    type: "adaptive"  # nuclear_norm, truncated, weighted, adaptive, spectral, low_rank
    weight: 0.01
    target_layers: ["attention", "feed_forward"]
    
    # 自适应参数
    adaptation_strategy: "loss_based"  # rank_based, loss_based, epoch_based
    min_weight: 0.001
    max_weight: 0.1
    
    # 层特定权重
    layer_weights:
      attention: 1.0
      feed_forward: 0.8
      embedding: 1.2
      output: 1.5
    
    # 性能优化
    use_power_iteration: true
    power_iterations: 5
    cache_svd: true
    batch_size: 8

# 权重调度 {#权重调度}
svd_scheduler:
  type: "cosine"  # cosine, linear, exponential, adaptive
  total_epochs: 100
  warmup_epochs: 10
  min_weight: 0.001
  max_weight: 0.1
```

### 🔧 集成到训练器 {#集成到训练器}

```python
class SVDTrainer:
    """集成SVD损失的训练器"""
    
    def __init__(self, model, config):
        self.model = model
        self.config = config
        
        # 创建SVD损失
        self.svd_loss = self._create_svd_loss()
        
        # 创建权重调度器
        self.weight_scheduler = self._create_weight_scheduler()
        
        # 创建分析器
        self.analyzer = SVDLossAnalyzer(model, self.svd_loss)
    
    def _create_svd_loss(self):
        svd_config = self.config['loss']['svd_loss']
        svd_type = svd_config['type']
        
        if svd_type == 'nuclear_norm':
            return NuclearNormSVDLoss(**svd_config)
        elif svd_type == 'adaptive':
            return AdaptiveSVDLoss(**svd_config)
        elif svd_type == 'weighted':
            return WeightedSVDLoss(**svd_config)
        # ... 其他类型
        else:
            raise ValueError(f"Unknown SVD loss type: {svd_type}")
    
    def _create_weight_scheduler(self):
        if 'svd_scheduler' in self.config:
            return DynamicSVDWeightScheduler(**self.config['svd_scheduler'])
        return None
    
    def train_epoch(self, epoch, dataloader):
        # 更新权重
        if self.weight_scheduler:
            new_weight = self.weight_scheduler.step(epoch)
            self.svd_loss.weight = new_weight
        
        total_loss = 0.0
        total_svd_loss = 0.0
        
        for batch in dataloader:
            # 前向传播
            outputs = self.model(batch['input'])
            main_loss = F.mse_loss(outputs, batch['target'])
            
            # SVD损失
            svd_result = self.svd_loss(self.model)
            svd_loss_value = svd_result['svd_loss']
            
            # 总损失
            loss = main_loss + svd_loss_value
            
            # 反向传播
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            
            total_loss += loss.item()
            total_svd_loss += svd_loss_value.item()
        
        # 记录分析数据
        metrics = {
            'train_loss': total_loss / len(dataloader),
            'svd_loss': total_svd_loss / len(dataloader)
        }
        self.analyzer.track_training_progress(epoch, metrics)
        
        return metrics
    
    def generate_analysis_report(self):
        return self.analyzer.generate_report()
```

---

## 📚 总结 {#总结}

SVD损失函数为VIVTransformer提供了强大的正则化能力，主要优势包括：

### 🎯 核心优势 {#核心优势}

1. **理论基础**: 基于矩阵分析和优化理论的坚实基础
2. **灵活性**: 多种变体适应不同的应用场景
3. **自适应性**: 动态调整正则化强度
4. **效率**: 优化的实现确保计算效率
5. **可解释性**: 提供丰富的分析和可视化工具

### 🚀 应用建议 {#应用建议}

- **初始权重**: 从较小的权重（0.001-0.01）开始
- **层选择**: 重点关注注意力层和前馈层
- **自适应策略**: 根据验证损失动态调整
- **性能监控**: 定期分析模型秩结构变化
- **超参数调优**: 使用网格搜索或贝叶斯优化

### 📈 未来方向 {#未来方向}

- **分布式SVD**: 支持大规模分布式训练
- **近似算法**: 更高效的SVD近似方法
- **多模态扩展**: 适应多模态学习场景
- **硬件优化**: 利用专用硬件加速SVD计算

通过合理使用SVD损失函数，可以显著提升VIVTransformer的训练稳定性和泛化能力。

---

*需要帮助？查看 [FAQ](faq.html) 或 [故障排除](troubleshooting.html) 页面。*
