---
layout: default
title: Loss Functions
description: 损失函数的设计和实现
permalink: /pages/loss-functions/
---

# 损失函数详解 {#损失函数详解}

本文档详细介绍VIVTransformer项目中使用的损失函数设计，包括基础损失、SVD正则化和组合策略。

## 📋 目录 {#目录}

- [损失函数概览](#损失函数概览)
- [基础损失函数](#基础损失函数)
- [SVD正则化](#svd正则化)
- [损失函数组合](#损失函数组合)
- [自适应权重](#自适应权重)
- [损失函数实现](#损失函数实现)
- [性能分析](#性能分析)
- [使用指南](#使用指南)

## 损失函数概览 {#损失函数概览}

### 🎯 设计目标 {#设计目标}

VIVTransformer的损失函数设计旨在实现：
- **精确预测**: 最小化预测误差
- **结构约束**: 通过SVD正则化保持模型结构
- **泛化能力**: 防止过拟合，提高泛化性能
- **计算效率**: 平衡精度和计算成本

### 🏗️ 损失架构 {#损失架构}

```
总损失函数
├── 📊 基础损失 (Base Loss)
│   ├── MSE损失
│   ├── MAE损失
│   ├── Huber损失
│   └── 自定义损失
├── 🔍 SVD正则化 (SVD Regularization)
│   ├── 奇异值约束
│   ├── 低秩约束
│   └── 结构保持
├── ⚖️ 权重策略 (Weighting Strategy)
│   ├── 固定权重
│   ├── 自适应权重
│   └── 动态调整
└── 🎛️ 组合策略 (Combination Strategy)
    ├── 加权求和
    ├── 多任务学习
    └── 不确定性加权
```

### 📐 数学表达式 {#数学表达式}

总损失函数的一般形式：

```
L_total = α * L_base + β * L_svd + γ * L_aux
```

其中：
- `L_base`: 基础损失函数
- `L_svd`: SVD正则化损失
- `L_aux`: 辅助损失函数
- `α, β, γ`: 权重参数

## 基础损失函数 {#基础损失函数}

### 📊 均方误差损失 (MSE) {#均方误差损失-mse}

**数学定义**:
```
L_MSE = (1/N) * Σ(y_pred - y_true)²
```

**特点**:
- 对大误差敏感
- 可微分，便于优化
- 适用于回归任务

**实现**:
```python
class MSELoss(nn.Module):
    """均方误差损失"""
    
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction
    
    def forward(self, pred, target):
        loss = (pred - target) ** 2
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
```

### 📏 平均绝对误差损失 (MAE) {#平均绝对误差损失-mae}

**数学定义**:
```
L_MAE = (1/N) * Σ|y_pred - y_true|
```

**特点**:
- 对异常值鲁棒
- 线性惩罚
- 适用于存在噪声的数据

**实现**:
```python
class MAELoss(nn.Module):
    """平均绝对误差损失"""
    
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction
    
    def forward(self, pred, target):
        loss = torch.abs(pred - target)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
```

### 🎯 Huber损失 {#huber损失}

**数学定义**:
```
L_Huber = {
    0.5 * (y_pred - y_true)²,     if |y_pred - y_true| ≤ δ
    δ * |y_pred - y_true| - 0.5 * δ², otherwise
}
```

**特点**:
- 结合MSE和MAE的优点
- 对小误差二次惩罚，对大误差线性惩罚
- 可调节的鲁棒性

**实现**:
```python
class HuberLoss(nn.Module):
    """Huber损失"""
    
    def __init__(self, delta=1.0, reduction='mean'):
        super().__init__()
        self.delta = delta
        self.reduction = reduction
    
    def forward(self, pred, target):
        residual = torch.abs(pred - target)
        
        # 小误差使用二次损失
        quadratic = torch.clamp(residual, max=self.delta)
        quadratic_loss = 0.5 * quadratic ** 2
        
        # 大误差使用线性损失
        linear = residual - quadratic
        linear_loss = self.delta * linear
        
        loss = quadratic_loss + linear_loss
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
```

### 🔄 自定义损失函数 {#自定义损失函数}

```python
class CustomRegressionLoss(nn.Module):
    """自定义回归损失"""
    
    def __init__(self, alpha=0.5, beta=0.3, gamma=0.2):
        super().__init__()
        self.alpha = alpha  # MSE权重
        self.beta = beta    # MAE权重
        self.gamma = gamma  # Huber权重
        
        self.mse_loss = MSELoss()
        self.mae_loss = MAELoss()
        self.huber_loss = HuberLoss(delta=1.0)
    
    def forward(self, pred, target):
        mse = self.mse_loss(pred, target)
        mae = self.mae_loss(pred, target)
        huber = self.huber_loss(pred, target)
        
        total_loss = self.alpha * mse + self.beta * mae + self.gamma * huber
        
        return total_loss, {
            'mse': mse.item(),
            'mae': mae.item(),
            'huber': huber.item(),
            'total': total_loss.item()
        }
```

## SVD正则化 {#svd正则化}

### 🔍 SVD正则化原理 {#svd正则化原理}

SVD（奇异值分解）正则化通过约束模型权重矩阵的奇异值来实现结构化正则化：

```
W = U * Σ * V^T
```

其中：
- `U, V`: 正交矩阵
- `Σ`: 对角矩阵，包含奇异值

### 📐 SVD损失函数 {#svd损失函数}

**数学定义**:
```
L_SVD = Σ(i=1 to k) w_i * σ_i
```

其中：
- `σ_i`: 第i个奇异值
- `w_i`: 第i个奇异值的权重
- `k`: 考虑的奇异值数量

**实现**:
```python
class SVDRegularization(nn.Module):
    """SVD正则化损失"""
    
    def __init__(self, weights=None, num_components=10):
        super().__init__()
        
        if weights is None:
            # 默认权重：递减权重
            weights = [0.1 * (0.9 ** i) for i in range(num_components)]
        
        self.register_buffer('weights', torch.tensor(weights, dtype=torch.float32))
        self.num_components = num_components
    
    def forward(self, weight_matrix):
        """计算SVD正则化损失"""
        
        # 执行SVD分解
        try:
            U, S, V = torch.svd(weight_matrix)
        except RuntimeError:
            # 如果SVD失败，返回零损失
            return torch.tensor(0.0, device=weight_matrix.device, requires_grad=True)
        
        # 取前k个奇异值
        singular_values = S[:self.num_components]
        weights = self.weights[:len(singular_values)]
        
        # 计算加权奇异值损失
        svd_loss = torch.sum(weights * singular_values)
        
        return svd_loss
    
    def get_singular_values(self, weight_matrix):
        """获取奇异值（用于分析）"""
        try:
            _, S, _ = torch.svd(weight_matrix)
            return S.detach().cpu().numpy()
        except RuntimeError:
            return np.array([])
```

### 🎛️ 多层SVD正则化 {#多层svd正则化}

```python
class MultiLayerSVDRegularization(nn.Module):
    """多层SVD正则化"""
    
    def __init__(self, layer_weights=None, svd_weights=None, num_components=10):
        super().__init__()
        
        self.svd_reg = SVDRegularization(svd_weights, num_components)
        
        # 不同层的权重
        if layer_weights is None:
            layer_weights = [1.0, 0.8, 0.6, 0.4, 0.2]  # 示例权重
        
        self.layer_weights = layer_weights
    
    def forward(self, model):
        """对模型的多个层应用SVD正则化"""
        
        total_svd_loss = 0.0
        layer_losses = {}
        
        # 遍历模型的注意力层
        for i, layer in enumerate(model.transformer_layers):
            if hasattr(layer, 'attention'):
                # 获取注意力权重矩阵
                if hasattr(layer.attention, 'query_projection'):
                    weight_matrix = layer.attention.query_projection.weight
                    
                    # 计算SVD损失
                    svd_loss = self.svd_reg(weight_matrix)
                    
                    # 应用层权重
                    layer_weight = self.layer_weights[i] if i < len(self.layer_weights) else 0.1
                    weighted_loss = layer_weight * svd_loss
                    
                    total_svd_loss += weighted_loss
                    layer_losses[f'layer_{i}'] = weighted_loss.item()
        
        return total_svd_loss, layer_losses
```

### 📊 自适应SVD权重 {#自适应svd权重}

```python
class AdaptiveSVDRegularization(nn.Module):
    """自适应SVD正则化"""
    
    def __init__(self, num_components=10, adaptation_rate=0.01):
        super().__init__()
        
        # 可学习的权重参数
        self.weights = nn.Parameter(torch.ones(num_components) * 0.1)
        self.num_components = num_components
        self.adaptation_rate = adaptation_rate
        
        # 奇异值历史（用于自适应调整）
        self.register_buffer('sv_history', torch.zeros(100, num_components))
        self.history_idx = 0
    
    def forward(self, weight_matrix):
        """自适应SVD正则化"""
        
        # 执行SVD分解
        try:
            U, S, V = torch.svd(weight_matrix)
        except RuntimeError:
            return torch.tensor(0.0, device=weight_matrix.device, requires_grad=True)
        
        # 取前k个奇异值
        singular_values = S[:self.num_components]
        
        # 更新奇异值历史
        if self.training:
            with torch.no_grad():
                self.sv_history[self.history_idx] = singular_values.detach()
                self.history_idx = (self.history_idx + 1) % self.sv_history.size(0)
        
        # 使用softmax确保权重为正且归一化
        normalized_weights = torch.softmax(self.weights, dim=0)
        
        # 计算加权奇异值损失
        svd_loss = torch.sum(normalized_weights * singular_values)
        
        return svd_loss
    
    def get_weight_statistics(self):
        """获取权重统计信息"""
        with torch.no_grad():
            weights = torch.softmax(self.weights, dim=0)
            return {
                'weights': weights.cpu().numpy(),
                'entropy': -torch.sum(weights * torch.log(weights + 1e-8)).item(),
                'max_weight': weights.max().item(),
                'min_weight': weights.min().item()
            }
```

## 损失函数组合 {#损失函数组合}

### ⚖️ 组合损失函数 {#组合损失函数}

```python
class CombinedLoss(nn.Module):
    """组合损失函数"""
    
    def __init__(self, config):
        super().__init__()
        
        # 基础损失
        self.base_weight = config.get('base_weight', 0.5)
        self.base_loss_type = config.get('base_loss', 'mse')
        
        if self.base_loss_type == 'mse':
            self.base_loss = MSELoss()
        elif self.base_loss_type == 'mae':
            self.base_loss = MAELoss()
        elif self.base_loss_type == 'huber':
            self.base_loss = HuberLoss()
        else:
            self.base_loss = CustomRegressionLoss()
        
        # SVD正则化
        self.use_svd = config.get('use_svd_regularization', True)
        if self.use_svd:
            svd_weights = config.get('svd_weights', None)
            self.svd_reg = SVDRegularization(svd_weights)
            self.svd_weight = config.get('svd_weight', 0.1)
        
        # 辅助损失
        self.aux_losses = config.get('auxiliary_losses', [])
        self.aux_weights = config.get('auxiliary_weights', [])
    
    def forward(self, pred, target, model=None):
        """计算组合损失"""
        
        losses = {}
        total_loss = 0.0
        
        # 1. 基础损失
        if isinstance(self.base_loss, CustomRegressionLoss):
            base_loss, base_components = self.base_loss(pred, target)
            losses.update(base_components)
        else:
            base_loss = self.base_loss(pred, target)
            losses['base'] = base_loss.item()
        
        total_loss += self.base_weight * base_loss
        
        # 2. SVD正则化
        if self.use_svd and model is not None:
            svd_loss = 0.0
            
            # 对所有线性层应用SVD正则化
            for name, module in model.named_modules():
                if isinstance(module, nn.Linear):
                    layer_svd_loss = self.svd_reg(module.weight)
                    svd_loss += layer_svd_loss
            
            losses['svd'] = svd_loss.item()
            total_loss += self.svd_weight * svd_loss
        
        # 3. 辅助损失
        for i, aux_loss_fn in enumerate(self.aux_losses):
            aux_weight = self.aux_weights[i] if i < len(self.aux_weights) else 0.1
            aux_loss = aux_loss_fn(pred, target)
            
            losses[f'aux_{i}'] = aux_loss.item()
            total_loss += aux_weight * aux_loss
        
        losses['total'] = total_loss.item()
        
        return total_loss, losses
```

### 🎯 多任务损失 {#多任务损失}

```python
class MultiTaskLoss(nn.Module):
    """多任务学习损失"""
    
    def __init__(self, task_weights=None, uncertainty_weighting=False):
        super().__init__()
        
        self.task_weights = task_weights or [1.0, 1.0, 1.0]
        self.uncertainty_weighting = uncertainty_weighting
        
        if uncertainty_weighting:
            # 可学习的不确定性参数
            self.log_vars = nn.Parameter(torch.zeros(len(self.task_weights)))
    
    def forward(self, predictions, targets):
        """多任务损失计算"""
        
        total_loss = 0.0
        task_losses = {}
        
        for i, (pred, target) in enumerate(zip(predictions, targets)):
            # 计算任务特定损失
            task_loss = F.mse_loss(pred, target)
            
            if self.uncertainty_weighting:
                # 使用不确定性加权
                precision = torch.exp(-self.log_vars[i])
                weighted_loss = precision * task_loss + self.log_vars[i]
            else:
                # 使用固定权重
                weighted_loss = self.task_weights[i] * task_loss
            
            total_loss += weighted_loss
            task_losses[f'task_{i}'] = task_loss.item()
            
            if self.uncertainty_weighting:
                task_losses[f'uncertainty_{i}'] = torch.exp(self.log_vars[i]).item()
        
        return total_loss, task_losses
```

## 自适应权重 {#自适应权重}

### 🔄 动态权重调整 {#动态权重调整}

```python
class DynamicWeightScheduler:
    """动态权重调度器"""
    
    def __init__(self, initial_weights, schedule_type='cosine'):
        self.initial_weights = initial_weights
        self.schedule_type = schedule_type
        self.current_weights = initial_weights.copy()
    
    def step(self, epoch, total_epochs, metrics=None):
        """更新权重"""
        
        if self.schedule_type == 'cosine':
            # 余弦调度
            progress = epoch / total_epochs
            factor = 0.5 * (1 + np.cos(np.pi * progress))
            
            for key in self.current_weights:
                self.current_weights[key] = self.initial_weights[key] * factor
        
        elif self.schedule_type == 'linear':
            # 线性调度
            progress = epoch / total_epochs
            factor = 1.0 - progress
            
            for key in self.current_weights:
                self.current_weights[key] = self.initial_weights[key] * factor
        
        elif self.schedule_type == 'adaptive' and metrics:
            # 基于性能的自适应调整
            self._adaptive_adjustment(metrics)
    
    def _adaptive_adjustment(self, metrics):
        """基于性能指标的自适应调整"""
        
        # 如果验证损失增加，增加正则化权重
        if 'val_loss_trend' in metrics:
            if metrics['val_loss_trend'] > 0:  # 损失增加
                self.current_weights['svd_weight'] *= 1.1
            else:  # 损失减少
                self.current_weights['svd_weight'] *= 0.95
        
        # 限制权重范围
        for key in self.current_weights:
            self.current_weights[key] = np.clip(
                self.current_weights[key], 0.001, 1.0
            )
    
    def get_weights(self):
        """获取当前权重"""
        return self.current_weights.copy()
```

### 📊 权重优化器 {#权重优化器}

```python
class LossWeightOptimizer:
    """损失权重优化器"""
    
    def __init__(self, initial_weights, learning_rate=0.01):
        self.weights = {k: v for k, v in initial_weights.items()}
        self.lr = learning_rate
        self.history = []
    
    def update_weights(self, loss_components, target_ratios=None):
        """基于损失组件更新权重"""
        
        if target_ratios is None:
            # 默认目标比例
            target_ratios = {
                'base': 0.7,
                'svd': 0.2,
                'aux': 0.1
            }
        
        # 计算当前比例
        total_loss = sum(loss_components.values())
        current_ratios = {k: v/total_loss for k, v in loss_components.items()}
        
        # 计算比例误差
        ratio_errors = {}
        for key in target_ratios:
            if key in current_ratios:
                ratio_errors[key] = target_ratios[key] - current_ratios[key]
        
        # 更新权重
        for key, error in ratio_errors.items():
            if key in self.weights:
                # 简单的比例控制
                self.weights[key] += self.lr * error
                self.weights[key] = max(0.001, self.weights[key])  # 确保权重为正
        
        # 记录历史
        self.history.append({
            'weights': self.weights.copy(),
            'ratios': current_ratios.copy(),
            'errors': ratio_errors.copy()
        })
    
    def get_weights(self):
        """获取当前权重"""
        return self.weights.copy()
    
    def plot_history(self):
        """绘制权重历史"""
        if not self.history:
            return
        
        import matplotlib.pyplot as plt
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # 权重历史
        for key in self.weights.keys():
            weights = [h['weights'][key] for h in self.history]
            ax1.plot(weights, label=f'{key}_weight')
        
        ax1.set_title('权重变化历史')
        ax1.set_xlabel('更新步数')
        ax1.set_ylabel('权重值')
        ax1.legend()
        ax1.grid(True)
        
        # 比例历史
        for key in self.weights.keys():
            ratios = [h['ratios'].get(key, 0) for h in self.history]
            ax2.plot(ratios, label=f'{key}_ratio')
        
        ax2.set_title('损失比例历史')
        ax2.set_xlabel('更新步数')
        ax2.set_ylabel('比例')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()
```

## 性能分析 {#性能分析}

### 📊 损失函数性能评估 {#损失函数性能评估}

```python
class LossAnalyzer:
    """损失函数分析器"""
    
    def __init__(self):
        self.loss_history = []
        self.component_history = []
    
    def record_loss(self, total_loss, components):
        """记录损失值"""
        self.loss_history.append(total_loss)
        self.component_history.append(components.copy())
    
    def analyze_convergence(self, window_size=10):
        """分析收敛性"""
        if len(self.loss_history) < window_size:
            return None
        
        recent_losses = self.loss_history[-window_size:]
        
        # 计算趋势
        x = np.arange(len(recent_losses))
        slope, intercept = np.polyfit(x, recent_losses, 1)
        
        # 计算变异系数
        cv = np.std(recent_losses) / np.mean(recent_losses)
        
        return {
            'trend_slope': slope,
            'coefficient_of_variation': cv,
            'is_converging': slope < -1e-6 and cv < 0.1,
            'recent_mean': np.mean(recent_losses),
            'recent_std': np.std(recent_losses)
        }
    
    def analyze_components(self):
        """分析损失组件"""
        if not self.component_history:
            return None
        
        # 提取各组件的历史
        component_stats = {}
        
        for key in self.component_history[0].keys():
            values = [comp[key] for comp in self.component_history]
            
            component_stats[key] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'final': values[-1],
                'trend': np.polyfit(range(len(values)), values, 1)[0]
            }
        
        return component_stats
    
    def plot_loss_history(self):
        """绘制损失历史"""
        if not self.loss_history:
            return
        
        import matplotlib.pyplot as plt
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # 总损失
        ax1.plot(self.loss_history, 'b-', linewidth=2)
        ax1.set_title('总损失变化')
        ax1.set_xlabel('训练步数')
        ax1.set_ylabel('损失值')
        ax1.grid(True)
        
        # 损失组件
        if self.component_history:
            for key in self.component_history[0].keys():
                if key != 'total':
                    values = [comp[key] for comp in self.component_history]
                    ax2.plot(values, label=key)
            
            ax2.set_title('损失组件变化')
            ax2.set_xlabel('训练步数')
            ax2.set_ylabel('损失值')
            ax2.legend()
            ax2.grid(True)
        
        plt.tight_layout()
        plt.show()
```

## 使用指南 {#使用指南}

### 🚀 快速开始 {#快速开始}

```python
# 1. 基础使用 {#1-基础使用}
from loss_functions import CombinedLoss

# 配置损失函数 {#配置损失函数}
loss_config = {
    'base_weight': 0.7,
    'base_loss': 'mse',
    'use_svd_regularization': True,
    'svd_weights': [0.1, 0.08, 0.06, 0.04, 0.02],
    'svd_weight': 0.2
}

# 创建损失函数 {#创建损失函数}
criterion = CombinedLoss(loss_config)

# 在训练循环中使用 {#在训练循环中使用}
for batch in dataloader:
    pred = model(batch['input'])
    target = batch['target']
    
    loss, loss_components = criterion(pred, target, model)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print(f"Loss: {loss.item():.6f}, Components: {loss_components}")
```

### 🎛️ 高级配置 {#高级配置}

```python
# 2. 自适应权重使用 {#2-自适应权重使用}
from loss_functions import AdaptiveSVDRegularization, DynamicWeightScheduler

# 自适应SVD正则化 {#自适应svd正则化}
svd_reg = AdaptiveSVDRegularization(num_components=10)

# 动态权重调度 {#动态权重调度}
weight_scheduler = DynamicWeightScheduler(
    initial_weights={'base_weight': 0.7, 'svd_weight': 0.2},
    schedule_type='cosine'
)

# 在训练过程中更新权重 {#在训练过程中更新权重}
for epoch in range(num_epochs):
    # 更新权重
    current_weights = weight_scheduler.step(epoch, num_epochs)
    
    # 更新损失函数权重
    criterion.base_weight = current_weights['base_weight']
    criterion.svd_weight = current_weights['svd_weight']
    
    # 训练一个epoch
    train_epoch(model, dataloader, criterion, optimizer)
```

### 📊 损失分析 {#损失分析}

```python
# 3. 损失分析 {#3-损失分析}
from loss_functions import LossAnalyzer

analyzer = LossAnalyzer()

# 在训练过程中记录损失 {#在训练过程中记录损失}
for epoch in range(num_epochs):
    for batch in dataloader:
        loss, components = criterion(pred, target, model)
        
        # 记录损失
        analyzer.record_loss(loss.item(), components)
        
        # 训练步骤...

# 分析结果 {#分析结果}
convergence_info = analyzer.analyze_convergence()
component_stats = analyzer.analyze_components()

print("收敛分析:", convergence_info)
print("组件统计:", component_stats)

# 可视化 {#可视化}
analyzer.plot_loss_history()
```

### ⚙️ 配置建议 {#配置建议}

**初学者配置**:
```yaml
loss:
  base_weight: 1.0
  base_loss: "mse"
  use_svd_regularization: false
```

**标准配置**:
```yaml
loss:
  base_weight: 0.7
  base_loss: "mse"
  use_svd_regularization: true
  svd_weights: [0.1, 0.08, 0.06, 0.04, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02]
  svd_weight: 0.2
```

**高级配置**:
```yaml
loss:
  base_weight: 0.5
  base_loss: "custom"
  use_svd_regularization: true
  svd_weights: "adaptive"  # 使用自适应权重
  svd_weight: 0.3
  auxiliary_losses: ["consistency", "diversity"]
  auxiliary_weights: [0.1, 0.1]
```

---

**💡 提示**: 损失函数的设计和调优是深度学习成功的关键。建议从简单配置开始，根据实验结果逐步调整权重和组合策略。记住，不同的任务可能需要不同的损失函数配置！

---

*需要帮助？查看 [FAQ](faq.html) 或 [故障排除](troubleshooting.html) 页面。*
