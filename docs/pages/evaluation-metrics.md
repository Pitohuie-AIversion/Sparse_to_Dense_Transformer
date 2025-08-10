---
layout: doc
title: Evaluation Metrics
parent: Evaluation & Results
nav_order: 1
description: "模型评估指标和方法"
permalink: /pages/evaluation-metrics/
---

# 评估指标详解 {#评估指标详解}

本文档详细介绍VIVTransformer项目中使用的评估指标，包括性能指标、效率指标和可解释性指标。

## 📋 目录 {#目录}

- [评估指标概览](#评估指标概览)
- [性能指标](#性能指标)
- [效率指标](#效率指标)
- [可解释性指标](#可解释性指标)
- [注意力分析指标](#注意力分析指标)
- [指标实现](#指标实现)
- [评估框架](#评估框架)
- [使用指南](#使用指南)

## 评估指标概览 {#评估指标概览}

### 🎯 评估目标 {#评估目标}

VIVTransformer的评估体系旨在全面衡量模型的：
- **预测精度**: 模型输出的准确性
- **计算效率**: 训练和推理的速度
- **内存效率**: 内存使用情况
- **可解释性**: 注意力机制的可理解性
- **鲁棒性**: 对噪声和异常的抵抗能力

### 🏗️ 指标体系 {#指标体系}

```
评估指标体系
├── 📊 性能指标 (Performance Metrics)
│   ├── 回归指标 (MSE, MAE, R²)
│   ├── 分类指标 (Accuracy, F1-Score)
│   └── 排序指标 (NDCG, MAP)
├── ⚡ 效率指标 (Efficiency Metrics)
│   ├── 计算效率 (FLOPs, 推理时间)
│   ├── 内存效率 (参数量, 内存占用)
│   └── 能耗指标 (功耗, 碳排放)
├── 🔍 可解释性指标 (Interpretability Metrics)
│   ├── 注意力一致性
│   ├── 梯度稳定性
│   └── 特征重要性
└── 🎛️ 综合指标 (Composite Metrics)
    ├── 效率-精度权衡
    ├── 鲁棒性评分
    └── 整体性能指数
```

## 性能指标 {#性能指标}

### 📊 回归指标 {#回归指标}

#### 均方误差 (MSE) {#均方误差-mse}

**数学定义**:
```
MSE = (1/N) * Σ(y_pred - y_true)²
```

**特点**:
- 对大误差敏感
- 单位为原始数据单位的平方
- 值越小越好

**实现**:
```python
import torch
import numpy as np
from typing import Dict, List, Tuple

class MSEMetric:
    """均方误差指标"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """重置指标"""
        self.total_squared_error = 0.0
        self.total_samples = 0
    
    def update(self, pred: torch.Tensor, target: torch.Tensor):
        """更新指标"""
        squared_error = torch.sum((pred - target) ** 2)
        self.total_squared_error += squared_error.item()
        self.total_samples += pred.numel()
    
    def compute(self) -> float:
        """计算MSE"""
        if self.total_samples == 0:
            return 0.0
        return self.total_squared_error / self.total_samples
    
    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """直接计算MSE（不累积）"""
        return torch.mean((pred - target) ** 2).item()
```

#### 平均绝对误差 (MAE) {#平均绝对误差-mae}

**数学定义**:
```
MAE = (1/N) * Σ|y_pred - y_true|
```

**特点**:
- 对异常值鲁棒
- 单位与原始数据相同
- 易于解释

**实现**:
```python
class MAEMetric:
    """平均绝对误差指标"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.total_absolute_error = 0.0
        self.total_samples = 0
    
    def update(self, pred: torch.Tensor, target: torch.Tensor):
        absolute_error = torch.sum(torch.abs(pred - target))
        self.total_absolute_error += absolute_error.item()
        self.total_samples += pred.numel()
    
    def compute(self) -> float:
        if self.total_samples == 0:
            return 0.0
        return self.total_absolute_error / self.total_samples
    
    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        return torch.mean(torch.abs(pred - target)).item()
```

#### 决定系数 (R²) {#决定系数-r}

**数学定义**:
```
R² = 1 - (SS_res / SS_tot)
其中:
SS_res = Σ(y_true - y_pred)²
SS_tot = Σ(y_true - y_mean)²
```

**特点**:
- 范围通常在[0, 1]，越接近1越好
- 表示模型解释的方差比例
- 可能为负值（模型比均值预测还差）

**实现**:
```python
class R2Metric:
    """决定系数指标"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.predictions = []
        self.targets = []
    
    def update(self, pred: torch.Tensor, target: torch.Tensor):
        self.predictions.append(pred.detach().cpu())
        self.targets.append(target.detach().cpu())
    
    def compute(self) -> float:
        if not self.predictions:
            return 0.0
        
        pred = torch.cat(self.predictions, dim=0)
        target = torch.cat(self.targets, dim=0)
        
        # 计算总平方和和残差平方和
        target_mean = torch.mean(target)
        ss_tot = torch.sum((target - target_mean) ** 2)
        ss_res = torch.sum((target - pred) ** 2)
        
        if ss_tot == 0:
            return 1.0 if ss_res == 0 else 0.0
        
        r2 = 1 - (ss_res / ss_tot)
        return r2.item()
    
    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        target_mean = torch.mean(target)
        ss_tot = torch.sum((target - target_mean) ** 2)
        ss_res = torch.sum((target - pred) ** 2)
        
        if ss_tot == 0:
            return 1.0 if ss_res == 0 else 0.0
        
        r2 = 1 - (ss_res / ss_tot)
        return r2.item()
```

#### 平均绝对百分比误差 (MAPE) {#平均绝对百分比误差-mape}

**数学定义**:
```
MAPE = (100/N) * Σ|((y_true - y_pred) / y_true)|
```

**实现**:
```python
class MAPEMetric:
    """平均绝对百分比误差指标"""
    
    def __init__(self, epsilon=1e-8):
        self.epsilon = epsilon  # 避免除零
        self.reset()
    
    def reset(self):
        self.total_percentage_error = 0.0
        self.total_samples = 0
    
    def update(self, pred: torch.Tensor, target: torch.Tensor):
        # 避免除零
        target_safe = torch.where(torch.abs(target) < self.epsilon, 
                                 torch.sign(target) * self.epsilon, target)
        
        percentage_error = torch.abs((target - pred) / target_safe) * 100
        self.total_percentage_error += torch.sum(percentage_error).item()
        self.total_samples += pred.numel()
    
    def compute(self) -> float:
        if self.total_samples == 0:
            return 0.0
        return self.total_percentage_error / self.total_samples
    
    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        target_safe = torch.where(torch.abs(target) < self.epsilon, 
                                 torch.sign(target) * self.epsilon, target)
        percentage_error = torch.abs((target - pred) / target_safe) * 100
        return torch.mean(percentage_error).item()
```

### 🎯 分类指标 {#分类指标}

```python
class ClassificationMetrics:
    """分类指标集合"""
    
    def __init__(self, num_classes: int, average: str = 'macro'):
        self.num_classes = num_classes
        self.average = average  # 'macro', 'micro', 'weighted'
        self.reset()
    
    def reset(self):
        self.confusion_matrix = torch.zeros(self.num_classes, self.num_classes)
        self.total_samples = 0
        self.correct_predictions = 0
    
    def update(self, pred: torch.Tensor, target: torch.Tensor):
        """更新混淆矩阵"""
        pred_labels = torch.argmax(pred, dim=-1)
        
        for p, t in zip(pred_labels.flatten(), target.flatten()):
            self.confusion_matrix[t.long(), p.long()] += 1
        
        self.total_samples += target.numel()
        self.correct_predictions += (pred_labels == target).sum().item()
    
    def accuracy(self) -> float:
        """计算准确率"""
        if self.total_samples == 0:
            return 0.0
        return self.correct_predictions / self.total_samples
    
    def precision(self) -> float:
        """计算精确率"""
        if self.confusion_matrix.sum() == 0:
            return 0.0
        
        precisions = []
        for i in range(self.num_classes):
            tp = self.confusion_matrix[i, i]
            fp = self.confusion_matrix[:, i].sum() - tp
            
            if tp + fp == 0:
                precisions.append(0.0)
            else:
                precisions.append((tp / (tp + fp)).item())
        
        if self.average == 'macro':
            return np.mean(precisions)
        elif self.average == 'micro':
            tp_total = torch.diag(self.confusion_matrix).sum()
            fp_total = self.confusion_matrix.sum() - tp_total
            return (tp_total / (tp_total + fp_total)).item()
        
        return np.mean(precisions)
    
    def recall(self) -> float:
        """计算召回率"""
        if self.confusion_matrix.sum() == 0:
            return 0.0
        
        recalls = []
        for i in range(self.num_classes):
            tp = self.confusion_matrix[i, i]
            fn = self.confusion_matrix[i, :].sum() - tp
            
            if tp + fn == 0:
                recalls.append(0.0)
            else:
                recalls.append((tp / (tp + fn)).item())
        
        if self.average == 'macro':
            return np.mean(recalls)
        elif self.average == 'micro':
            tp_total = torch.diag(self.confusion_matrix).sum()
            fn_total = self.confusion_matrix.sum() - tp_total
            return (tp_total / (tp_total + fn_total)).item()
        
        return np.mean(recalls)
    
    def f1_score(self) -> float:
        """计算F1分数"""
        prec = self.precision()
        rec = self.recall()
        
        if prec + rec == 0:
            return 0.0
        
        return 2 * (prec * rec) / (prec + rec)
```

## 效率指标 {#效率指标}

### ⚡ 计算效率指标 {#计算效率指标}

```python
import time
import psutil
import torch.profiler
from typing import Dict, Any

class ComputeEfficiencyMetrics:
    """计算效率指标"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.inference_times = []
        self.memory_usage = []
        self.flops_count = 0
    
    def measure_inference_time(self, model, input_data, num_runs=100, warmup_runs=10):
        """测量推理时间"""
        model.eval()
        
        # 预热
        with torch.no_grad():
            for _ in range(warmup_runs):
                _ = model(input_data)
        
        # 同步GPU（如果使用）
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        # 测量推理时间
        times = []
        with torch.no_grad():
            for _ in range(num_runs):
                start_time = time.perf_counter()
                _ = model(input_data)
                
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                end_time = time.perf_counter()
                times.append(end_time - start_time)
        
        self.inference_times.extend(times)
        
        return {
            'mean_time': np.mean(times),
            'std_time': np.std(times),
            'min_time': np.min(times),
            'max_time': np.max(times),
            'throughput': input_data.size(0) / np.mean(times)  # samples/second
        }
    
    def measure_memory_usage(self, model, input_data):
        """测量内存使用"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
            # 测量推理内存
            with torch.no_grad():
                _ = model(input_data)
            
            memory_stats = {
                'peak_memory_mb': torch.cuda.max_memory_allocated() / 1024**2,
                'current_memory_mb': torch.cuda.memory_allocated() / 1024**2,
                'cached_memory_mb': torch.cuda.memory_reserved() / 1024**2
            }
        else:
            # CPU内存测量
            process = psutil.Process()
            memory_before = process.memory_info().rss / 1024**2
            
            with torch.no_grad():
                _ = model(input_data)
            
            memory_after = process.memory_info().rss / 1024**2
            
            memory_stats = {
                'memory_increase_mb': memory_after - memory_before,
                'total_memory_mb': memory_after
            }
        
        return memory_stats
    
    def count_flops(self, model, input_data):
        """计算FLOPs"""
        try:
            from fvcore.nn import FlopCountMode, flop_count
            
            flops_dict, _ = flop_count(
                model, 
                (input_data,),
                supported_ops=None
            )
            
            total_flops = sum(flops_dict.values())
            self.flops_count = total_flops
            
            return {
                'total_flops': total_flops,
                'flops_per_sample': total_flops / input_data.size(0),
                'gflops': total_flops / 1e9
            }
        except ImportError:
            print("警告: fvcore未安装，无法计算FLOPs")
            return {'total_flops': 0, 'flops_per_sample': 0, 'gflops': 0}
    
    def profile_model(self, model, input_data, output_path="profile_trace.json"):
        """详细性能分析"""
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
            profile_memory=True,
            with_stack=True
        ) as prof:
            with torch.no_grad():
                _ = model(input_data)
        
        # 保存分析结果
        prof.export_chrome_trace(output_path)
        
        # 返回关键统计信息
        key_averages = prof.key_averages(group_by_stack_n=5)
        
        cpu_time = sum([item.cpu_time_total for item in key_averages])
        cuda_time = sum([item.cuda_time_total for item in key_averages])
        
        return {
            'cpu_time_ms': cpu_time / 1000,
            'cuda_time_ms': cuda_time / 1000,
            'profile_path': output_path
        }
```

### 💾 内存效率指标 {#内存效率指标}

```python
class MemoryEfficiencyMetrics:
    """内存效率指标"""
    
    def __init__(self):
        pass
    
    def count_parameters(self, model):
        """计算模型参数量"""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'non_trainable_parameters': total_params - trainable_params,
            'model_size_mb': total_params * 4 / 1024**2  # 假设float32
        }
    
    def analyze_layer_memory(self, model):
        """分析各层内存使用"""
        layer_info = []
        
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # 叶子节点
                params = sum(p.numel() for p in module.parameters())
                memory_mb = params * 4 / 1024**2
                
                layer_info.append({
                    'name': name,
                    'type': type(module).__name__,
                    'parameters': params,
                    'memory_mb': memory_mb
                })
        
        # 按内存使用排序
        layer_info.sort(key=lambda x: x['memory_mb'], reverse=True)
        
        return layer_info
    
    def memory_efficiency_score(self, model, baseline_params=1e6):
        """计算内存效率分数"""
        params = self.count_parameters(model)['total_parameters']
        
        # 效率分数：基线参数量 / 实际参数量
        efficiency = baseline_params / params
        
        return {
            'efficiency_score': efficiency,
            'efficiency_level': self._get_efficiency_level(efficiency)
        }
    
    def _get_efficiency_level(self, score):
        """获取效率等级"""
        if score >= 2.0:
            return "非常高效"
        elif score >= 1.5:
            return "高效"
        elif score >= 1.0:
            return "中等"
        elif score >= 0.5:
            return "低效"
        else:
            return "非常低效"
```

## 可解释性指标 {#可解释性指标}

### 🔍 注意力分析指标 {#注意力分析指标}

```python
class AttentionAnalysisMetrics:
    """注意力分析指标"""
    
    def __init__(self):
        pass
    
    def attention_entropy(self, attention_weights):
        """计算注意力熵"""
        # attention_weights: [batch_size, num_heads, seq_len, seq_len]
        
        # 添加小值避免log(0)
        eps = 1e-8
        attention_weights = attention_weights + eps
        
        # 计算熵
        entropy = -torch.sum(attention_weights * torch.log(attention_weights), dim=-1)
        
        return {
            'mean_entropy': entropy.mean().item(),
            'std_entropy': entropy.std().item(),
            'entropy_per_head': entropy.mean(dim=(0, 2)).cpu().numpy(),
            'entropy_per_position': entropy.mean(dim=(0, 1)).cpu().numpy()
        }
    
    def attention_sparsity(self, attention_weights, threshold=0.1):
        """计算注意力稀疏性"""
        # 计算超过阈值的注意力权重比例
        above_threshold = (attention_weights > threshold).float()
        sparsity = 1.0 - above_threshold.mean()
        
        return {
            'sparsity_ratio': sparsity.item(),
            'active_ratio': above_threshold.mean().item(),
            'sparsity_per_head': (1.0 - above_threshold.mean(dim=(0, 2, 3))).cpu().numpy()
        }
    
    def attention_consistency(self, attention_weights_list):
        """计算注意力一致性（跨时间步或样本）"""
        if len(attention_weights_list) < 2:
            return {'consistency_score': 1.0}
        
        # 计算相邻注意力图的相似性
        similarities = []
        
        for i in range(len(attention_weights_list) - 1):
            att1 = attention_weights_list[i]
            att2 = attention_weights_list[i + 1]
            
            # 计算余弦相似度
            att1_flat = att1.flatten()
            att2_flat = att2.flatten()
            
            similarity = torch.cosine_similarity(att1_flat, att2_flat, dim=0)
            similarities.append(similarity.item())
        
        return {
            'consistency_score': np.mean(similarities),
            'consistency_std': np.std(similarities),
            'pairwise_similarities': similarities
        }
    
    def attention_head_diversity(self, attention_weights):
        """计算注意力头多样性"""
        # attention_weights: [batch_size, num_heads, seq_len, seq_len]
        batch_size, num_heads, seq_len, _ = attention_weights.shape
        
        # 计算头之间的相似性
        head_similarities = []
        
        for i in range(num_heads):
            for j in range(i + 1, num_heads):
                head_i = attention_weights[:, i, :, :].flatten()
                head_j = attention_weights[:, j, :, :].flatten()
                
                similarity = torch.cosine_similarity(head_i, head_j, dim=0)
                head_similarities.append(similarity.item())
        
        # 多样性 = 1 - 平均相似性
        diversity = 1.0 - np.mean(head_similarities)
        
        return {
            'diversity_score': diversity,
            'mean_similarity': np.mean(head_similarities),
            'similarity_std': np.std(head_similarities)
        }
    
    def attention_focus_score(self, attention_weights):
        """计算注意力聚焦程度"""
        # 计算每个位置的最大注意力权重
        max_attention = torch.max(attention_weights, dim=-1)[0]
        
        # 聚焦分数：高权重位置的比例
        focus_threshold = 0.5
        focused_positions = (max_attention > focus_threshold).float()
        
        return {
            'focus_score': focused_positions.mean().item(),
            'max_attention_mean': max_attention.mean().item(),
            'max_attention_std': max_attention.std().item()
        }
```

### 📊 梯度分析指标 {#梯度分析指标}

```python
class GradientAnalysisMetrics:
    """梯度分析指标"""
    
    def __init__(self):
        pass
    
    def gradient_norm(self, model):
        """计算梯度范数"""
        total_norm = 0.0
        param_count = 0
        
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                param_count += 1
        
        total_norm = total_norm ** (1. / 2)
        
        return {
            'total_gradient_norm': total_norm,
            'average_gradient_norm': total_norm / max(param_count, 1)
        }
    
    def gradient_stability(self, gradient_norms_history):
        """计算梯度稳定性"""
        if len(gradient_norms_history) < 2:
            return {'stability_score': 1.0}
        
        # 计算梯度范数的变异系数
        mean_norm = np.mean(gradient_norms_history)
        std_norm = np.std(gradient_norms_history)
        
        cv = std_norm / (mean_norm + 1e-8)
        stability = 1.0 / (1.0 + cv)  # 稳定性分数
        
        return {
            'stability_score': stability,
            'coefficient_of_variation': cv,
            'mean_gradient_norm': mean_norm,
            'std_gradient_norm': std_norm
        }
    
    def layer_gradient_analysis(self, model):
        """分析各层梯度"""
        layer_gradients = {}
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.data.norm(2).item()
                grad_mean = param.grad.data.mean().item()
                grad_std = param.grad.data.std().item()
                
                layer_gradients[name] = {
                    'norm': grad_norm,
                    'mean': grad_mean,
                    'std': grad_std,
                    'shape': list(param.shape)
                }
        
        return layer_gradients
```

## 评估框架 {#评估框架}

### 🔧 综合评估器 {#综合评估器}

```python
class ComprehensiveEvaluator:
    """综合评估器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # 初始化各类指标
        self.performance_metrics = self._init_performance_metrics()
        self.efficiency_metrics = ComputeEfficiencyMetrics()
        self.memory_metrics = MemoryEfficiencyMetrics()
        self.attention_metrics = AttentionAnalysisMetrics()
        self.gradient_metrics = GradientAnalysisMetrics()
        
        # 评估历史
        self.evaluation_history = []
    
    def _init_performance_metrics(self):
        """初始化性能指标"""
        metrics = {
            'mse': MSEMetric(),
            'mae': MAEMetric(),
            'r2': R2Metric(),
            'mape': MAPEMetric()
        }
        
        if self.config.get('task_type') == 'classification':
            num_classes = self.config.get('num_classes', 2)
            metrics['classification'] = ClassificationMetrics(num_classes)
        
        return metrics
    
    def evaluate_model(self, model, dataloader, device='cpu'):
        """全面评估模型"""
        model.eval()
        
        # 重置所有指标
        for metric in self.performance_metrics.values():
            if hasattr(metric, 'reset'):
                metric.reset()
        
        all_predictions = []
        all_targets = []
        attention_weights_list = []
        
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(dataloader):
                data = data.to(device)
                target = target.to(device)
                
                # 前向传播
                if hasattr(model, 'forward_with_attention'):
                    output, attention_weights = model.forward_with_attention(data)
                    attention_weights_list.append(attention_weights)
                else:
                    output = model(data)
                
                # 更新性能指标
                for metric in self.performance_metrics.values():
                    if hasattr(metric, 'update'):
                        metric.update(output, target)
                
                all_predictions.append(output.cpu())
                all_targets.append(target.cpu())
        
        # 计算性能指标
        performance_results = {}
        for name, metric in self.performance_metrics.items():
            if hasattr(metric, 'compute'):
                performance_results[name] = metric.compute()
        
        # 计算效率指标
        sample_input = next(iter(dataloader))[0][:1].to(device)
        efficiency_results = self._evaluate_efficiency(model, sample_input)
        
        # 计算注意力指标
        attention_results = {}
        if attention_weights_list:
            attention_results = self._evaluate_attention(attention_weights_list)
        
        # 综合结果
        evaluation_result = {
            'performance': performance_results,
            'efficiency': efficiency_results,
            'attention': attention_results,
            'timestamp': time.time()
        }
        
        self.evaluation_history.append(evaluation_result)
        
        return evaluation_result
    
    def _evaluate_efficiency(self, model, sample_input):
        """评估效率指标"""
        results = {}
        
        # 推理时间
        timing_results = self.efficiency_metrics.measure_inference_time(
            model, sample_input, num_runs=50
        )
        results.update(timing_results)
        
        # 内存使用
        memory_results = self.efficiency_metrics.measure_memory_usage(
            model, sample_input
        )
        results.update(memory_results)
        
        # 参数量
        param_results = self.memory_metrics.count_parameters(model)
        results.update(param_results)
        
        # FLOPs
        flops_results = self.efficiency_metrics.count_flops(model, sample_input)
        results.update(flops_results)
        
        return results
    
    def _evaluate_attention(self, attention_weights_list):
        """评估注意力指标"""
        if not attention_weights_list:
            return {}
        
        # 使用第一个批次的注意力权重进行分析
        attention_weights = attention_weights_list[0]
        
        results = {}
        
        # 注意力熵
        entropy_results = self.attention_metrics.attention_entropy(attention_weights)
        results.update({f'entropy_{k}': v for k, v in entropy_results.items()})
        
        # 注意力稀疏性
        sparsity_results = self.attention_metrics.attention_sparsity(attention_weights)
        results.update({f'sparsity_{k}': v for k, v in sparsity_results.items()})
        
        # 注意力头多样性
        diversity_results = self.attention_metrics.attention_head_diversity(attention_weights)
        results.update({f'diversity_{k}': v for k, v in diversity_results.items()})
        
        # 注意力聚焦程度
        focus_results = self.attention_metrics.attention_focus_score(attention_weights)
        results.update({f'focus_{k}': v for k, v in focus_results.items()})
        
        # 如果有多个时间步，计算一致性
        if len(attention_weights_list) > 1:
            consistency_results = self.attention_metrics.attention_consistency(
                attention_weights_list[:5]  # 使用前5个批次
            )
            results.update({f'consistency_{k}': v for k, v in consistency_results.items()})
        
        return results
    
    def generate_report(self, save_path=None):
        """生成评估报告"""
        if not self.evaluation_history:
            return "没有评估历史数据"
        
        latest_result = self.evaluation_history[-1]
        
        report = []
        report.append("# VIVTransformer 模型评估报告\n")
        report.append(f"评估时间: {time.ctime(latest_result['timestamp'])}\n")
        
        # 性能指标
        report.append("## 性能指标")
        for metric, value in latest_result['performance'].items():
            if isinstance(value, float):
                report.append(f"- {metric.upper()}: {value:.6f}")
        report.append("")
        
        # 效率指标
        report.append("## 效率指标")
        efficiency = latest_result['efficiency']
        report.append(f"- 平均推理时间: {efficiency.get('mean_time', 0):.4f} 秒")
        report.append(f"- 吞吐量: {efficiency.get('throughput', 0):.2f} 样本/秒")
        report.append(f"- 总参数量: {efficiency.get('total_parameters', 0):,}")
        report.append(f"- 模型大小: {efficiency.get('model_size_mb', 0):.2f} MB")
        
        if 'peak_memory_mb' in efficiency:
            report.append(f"- 峰值GPU内存: {efficiency['peak_memory_mb']:.2f} MB")
        
        if 'gflops' in efficiency:
            report.append(f"- 计算量: {efficiency['gflops']:.2f} GFLOPs")
        
        report.append("")
        
        # 注意力分析
        if latest_result['attention']:
            report.append("## 注意力分析")
            attention = latest_result['attention']
            
            if 'entropy_mean_entropy' in attention:
                report.append(f"- 平均注意力熵: {attention['entropy_mean_entropy']:.4f}")
            
            if 'sparsity_sparsity_ratio' in attention:
                report.append(f"- 注意力稀疏性: {attention['sparsity_sparsity_ratio']:.4f}")
            
            if 'diversity_diversity_score' in attention:
                report.append(f"- 注意力头多样性: {attention['diversity_diversity_score']:.4f}")
            
            if 'focus_focus_score' in attention:
                report.append(f"- 注意力聚焦程度: {attention['focus_focus_score']:.4f}")
            
            report.append("")
        
        # 综合评分
        report.append("## 综合评分")
        overall_score = self._calculate_overall_score(latest_result)
        report.append(f"- 综合性能分数: {overall_score:.2f}/100")
        
        report_text = "\n".join(report)
        
        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(report_text)
            print(f"评估报告已保存到: {save_path}")
        
        return report_text
    
    def _calculate_overall_score(self, result):
        """计算综合评分"""
        score = 0.0
        
        # 性能分数 (40%)
        performance = result['performance']
        if 'r2' in performance:
            score += max(0, performance['r2']) * 40
        elif 'mse' in performance:
            # MSE越小越好，转换为分数
            mse_score = max(0, 1 - performance['mse'])
            score += mse_score * 40
        
        # 效率分数 (30%)
        efficiency = result['efficiency']
        if 'throughput' in efficiency:
            # 标准化吞吐量分数
            throughput_score = min(1.0, efficiency['throughput'] / 100)
            score += throughput_score * 30
        
        # 注意力质量分数 (20%)
        attention = result['attention']
        if attention:
            attention_score = 0
            if 'diversity_diversity_score' in attention:
                attention_score += attention['diversity_diversity_score'] * 10
            if 'focus_focus_score' in attention:
                attention_score += attention['focus_focus_score'] * 10
            score += attention_score
        
        # 模型复杂度分数 (10%)
        if 'total_parameters' in efficiency:
            # 参数量适中得分更高
            param_millions = efficiency['total_parameters'] / 1e6
            if param_millions < 1:
                complexity_score = 10
            elif param_millions < 10:
                complexity_score = 8
            elif param_millions < 100:
                complexity_score = 6
            else:
                complexity_score = 4
            score += complexity_score
        
        return min(100, max(0, score))
```

## 使用指南 {#使用指南}

### 🚀 快速开始 {#快速开始}

```python
# 基础评估示例 {#基础评估示例}
from evaluation_metrics import ComprehensiveEvaluator

# 配置评估器 {#配置评估器}
eval_config = {
    'task_type': 'regression',  # 或 'classification'
    'num_classes': 10  # 仅分类任务需要
}

# 创建评估器 {#创建评估器}
evaluator = ComprehensiveEvaluator(eval_config)

# 评估模型 {#评估模型}
results = evaluator.evaluate_model(model, test_dataloader, device='cuda')

# 打印结果 {#打印结果}
print("性能指标:", results['performance'])
print("效率指标:", results['efficiency'])
print("注意力分析:", results['attention'])

# 生成报告 {#生成报告}
report = evaluator.generate_report('evaluation_report.md')
print(report)
```

### 📊 自定义评估 {#自定义评估}

```python
# 单独使用特定指标 {#单独使用特定指标}
from evaluation_metrics import MSEMetric, AttentionAnalysisMetrics

# 性能评估 {#性能评估}
mse_metric = MSEMetric()
for pred, target in predictions:
    mse_metric.update(pred, target)

final_mse = mse_metric.compute()
print(f"MSE: {final_mse:.6f}")

# 注意力分析 {#注意力分析}
attention_analyzer = AttentionAnalysisMetrics()
entropy_results = attention_analyzer.attention_entropy(attention_weights)
print(f"注意力熵: {entropy_results['mean_entropy']:.4f}")
```

### 🔧 批量评估 {#批量评估}

```python
# 批量评估多个模型 {#批量评估多个模型}
models = {'model_a': model_a, 'model_b': model_b, 'model_c': model_c}
results_comparison = {}

for name, model in models.items():
    print(f"评估模型: {name}")
    results = evaluator.evaluate_model(model, test_dataloader)
    results_comparison[name] = results
    
    # 生成单独报告
    evaluator.generate_report(f'{name}_report.md')

# 比较结果 {#比较结果}
print("\n模型比较:")
for name, results in results_comparison.items():
    mse = results['performance'].get('mse', 'N/A')
    throughput = results['efficiency'].get('throughput', 'N/A')
    params = results['efficiency'].get('total_parameters', 'N/A')
    
    print(f"{name}: MSE={mse:.6f}, 吞吐量={throughput:.2f}, 参数量={params:,}")
```

---

**💡 提示**: 评估指标的选择应该根据具体任务和应用场景来确定。建议使用多个指标进行综合评估，以获得模型性能的全面了解。定期评估和监控模型性能有助于及时发现问题和优化方向！

---

*需要帮助？查看 [FAQ](faq) 或 [故障排除](troubleshooting) 页面。*
