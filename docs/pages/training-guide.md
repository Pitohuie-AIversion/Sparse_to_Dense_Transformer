---
layout: default
title: Training Guide
parent: Training & Optimization
nav_order: 1
description: "完整的模型训练指南"
permalink: /pages/training-guide/
---

# 训练指南 {#训练指南}

本文档提供VIVTransformer项目的详细训练指南，包括训练流程、参数调优、监控和故障排除。

## 📋 目录 {#目录}

- [训练概览](#训练概览)
- [训练准备](#训练准备)
- [基础训练](#基础训练)
- [高级训练技巧](#高级训练技巧)
- [参数调优](#参数调优)
- [训练监控](#训练监控)
- [故障排除](#故障排除)
- [最佳实践](#最佳实践)

## 训练概览 {#训练概览}

### 🎯 训练目标 {#训练目标}

VIVTransformer的训练目标是学习有效的注意力机制，以实现：
- **高精度预测**: 在目标任务上达到最佳性能
- **计算效率**: 平衡精度和计算成本
- **泛化能力**: 在不同数据分布上保持稳定性能
- **可解释性**: 提供可理解的注意力模式

### 🏗️ 训练架构 {#训练架构}

```
训练流程
├── 📊 数据准备
│   ├── 数据加载
│   ├── 预处理
│   └── 数据增强
├── 🧠 模型初始化
│   ├── 权重初始化
│   ├── 注意力机制配置
│   └── 损失函数设置
├── 🔄 训练循环
│   ├── 前向传播
│   ├── 损失计算
│   ├── 反向传播
│   └── 参数更新
├── 📈 验证评估
│   ├── 性能指标
│   ├── 注意力可视化
│   └── 模型保存
└── 🎯 测试部署
    ├── 最终评估
    ├── 模型导出
    └── 性能分析
```

## 训练准备 {#训练准备}

### 🔧 环境配置 {#环境配置}

```bash
# 1. 检查CUDA环境 {#1-检查cuda环境}
nvidia-smi
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# 2. 设置环境变量 {#2-设置环境变量}
export CUDA_VISIBLE_DEVICES=0,1  # 指定GPU
export VIV_ENV=training          # 设置训练环境
export PYTHONPATH=$PYTHONPATH:$(pwd)  # 添加项目路径

# 3. 创建必要目录 {#3-创建必要目录}
mkdir -p logs checkpoints results
```

### 📊 数据准备 {#数据准备}

```python
# data_preparation.py {#data-preparation-py}
import torch
import numpy as np
from torch.utils.data import DataLoader, random_split
from utils.data_utils import VIVDataset, collate_fn

def prepare_data(config):
    """准备训练数据"""
    
    # 1. 加载数据集
    print("📊 加载数据集...")
    dataset = VIVDataset(
        data_path=config['data']['path'],
        transform=get_transforms(config),
        cache_dir=config['data'].get('cache_dir')
    )
    
    # 2. 数据分割
    train_size = int(config['data']['train_ratio'] * len(dataset))
    valid_size = int(config['data']['valid_ratio'] * len(dataset))
    test_size = len(dataset) - train_size - valid_size
    
    train_dataset, valid_dataset, test_dataset = random_split(
        dataset, [train_size, valid_size, test_size],
        generator=torch.Generator().manual_seed(config['global']['seed'])
    )
    
    # 3. 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=True,
        num_workers=config['data']['num_workers'],
        pin_memory=config['data']['pin_memory'],
        collate_fn=collate_fn
    )
    
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=config['data']['pin_memory'],
        collate_fn=collate_fn
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=config['data']['pin_memory'],
        collate_fn=collate_fn
    )
    
    print(f"✅ 数据准备完成:")
    print(f"   训练集: {len(train_dataset)} 样本")
    print(f"   验证集: {len(valid_dataset)} 样本")
    print(f"   测试集: {len(test_dataset)} 样本")
    
    return train_loader, valid_loader, test_loader

def get_transforms(config):
    """获取数据变换"""
    transforms = []
    
    if config['data'].get('normalize', True):
        transforms.append(Normalize())
    
    if config['data'].get('use_augmentation', False):
        transforms.append(RandomAugmentation())
    
    return Compose(transforms)
```

### 🧠 模型初始化 {#模型初始化}

```python
# model_initialization.py {#model-initialization-py}
import torch
import torch.nn as nn
from models.viv_transformer import VIVTransformer
from utils.model_utils import count_parameters, initialize_weights

def initialize_model(config):
    """初始化模型"""
    
    print("🧠 初始化模型...")
    
    # 1. 创建模型
    model = VIVTransformer(
        attention_type=config['model']['attention_type'],
        d_model=config['model']['d_model'],
        num_heads=config['model']['num_heads'],
        num_layers=config['model']['num_layers'],
        input_dim=config['model']['input_dim'],
        output_dim=config['model']['output_dim'],
        dropout=config['model']['dropout']
    )
    
    # 2. 权重初始化
    init_method = config['model'].get('init_method', 'xavier_uniform')
    initialize_weights(model, method=init_method)
    
    # 3. 移动到设备
    device = torch.device(config['global']['device'])
    model = model.to(device)
    
    # 4. 打印模型信息
    total_params = count_parameters(model)
    trainable_params = count_parameters(model, trainable_only=True)
    
    print(f"✅ 模型初始化完成:")
    print(f"   注意力类型: {config['model']['attention_type']}")
    print(f"   模型维度: {config['model']['d_model']}")
    print(f"   总参数量: {total_params:,}")
    print(f"   可训练参数: {trainable_params:,}")
    print(f"   设备: {device}")
    
    return model

def setup_optimizer(model, config):
    """设置优化器"""
    
    optimizer_type = config['training']['optimizer'].lower()
    lr = config['training']['learning_rate']
    weight_decay = config['training']['weight_decay']
    
    if optimizer_type == 'adamw':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=(config['training']['beta1'], config['training']['beta2']),
            eps=config['training']['eps']
        )
    elif optimizer_type == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
    elif optimizer_type == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            momentum=0.9
        )
    else:
        raise ValueError(f"不支持的优化器类型: {optimizer_type}")
    
    print(f"✅ 优化器设置完成: {optimizer_type.upper()}")
    return optimizer

def setup_scheduler(optimizer, config):
    """设置学习率调度器"""
    
    scheduler_type = config['training'].get('lr_scheduler', 'cosine')
    
    if scheduler_type == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config['training']['epochs'],
            eta_min=config['training'].get('min_lr', 1e-6)
        )
    elif scheduler_type == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config['training'].get('step_size', 10),
            gamma=config['training'].get('gamma', 0.1)
        )
    elif scheduler_type == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
    else:
        scheduler = None
    
    if scheduler:
        print(f"✅ 学习率调度器设置完成: {scheduler_type}")
    
    return scheduler
```

## 基础训练 {#基础训练}

### 🔄 训练循环 {#训练循环}

```python
# training_loop.py {#training-loop-py}
import torch
import torch.nn as nn
from tqdm import tqdm
from utils.metrics import calculate_metrics
from utils.loss_utils import get_loss_function
from utils.visualization import visualize_attention

class Trainer:
    """训练器类"""
    
    def __init__(self, model, train_loader, valid_loader, optimizer, 
                 scheduler, config):
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        
        # 设置损失函数
        self.criterion = get_loss_function(config['loss'])
        
        # 设置设备
        self.device = torch.device(config['global']['device'])
        
        # 训练状态
        self.current_epoch = 0
        self.best_loss = float('inf')
        self.patience_counter = 0
        
        # 历史记录
        self.train_history = []
        self.valid_history = []
    
    def train(self):
        """开始训练"""
        
        print("🚀 开始训练...")
        
        for epoch in range(self.config['training']['epochs']):
            self.current_epoch = epoch
            
            # 训练一个epoch
            train_metrics = self.train_epoch()
            
            # 验证
            if epoch % self.config['evaluation']['eval_frequency'] == 0:
                valid_metrics = self.validate()
                
                # 记录历史
                self.train_history.append(train_metrics)
                self.valid_history.append(valid_metrics)
                
                # 打印进度
                self.print_progress(train_metrics, valid_metrics)
                
                # 保存最佳模型
                if valid_metrics['loss'] < self.best_loss:
                    self.best_loss = valid_metrics['loss']
                    self.save_checkpoint('best_model.pth')
                    self.patience_counter = 0
                else:
                    self.patience_counter += 1
                
                # 早停检查
                if self.should_early_stop():
                    print("🛑 触发早停条件，停止训练")
                    break
            
            # 更新学习率
            if self.scheduler:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(valid_metrics['loss'])
                else:
                    self.scheduler.step()
            
            # 定期保存检查点
            if epoch % self.config['evaluation']['save_frequency'] == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch}.pth')
        
        print("✅ 训练完成")
        return self.train_history, self.valid_history
    
    def train_epoch(self):
        """训练一个epoch"""
        
        self.model.train()
        total_loss = 0
        total_samples = 0
        
        progress_bar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch+1}')
        
        for batch_idx, (data, target) in enumerate(progress_bar):
            # 移动数据到设备
            data = data.to(self.device)
            target = target.to(self.device)
            
            # 前向传播
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            if self.config['training'].get('gradient_clip_norm'):
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['training']['gradient_clip_norm']
                )
            
            # 参数更新
            self.optimizer.step()
            
            # 统计
            total_loss += loss.item() * data.size(0)
            total_samples += data.size(0)
            
            # 更新进度条
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Avg Loss': f'{total_loss/total_samples:.4f}'
            })
        
        avg_loss = total_loss / total_samples
        return {'loss': avg_loss}
    
    def validate(self):
        """验证模型"""
        
        self.model.eval()
        total_loss = 0
        total_samples = 0
        all_outputs = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in tqdm(self.valid_loader, desc='Validation'):
                data = data.to(self.device)
                target = target.to(self.device)
                
                output = self.model(data)
                loss = self.criterion(output, target)
                
                total_loss += loss.item() * data.size(0)
                total_samples += data.size(0)
                
                all_outputs.append(output.cpu())
                all_targets.append(target.cpu())
        
        # 计算指标
        all_outputs = torch.cat(all_outputs, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        metrics = calculate_metrics(all_outputs, all_targets)
        metrics['loss'] = total_loss / total_samples
        
        return metrics
    
    def should_early_stop(self):
        """检查是否应该早停"""
        if not self.config['training'].get('early_stop', False):
            return False
        
        patience = self.config['training'].get('early_stop_patience', 10)
        return self.patience_counter >= patience
    
    def print_progress(self, train_metrics, valid_metrics):
        """打印训练进度"""
        print(f"\nEpoch {self.current_epoch+1}/{self.config['training']['epochs']}:")
        print(f"  训练损失: {train_metrics['loss']:.6f}")
        print(f"  验证损失: {valid_metrics['loss']:.6f}")
        
        if 'mse' in valid_metrics:
            print(f"  验证MSE: {valid_metrics['mse']:.6f}")
        if 'mae' in valid_metrics:
            print(f"  验证MAE: {valid_metrics['mae']:.6f}")
        if 'r2' in valid_metrics:
            print(f"  验证R²: {valid_metrics['r2']:.6f}")
        
        # 打印学习率
        current_lr = self.optimizer.param_groups[0]['lr']
        print(f"  学习率: {current_lr:.2e}")
    
    def save_checkpoint(self, filename):
        """保存检查点"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_loss': self.best_loss,
            'config': self.config,
            'train_history': self.train_history,
            'valid_history': self.valid_history
        }
        
        torch.save(checkpoint, f'checkpoints/{filename}')
        print(f"💾 保存检查点: {filename}")
```

### 🎯 简单训练脚本 {#简单训练脚本}

```python
# simple_train.py {#simple-train-py}
import yaml
import torch
from training_loop import Trainer
from data_preparation import prepare_data
from model_initialization import initialize_model, setup_optimizer, setup_scheduler

def main():
    """主训练函数"""
    
    # 1. 加载配置
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # 2. 设置随机种子
    torch.manual_seed(config['global']['seed'])
    
    # 3. 准备数据
    train_loader, valid_loader, test_loader = prepare_data(config)
    
    # 4. 初始化模型
    model = initialize_model(config)
    
    # 5. 设置优化器和调度器
    optimizer = setup_optimizer(model, config)
    scheduler = setup_scheduler(optimizer, config)
    
    # 6. 创建训练器
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config
    )
    
    # 7. 开始训练
    train_history, valid_history = trainer.train()
    
    # 8. 保存训练历史
    torch.save({
        'train_history': train_history,
        'valid_history': valid_history,
        'config': config
    }, 'results/training_history.pth')
    
    print("🎉 训练完成！")

if __name__ == '__main__':
    main()
```

## 高级训练技巧 {#高级训练技巧}

### 🔥 混合精度训练 {#混合精度训练}

```python
# mixed_precision_training.py {#mixed-precision-training-py}
import torch
from torch.cuda.amp import GradScaler, autocast

class MixedPrecisionTrainer(Trainer):
    """混合精度训练器"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # 初始化梯度缩放器
        if self.config['global'].get('mixed_precision', False):
            self.scaler = GradScaler()
            self.use_amp = True
            print("🔥 启用混合精度训练")
        else:
            self.scaler = None
            self.use_amp = False
    
    def train_epoch(self):
        """混合精度训练epoch"""
        
        self.model.train()
        total_loss = 0
        total_samples = 0
        
        progress_bar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch+1}')
        
        for batch_idx, (data, target) in enumerate(progress_bar):
            data = data.to(self.device)
            target = target.to(self.device)
            
            self.optimizer.zero_grad()
            
            if self.use_amp:
                # 使用自动混合精度
                with autocast():
                    output = self.model(data)
                    loss = self.criterion(output, target)
                
                # 缩放损失并反向传播
                self.scaler.scale(loss).backward()
                
                # 梯度裁剪
                if self.config['training'].get('gradient_clip_norm'):
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config['training']['gradient_clip_norm']
                    )
                
                # 更新参数
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # 标准精度训练
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                
                if self.config['training'].get('gradient_clip_norm'):
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config['training']['gradient_clip_norm']
                    )
                
                self.optimizer.step()
            
            total_loss += loss.item() * data.size(0)
            total_samples += data.size(0)
            
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Avg Loss': f'{total_loss/total_samples:.4f}'
            })
        
        avg_loss = total_loss / total_samples
        return {'loss': avg_loss}
```

### 🔄 梯度累积 {#梯度累积}

```python
# gradient_accumulation.py {#gradient-accumulation-py}
class GradientAccumulationTrainer(Trainer):
    """梯度累积训练器"""
    
    def train_epoch(self):
        """带梯度累积的训练epoch"""
        
        self.model.train()
        total_loss = 0
        total_samples = 0
        
        accumulation_steps = self.config['training'].get('gradient_accumulation_steps', 1)
        
        progress_bar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch+1}')
        
        for batch_idx, (data, target) in enumerate(progress_bar):
            data = data.to(self.device)
            target = target.to(self.device)
            
            # 前向传播
            output = self.model(data)
            loss = self.criterion(output, target)
            
            # 缩放损失
            loss = loss / accumulation_steps
            
            # 反向传播
            loss.backward()
            
            # 累积梯度
            if (batch_idx + 1) % accumulation_steps == 0:
                # 梯度裁剪
                if self.config['training'].get('gradient_clip_norm'):
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config['training']['gradient_clip_norm']
                    )
                
                # 更新参数
                self.optimizer.step()
                self.optimizer.zero_grad()
            
            total_loss += loss.item() * data.size(0) * accumulation_steps
            total_samples += data.size(0)
            
            progress_bar.set_postfix({
                'Loss': f'{loss.item() * accumulation_steps:.4f}',
                'Avg Loss': f'{total_loss/total_samples:.4f}'
            })
        
        avg_loss = total_loss / total_samples
        return {'loss': avg_loss}
```

### 📊 多GPU训练 {#多gpu训练}

```python
# multi_gpu_training.py {#multi-gpu-training-py}
import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel, DistributedDataParallel

def setup_multi_gpu(model, config):
    """设置多GPU训练"""
    
    if torch.cuda.device_count() > 1:
        print(f"🚀 使用 {torch.cuda.device_count()} 个GPU进行训练")
        
        # 数据并行
        if config['training'].get('distributed', False):
            # 分布式训练
            model = DistributedDataParallel(model)
        else:
            # 数据并行
            model = DataParallel(model)
    
    return model

# 分布式训练初始化 {#分布式训练初始化}
def init_distributed_training():
    """初始化分布式训练"""
    import torch.distributed as dist
    import os
    
    # 初始化进程组
    dist.init_process_group(backend='nccl')
    
    # 设置本地rank
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    
    return local_rank
```

## 参数调优 {#参数调优}

### 🎛️ 超参数搜索 {#超参数搜索}

```python
# hyperparameter_search.py {#hyperparameter-search-py}
import itertools
import numpy as np
from typing import Dict, List, Any

class HyperparameterSearcher:
    """超参数搜索器"""
    
    def __init__(self, base_config: Dict[str, Any]):
        self.base_config = base_config
        self.search_space = self._define_search_space()
        self.results = []
    
    def _define_search_space(self) -> Dict[str, List]:
        """定义搜索空间"""
        return {
            'learning_rate': [1e-5, 5e-5, 1e-4, 5e-4, 1e-3],
            'batch_size': [32, 64, 128, 256],
            'dropout': [0.0, 0.1, 0.2, 0.3],
            'd_model': [128, 256, 512],
            'num_heads': [4, 8, 16],
            'weight_decay': [0.0, 0.01, 0.1]
        }
    
    def grid_search(self, max_trials: int = 50):
        """网格搜索"""
        print(f"🔍 开始网格搜索，最大试验次数: {max_trials}")
        
        # 生成所有参数组合
        param_names = list(self.search_space.keys())
        param_values = list(self.search_space.values())
        
        all_combinations = list(itertools.product(*param_values))
        
        # 随机选择组合（如果组合太多）
        if len(all_combinations) > max_trials:
            selected_combinations = np.random.choice(
                len(all_combinations), max_trials, replace=False
            )
            combinations = [all_combinations[i] for i in selected_combinations]
        else:
            combinations = all_combinations
        
        # 执行搜索
        for i, combination in enumerate(combinations):
            print(f"\n🧪 试验 {i+1}/{len(combinations)}")
            
            # 创建配置
            config = self.base_config.copy()
            for param_name, param_value in zip(param_names, combination):
                self._update_config(config, param_name, param_value)
            
            # 训练模型
            result = self._train_with_config(config)
            result['params'] = dict(zip(param_names, combination))
            result['trial_id'] = i
            
            self.results.append(result)
            
            print(f"   结果: {result['best_loss']:.6f}")
        
        # 找到最佳配置
        best_result = min(self.results, key=lambda x: x['best_loss'])
        print(f"\n🏆 最佳配置:")
        for param, value in best_result['params'].items():
            print(f"   {param}: {value}")
        print(f"   最佳损失: {best_result['best_loss']:.6f}")
        
        return best_result
    
    def random_search(self, max_trials: int = 50):
        """随机搜索"""
        print(f"🎲 开始随机搜索，试验次数: {max_trials}")
        
        for i in range(max_trials):
            print(f"\n🧪 试验 {i+1}/{max_trials}")
            
            # 随机采样参数
            config = self.base_config.copy()
            sampled_params = {}
            
            for param_name, param_values in self.search_space.items():
                sampled_value = np.random.choice(param_values)
                sampled_params[param_name] = sampled_value
                self._update_config(config, param_name, sampled_value)
            
            # 训练模型
            result = self._train_with_config(config)
            result['params'] = sampled_params
            result['trial_id'] = i
            
            self.results.append(result)
            
            print(f"   参数: {sampled_params}")
            print(f"   结果: {result['best_loss']:.6f}")
        
        # 找到最佳配置
        best_result = min(self.results, key=lambda x: x['best_loss'])
        print(f"\n🏆 最佳配置:")
        for param, value in best_result['params'].items():
            print(f"   {param}: {value}")
        print(f"   最佳损失: {best_result['best_loss']:.6f}")
        
        return best_result
    
    def _update_config(self, config: Dict[str, Any], param_name: str, param_value):
        """更新配置参数"""
        if param_name == 'learning_rate':
            config['training']['learning_rate'] = param_value
        elif param_name == 'batch_size':
            config['data']['batch_size'] = param_value
        elif param_name == 'dropout':
            config['model']['dropout'] = param_value
        elif param_name == 'd_model':
            config['model']['d_model'] = param_value
        elif param_name == 'num_heads':
            config['model']['num_heads'] = param_value
        elif param_name == 'weight_decay':
            config['training']['weight_decay'] = param_value
    
    def _train_with_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """使用指定配置训练模型"""
        # 这里应该调用实际的训练函数
        # 为了示例，我们返回模拟结果
        
        # 实际实现中，这里应该是:
        # trainer = create_trainer(config)
        # history = trainer.train()
        # return {'best_loss': min([h['loss'] for h in history])}
        
        # 模拟训练结果
        import time
        time.sleep(1)  # 模拟训练时间
        
        # 模拟损失值（实际中应该是真实的训练结果）
        simulated_loss = np.random.uniform(0.1, 1.0)
        
        return {
            'best_loss': simulated_loss,
            'final_loss': simulated_loss + np.random.uniform(-0.1, 0.1),
            'epochs_trained': config['training']['epochs']
        }
```

### 📈 学习率调优 {#学习率调优}

```python
# learning_rate_finder.py {#learning-rate-finder-py}
import torch
import matplotlib.pyplot as plt
import numpy as np

class LearningRateFinder:
    """学习率查找器"""
    
    def __init__(self, model, optimizer, criterion, device):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        
        # 保存初始状态
        self.initial_state = model.state_dict()
        self.initial_optimizer_state = optimizer.state_dict()
    
    def find_lr(self, train_loader, start_lr=1e-7, end_lr=10, num_iter=100):
        """查找最优学习率"""
        
        print(f"🔍 查找学习率范围: {start_lr:.2e} - {end_lr:.2e}")
        
        # 生成学习率序列
        lr_schedule = np.logspace(np.log10(start_lr), np.log10(end_lr), num_iter)
        
        losses = []
        lrs = []
        
        # 设置模型为训练模式
        self.model.train()
        
        data_iter = iter(train_loader)
        
        for i, lr in enumerate(lr_schedule):
            # 设置学习率
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            
            try:
                # 获取下一批数据
                data, target = next(data_iter)
            except StopIteration:
                # 重新开始数据迭代器
                data_iter = iter(train_loader)
                data, target = next(data_iter)
            
            data = data.to(self.device)
            target = target.to(self.device)
            
            # 前向传播
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            
            # 检查损失是否爆炸
            if i > 0 and loss.item() > 4 * min(losses):
                print(f"⚠️  损失爆炸，停止搜索 (lr={lr:.2e})")
                break
            
            # 反向传播
            loss.backward()
            self.optimizer.step()
            
            # 记录
            losses.append(loss.item())
            lrs.append(lr)
            
            if i % 10 == 0:
                print(f"   步骤 {i}/{num_iter}, lr={lr:.2e}, loss={loss.item():.6f}")
        
        # 恢复初始状态
        self.model.load_state_dict(self.initial_state)
        self.optimizer.load_state_dict(self.initial_optimizer_state)
        
        # 找到最优学习率
        optimal_lr = self._find_optimal_lr(lrs, losses)
        
        # 绘制结果
        self._plot_lr_loss(lrs, losses, optimal_lr)
        
        return optimal_lr, lrs, losses
    
    def _find_optimal_lr(self, lrs, losses):
        """找到最优学习率"""
        # 计算损失的梯度
        gradients = np.gradient(losses)
        
        # 找到梯度最小的点（损失下降最快的点）
        min_gradient_idx = np.argmin(gradients)
        
        # 取梯度最小点前的一个数量级作为最优学习率
        optimal_lr = lrs[min_gradient_idx] / 10
        
        print(f"🎯 建议学习率: {optimal_lr:.2e}")
        
        return optimal_lr
    
    def _plot_lr_loss(self, lrs, losses, optimal_lr):
        """绘制学习率-损失曲线"""
        plt.figure(figsize=(10, 6))
        plt.semilogx(lrs, losses)
        plt.axvline(x=optimal_lr, color='red', linestyle='--', 
                   label=f'建议学习率: {optimal_lr:.2e}')
        plt.xlabel('学习率')
        plt.ylabel('损失')
        plt.title('学习率查找器')
        plt.legend()
        plt.grid(True)
        plt.savefig('lr_finder_result.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("📊 学习率-损失曲线已保存为 lr_finder_result.png")
```

## 训练监控 {#训练监控}

### 📊 TensorBoard集成 {#tensorboard集成}

```python
# tensorboard_logger.py {#tensorboard-logger-py}
from torch.utils.tensorboard import SummaryWriter
import torch
import matplotlib.pyplot as plt
import numpy as np

class TensorBoardLogger:
    """TensorBoard日志记录器"""
    
    def __init__(self, log_dir, config):
        self.writer = SummaryWriter(log_dir)
        self.config = config
        self.step = 0
    
    def log_scalar(self, tag, value, step=None):
        """记录标量值"""
        if step is None:
            step = self.step
        self.writer.add_scalar(tag, value, step)
    
    def log_scalars(self, tag, value_dict, step=None):
        """记录多个标量值"""
        if step is None:
            step = self.step
        self.writer.add_scalars(tag, value_dict, step)
    
    def log_histogram(self, tag, values, step=None):
        """记录直方图"""
        if step is None:
            step = self.step
        self.writer.add_histogram(tag, values, step)
    
    def log_model_graph(self, model, input_sample):
        """记录模型图"""
        self.writer.add_graph(model, input_sample)
    
    def log_attention_weights(self, attention_weights, step=None):
        """记录注意力权重"""
        if step is None:
            step = self.step
        
        # 可视化注意力权重
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(attention_weights.detach().cpu().numpy(), cmap='Blues')
        ax.set_title('Attention Weights')
        plt.colorbar(im)
        
        self.writer.add_figure('Attention/Weights', fig, step)
        plt.close(fig)
    
    def log_training_metrics(self, metrics, step=None):
        """记录训练指标"""
        if step is None:
            step = self.step
        
        for metric_name, metric_value in metrics.items():
            self.log_scalar(f'Training/{metric_name}', metric_value, step)
    
    def log_validation_metrics(self, metrics, step=None):
        """记录验证指标"""
        if step is None:
            step = self.step
        
        for metric_name, metric_value in metrics.items():
            self.log_scalar(f'Validation/{metric_name}', metric_value, step)
    
    def log_learning_rate(self, lr, step=None):
        """记录学习率"""
        if step is None:
            step = self.step
        self.log_scalar('Training/LearningRate', lr, step)
    
    def log_gradients(self, model, step=None):
        """记录梯度信息"""
        if step is None:
            step = self.step
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                self.log_histogram(f'Gradients/{name}', param.grad, step)
                self.log_scalar(f'Gradients/{name}_norm', 
                              param.grad.norm().item(), step)
    
    def log_weights(self, model, step=None):
        """记录权重信息"""
        if step is None:
            step = self.step
        
        for name, param in model.named_parameters():
            self.log_histogram(f'Weights/{name}', param, step)
            self.log_scalar(f'Weights/{name}_norm', param.norm().item(), step)
    
    def increment_step(self):
        """增加步数"""
        self.step += 1
    
    def close(self):
        """关闭writer"""
        self.writer.close()
```

### 📈 实时监控 {#实时监控}

```python
# real_time_monitor.py {#real-time-monitor-py}
import time
import psutil
import GPUtil
from threading import Thread
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class RealTimeMonitor:
    """实时监控器"""
    
    def __init__(self, update_interval=5):
        self.update_interval = update_interval
        self.monitoring = False
        
        # 监控数据
        self.timestamps = []
        self.cpu_usage = []
        self.memory_usage = []
        self.gpu_usage = []
        self.gpu_memory = []
        
        # 图形界面
        self.fig, self.axes = plt.subplots(2, 2, figsize=(12, 8))
        self.fig.suptitle('实时系统监控')
    
    def start_monitoring(self):
        """开始监控"""
        self.monitoring = True
        
        # 启动监控线程
        monitor_thread = Thread(target=self._monitor_loop)
        monitor_thread.daemon = True
        monitor_thread.start()
        
        # 启动可视化
        self.ani = FuncAnimation(self.fig, self._update_plots, 
                                interval=self.update_interval*1000)
        plt.show()
    
    def stop_monitoring(self):
        """停止监控"""
        self.monitoring = False
    
    def _monitor_loop(self):
        """监控循环"""
        while self.monitoring:
            # 获取系统信息
            timestamp = time.time()
            cpu_percent = psutil.cpu_percent()
            memory_percent = psutil.virtual_memory().percent
            
            # 获取GPU信息
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu_percent = gpus[0].load * 100
                    gpu_mem_percent = gpus[0].memoryUtil * 100
                else:
                    gpu_percent = 0
                    gpu_mem_percent = 0
            except:
                gpu_percent = 0
                gpu_mem_percent = 0
            
            # 记录数据
            self.timestamps.append(timestamp)
            self.cpu_usage.append(cpu_percent)
            self.memory_usage.append(memory_percent)
            self.gpu_usage.append(gpu_percent)
            self.gpu_memory.append(gpu_mem_percent)
            
            # 保持最近100个数据点
            if len(self.timestamps) > 100:
                self.timestamps.pop(0)
                self.cpu_usage.pop(0)
                self.memory_usage.pop(0)
                self.gpu_usage.pop(0)
                self.gpu_memory.pop(0)
            
            time.sleep(self.update_interval)
    
    def _update_plots(self, frame):
        """更新图表"""
        if not self.timestamps:
            return
        
        # 清除所有子图
        for ax in self.axes.flat:
            ax.clear()
        
        # 转换时间戳为相对时间
        if self.timestamps:
            relative_times = [(t - self.timestamps[0]) / 60 for t in self.timestamps]
        else:
            relative_times = []
        
        # CPU使用率
        self.axes[0, 0].plot(relative_times, self.cpu_usage, 'b-')
        self.axes[0, 0].set_title('CPU使用率 (%)')
        self.axes[0, 0].set_ylim(0, 100)
        self.axes[0, 0].grid(True)
        
        # 内存使用率
        self.axes[0, 1].plot(relative_times, self.memory_usage, 'g-')
        self.axes[0, 1].set_title('内存使用率 (%)')
        self.axes[0, 1].set_ylim(0, 100)
        self.axes[0, 1].grid(True)
        
        # GPU使用率
        self.axes[1, 0].plot(relative_times, self.gpu_usage, 'r-')
        self.axes[1, 0].set_title('GPU使用率 (%)')
        self.axes[1, 0].set_ylim(0, 100)
        self.axes[1, 0].grid(True)
        
        # GPU内存使用率
        self.axes[1, 1].plot(relative_times, self.gpu_memory, 'm-')
        self.axes[1, 1].set_title('GPU内存使用率 (%)')
        self.axes[1, 1].set_ylim(0, 100)
        self.axes[1, 1].grid(True)
        
        # 设置x轴标签
        for ax in self.axes[1, :]:
            ax.set_xlabel('时间 (分钟)')
        
        plt.tight_layout()
```

## 故障排除 {#故障排除}

### 🔧 常见问题解决 {#常见问题解决}

```python
# troubleshooting.py {#troubleshooting-py}
import torch
import traceback
from typing import Dict, Any

class TrainingTroubleshooter:
    """训练故障排除器"""
    
    def __init__(self):
        self.common_issues = {
            'cuda_out_of_memory': self._handle_cuda_oom,
            'nan_loss': self._handle_nan_loss,
            'slow_training': self._handle_slow_training,
            'poor_convergence': self._handle_poor_convergence,
            'gradient_explosion': self._handle_gradient_explosion
        }
    
    def diagnose_and_fix(self, error_type: str, context: Dict[str, Any]):
        """诊断并修复问题"""
        
        if error_type in self.common_issues:
            return self.common_issues[error_type](context)
        else:
            return self._generic_troubleshooting(error_type, context)
    
    def _handle_cuda_oom(self, context):
        """处理CUDA内存不足"""
        suggestions = [
            "🔧 减小批大小 (batch_size)",
            "🔧 减小模型维度 (d_model)",
            "🔧 启用梯度累积",
            "🔧 使用混合精度训练",
            "🔧 清理GPU缓存: torch.cuda.empty_cache()",
            "🔧 减少数据加载器的num_workers"
        ]
        
        # 自动修复建议
        auto_fixes = {
            'batch_size': max(context.get('batch_size', 32) // 2, 1),
            'd_model': max(context.get('d_model', 256) // 2, 64),
            'gradient_accumulation_steps': 2,
            'mixed_precision': True
        }
        
        return {
            'suggestions': suggestions,
            'auto_fixes': auto_fixes,
            'severity': 'high'
        }
    
    def _handle_nan_loss(self, context):
        """处理NaN损失"""
        suggestions = [
            "🔧 降低学习率",
            "🔧 启用梯度裁剪",
            "🔧 检查数据中的NaN值",
            "🔧 使用更稳定的损失函数",
            "🔧 添加权重初始化",
            "🔧 增加批归一化或层归一化"
        ]
        
        auto_fixes = {
            'learning_rate': context.get('learning_rate', 0.001) * 0.1,
            'gradient_clip_norm': 1.0,
            'weight_decay': 0.01
        }
        
        return {
            'suggestions': suggestions,
            'auto_fixes': auto_fixes,
            'severity': 'critical'
        }
    
    def _handle_slow_training(self, context):
        """处理训练缓慢"""
        suggestions = [
            "🔧 增加批大小",
            "🔧 使用更多GPU",
            "🔧 启用混合精度训练",
            "🔧 优化数据加载 (增加num_workers)",
            "🔧 使用更高效的注意力机制",
            "🔧 启用数据预取 (pin_memory=True)"
        ]
        
        auto_fixes = {
            'batch_size': min(context.get('batch_size', 32) * 2, 512),
            'num_workers': min(context.get('num_workers', 4) * 2, 8),
            'pin_memory': True,
            'mixed_precision': True
        }
        
        return {
            'suggestions': suggestions,
            'auto_fixes': auto_fixes,
            'severity': 'medium'
        }
    
    def _handle_poor_convergence(self, context):
        """处理收敛性差"""
        suggestions = [
            "🔧 调整学习率",
            "🔧 使用学习率调度器",
            "🔧 增加训练轮数",
            "🔧 调整模型架构",
            "🔧 使用不同的优化器",
            "🔧 添加正则化"
        ]
        
        auto_fixes = {
            'lr_scheduler': 'cosine',
            'weight_decay': 0.01,
            'epochs': context.get('epochs', 10) * 2
        }
        
        return {
            'suggestions': suggestions,
            'auto_fixes': auto_fixes,
            'severity': 'medium'
        }
    
    def _handle_gradient_explosion(self, context):
        """处理梯度爆炸"""
        suggestions = [
            "🔧 启用梯度裁剪",
            "🔧 降低学习率",
            "🔧 使用更小的权重初始化",
            "🔧 添加批归一化",
            "🔧 检查损失函数设计"
        ]
        
        auto_fixes = {
            'gradient_clip_norm': 0.5,
            'learning_rate': context.get('learning_rate', 0.001) * 0.1
        }
        
        return {
            'suggestions': suggestions,
            'auto_fixes': auto_fixes,
            'severity': 'high'
        }
    
    def _generic_troubleshooting(self, error_type, context):
        """通用故障排除"""
        return {
            'suggestions': [
                "🔧 检查错误日志",
                "🔧 验证数据格式",
                "🔧 检查配置文件",
                "🔧 重启训练进程"
            ],
            'auto_fixes': {},
            'severity': 'unknown'
        }

# 自动故障检测 {#自动故障检测}
class AutoTroubleshooter:
    """自动故障检测器"""
    
    def __init__(self, trainer):
        self.trainer = trainer
        self.troubleshooter = TrainingTroubleshooter()
        self.monitoring = True
    
    def monitor_training(self):
        """监控训练过程"""
        while self.monitoring:
            try:
                # 检查各种问题
                self._check_memory_usage()
                self._check_loss_values()
                self._check_gradient_norms()
                self._check_training_speed()
                
                time.sleep(10)  # 每10秒检查一次
                
            except Exception as e:
                print(f"⚠️  监控过程中出现错误: {e}")
    
    def _check_memory_usage(self):
        """检查内存使用"""
        if torch.cuda.is_available():
            memory_used = torch.cuda.memory_allocated() / 1024**3  # GB
            memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            
            if memory_used / memory_total > 0.9:
                print("⚠️  GPU内存使用率过高，可能导致OOM")
    
    def _check_loss_values(self):
        """检查损失值"""
        if hasattr(self.trainer, 'train_history') and self.trainer.train_history:
            recent_losses = [h['loss'] for h in self.trainer.train_history[-5:]]
            
            # 检查NaN
            if any(np.isnan(loss) for loss in recent_losses):
                print("🚨 检测到NaN损失值！")
                fixes = self.troubleshooter.diagnose_and_fix('nan_loss', {})
                self._apply_fixes(fixes)
            
            # 检查损失爆炸
            if len(recent_losses) > 1:
                loss_ratio = recent_losses[-1] / recent_losses[0]
                if loss_ratio > 10:
                    print("🚨 检测到损失爆炸！")
    
    def _check_gradient_norms(self):
        """检查梯度范数"""
        total_norm = 0
        for p in self.trainer.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        
        if total_norm > 100:
            print(f"⚠️  梯度范数过大: {total_norm:.2f}")
    
    def _check_training_speed(self):
        """检查训练速度"""
        # 这里可以添加训练速度监控逻辑
        pass
    
    def _apply_fixes(self, fixes):
        """应用修复建议"""
        print("🔧 应用自动修复...")
        for fix_name, fix_value in fixes.get('auto_fixes', {}).items():
            print(f"   {fix_name}: {fix_value}")
            # 这里可以添加实际的修复逻辑
```

## 最佳实践 {#最佳实践}

### 📚 训练最佳实践 {#训练最佳实践}

1. **数据准备**
   - 确保数据质量和一致性
   - 使用适当的数据增强
   - 验证数据加载管道
   - 监控数据分布变化

2. **模型配置**
   - 从简单模型开始，逐步增加复杂度
   - 使用预训练权重（如果可用）
   - 合理设置模型维度和层数
   - 选择适合任务的注意力机制

3. **训练策略**
   - 使用学习率预热
   - 实施梯度裁剪
   - 监控训练和验证指标
   - 定期保存检查点

4. **调试技巧**
   - 从小数据集开始验证
   - 使用可视化工具监控训练
   - 记录详细的实验日志
   - 进行消融实验

5. **性能优化**
   - 使用混合精度训练
   - 优化数据加载流程
   - 合理使用多GPU
   - 监控系统资源使用

### 🎯 训练检查清单 {#训练检查清单}

**训练前检查**
- [ ] 数据路径正确且可访问
- [ ] 配置文件语法正确
- [ ] GPU内存充足
- [ ] 依赖包版本兼容
- [ ] 输出目录权限正确

**训练中监控**
- [ ] 损失值正常下降
- [ ] 梯度范数稳定
- [ ] 内存使用合理
- [ ] 训练速度符合预期
- [ ] 验证指标改善

**训练后验证**
- [ ] 模型文件完整保存
- [ ] 训练日志记录完整
- [ ] 最佳模型性能达标
- [ ] 可视化结果合理
- [ ] 实验可重现

---

**💡 提示**: 成功的训练需要耐心和系统性的方法。遇到问题时，先检查数据和配置，然后逐步调试模型和训练参数。记住，好的实验记录是成功的关键！

## 📚 相关文档

- [Loss Functions](loss-functions)
- [Hyperparameter Tuning](hyperparameter-tuning)
- [Convergence Analysis](convergence-analysis)


---

*需要帮助？查看 [FAQ](faq) 或 [故障排除](troubleshooting) 页面。*
