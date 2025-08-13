---
layout: default
title: Training Guide
parent: Training & Optimization
nav_order: 1
description: "Complete model training guide"
permalink: /pages/training-guide/
---

# Training Guide {#training-guide}

This document provides a detailed training guide for the VIVTransformer project, including training workflows, parameter tuning, monitoring, and troubleshooting.

## 📋 Table of Contents {#table-of-contents}

- [Training Overview](#training-overview)
- [Training Preparation](#training-preparation)
- [Basic Training](#basic-training)
- [Advanced Training Techniques](#advanced-training-techniques)
- [Parameter Tuning](#parameter-tuning)
- [Training Monitoring](#training-monitoring)
- [Troubleshooting](#troubleshooting)
- [Best Practices](#best-practices)

## Training Overview {#training-overview}

### 🎯 Training Objectives {#training-objectives}

The training objectives of VIVTransformer are to learn effective attention mechanisms to achieve:
- High-precision prediction: Achieve optimal performance on target tasks
- Computational efficiency: Balance accuracy and computational cost
- Generalization capability: Maintain stable performance across different data distributions
- Interpretability: Provide understandable attention patterns

### 🏗️ Training Architecture {#training-architecture}

```
Training Pipeline
├── 📊 Data Preparation
│   ├── Data Loading
│   ├── Preprocessing
│   └── Data Augmentation
├── 🧠 Model Initialization
│   ├── Weight Initialization
│   ├── Attention Mechanism Configuration
│   └── Loss Function Setup
├── 🔄 Training Loop
│   ├── Forward Pass
│   ├── Loss Computation
│   ├── Backward Pass
│   └── Parameter Update
├── 📈 Validation Evaluation
│   ├── Performance Metrics
│   ├── Attention Visualization
│   └── Model Saving
└── 🎯 Testing Deployment
    ├── Final Evaluation
    ├── Model Export
    └── Performance Analysis
```

## Training Preparation {#training-preparation}

### 🔧 Environment Configuration {#environment-configuration}

```bash
# 1. Check CUDA Environment
nvidia-smi
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# 2. Set Environment Variables
export CUDA_VISIBLE_DEVICES=0,1  # Specify GPU
export VIV_ENV=training          # Set training environment
export PYTHONPATH=$PYTHONPATH:$(pwd)  # Add project path

# 3. Create Necessary Directories
mkdir -p logs checkpoints results
```

### 📊 Data Preparation {#data-preparation}

```python
# data_preparation.py
import torch
import numpy as np
from torch.utils.data import DataLoader, random_split
from utils.data_utils import VIVDataset, collate_fn

def prepare_data(config):
    """Prepare training data"""
    
    # 1. Load dataset
    print("📊 Loading dataset...")
    dataset = VIVDataset(
        data_path=config['data']['path'],
        transform=get_transforms(config),
        cache_dir=config['data'].get('cache_dir')
    )
    
    # 2. Data splitting
    train_size = int(config['data']['train_ratio'] * len(dataset))
    valid_size = int(config['data']['valid_ratio'] * len(dataset))
    test_size = len(dataset) - train_size - valid_size
    
    train_dataset, valid_dataset, test_dataset = random_split(
        dataset, [train_size, valid_size, test_size]
    )
    
    # 3. Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['data']['num_workers'],
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=config['training']['eval_batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    return train_loader, valid_loader, test_dataset
```

## Basic Training {#basic-training}

### 🏁 Quick Start Training {#quick-start-training}

```python
# quick_start.py
import torch
from model.viv_transformer import VIVTransformer
from utils.trainer import Trainer
from utils.config import load_config

def quick_start():
    """Quick start training"""
    
    # 1. Load configuration
    config = load_config('configs/default.yaml')
    
    # 2. Initialize model
    model = VIVTransformer(config['model'])
    
    # 3. Prepare data
    train_loader, valid_loader, _ = prepare_data(config)
    
    # 4. Start training
    trainer = Trainer(model, config)
    trainer.train(train_loader, valid_loader)

if __name__ == "__main__":
    quick_start()
```

### 🎛️ Training Configuration {#training-configuration}

```yaml
# configs/training.yaml
training:
  # Basic Parameters
  epochs: 100
  batch_size: 32
  learning_rate: 0.001
  weight_decay: 0.0001
  
  # Optimization
  optimizer: "AdamW"
  scheduler: "CosineAnnealingLR"
  warmup_steps: 1000
  
  # Loss Function
  loss_function: "multi_loss"
  loss_weights:
    reconstruction: 1.0
    attention_sparsity: 0.1
    svd_consistency: 0.2
  
  # Training Strategy
  gradient_clipping: 1.0
  accumulation_steps: 1
  early_stopping_patience: 10
  
  # Checkpoint
  save_interval: 10
  checkpoint_dir: "checkpoints"
  save_top_k: 3
```

### 🔄 Training Loop {#training-loop}

```python
# trainer.py
class Trainer:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Setup optimizer and scheduler
        self.optimizer = self._setup_optimizer()
        self.scheduler = self._setup_scheduler()
        self.criterion = self._setup_criterion()
        
    def train_epoch(self, train_loader, epoch):
        """Train one epoch"""
        self.model.train()
        total_loss = 0
        num_batches = len(train_loader)
        
        for batch_idx, batch in enumerate(train_loader):
            # Move to device
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # Forward pass
            outputs = self.model(batch['input'])
            loss = self.criterion(outputs, batch['target'])
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                self.config['training']['gradient_clipping']
            )
            
            # Update parameters
            self.optimizer.step()
            self.optimizer.zero_grad()
            
            total_loss += loss.item()
            
            # Progress logging
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}/{num_batches}, Loss: {loss.item():.4f}")
        
        return total_loss / num_batches
```

## Advanced Training Techniques {#advanced-training-techniques}

### 🚀 Multi-GPU Training {#multi-gpu-training}

```python
# multi_gpu_training.py
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def setup_distributed():
    """Setup distributed training"""
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(rank)
    return rank

def train_distributed(model, train_loader, config):
    """Distributed training"""
    rank = setup_distributed()
    
    # Wrap model with DDP
    model = DDP(model, device_ids=[rank])
    
    # Training loop
    for epoch in range(config['training']['epochs']):
        train_loader.sampler.set_epoch(epoch)
        train_epoch(model, train_loader, epoch)
```

### 🔀 Mixed Precision Training {#mixed-precision-training}

```python
# mixed_precision.py
from torch.cuda.amp import GradScaler, autocast

class MixedPrecisionTrainer(Trainer):
    def __init__(self, model, config):
        super().__init__(model, config)
        self.scaler = GradScaler()
    
    def train_epoch(self, train_loader, epoch):
        """Mixed precision training epoch"""
        self.model.train()
        total_loss = 0
        
        for batch_idx, batch in enumerate(train_loader):
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # Use autocast for forward pass
            with autocast():
                outputs = self.model(batch['input'])
                loss = self.criterion(outputs, batch['target'])
            
            # Scaled backward pass
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
            
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
```

### 🎯 Custom Loss Functions {#custom-loss-functions}

```python
# custom_loss.py
import torch
import torch.nn as nn

class VIVLoss(nn.Module):
    """Custom VIVTransformer loss function"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.mse_loss = nn.MSELoss()
        self.ce_loss = nn.CrossEntropyLoss()
        
    def forward(self, outputs, targets):
        """Compute multi-component loss"""
        losses = {}
        
        # 1. Reconstruction loss
        losses['reconstruction'] = self.mse_loss(
            outputs['reconstruction'], 
            targets['input']
        )
        
        # 2. Attention sparsity loss
        attention_weights = outputs['attention_weights']
        losses['sparsity'] = torch.mean(torch.abs(attention_weights))
        
        # 3. SVD consistency loss
        if 'svd_components' in outputs:
            losses['svd_consistency'] = self._svd_loss(
                outputs['svd_components']
            )
        
        # Weighted combination
        total_loss = 0
        for loss_name, loss_value in losses.items():
            weight = self.config['loss_weights'].get(loss_name, 1.0)
            total_loss += weight * loss_value
        
        return total_loss, losses
```

## Parameter Tuning {#parameter-tuning}

### 🎛️ Hyperparameter Search {#hyperparameter-search}

```python
# hyperparameter_search.py
import optuna
from optuna.pruners import MedianPruner

def objective(trial):
    """Objective function for hyperparameter optimization"""
    
    # Suggest hyperparameters
    config = {
        'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2),
        'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64]),
        'attention_heads': trial.suggest_int('attention_heads', 4, 16),
        'hidden_dim': trial.suggest_categorical('hidden_dim', [256, 512, 1024]),
        'dropout': trial.suggest_float('dropout', 0.1, 0.5)
    }
    
    # Train model with suggested parameters
    model = VIVTransformer(config)
    trainer = Trainer(model, config)
    val_loss = trainer.train(train_loader, valid_loader)
    
    return val_loss

def run_hyperparameter_search():
    """Run hyperparameter search"""
    study = optuna.create_study(
        direction='minimize',
        pruner=MedianPruner()
    )
    
    study.optimize(objective, n_trials=100)
    
    print("Best parameters:", study.best_params)
    print("Best value:", study.best_value)
```

### 📊 Learning Rate Scheduling {#learning-rate-scheduling}

```python
# lr_scheduling.py
import torch.optim.lr_scheduler as lr_scheduler

def get_scheduler(optimizer, config):
    """Get learning rate scheduler"""
    scheduler_type = config['training']['scheduler']
    
    if scheduler_type == 'StepLR':
        return lr_scheduler.StepLR(
            optimizer,
            step_size=config['training']['step_size'],
            gamma=config['training']['gamma']
        )
    
    elif scheduler_type == 'CosineAnnealingLR':
        return lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config['training']['epochs']
        )
    
    elif scheduler_type == 'ReduceLROnPlateau':
        return lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5
        )
    
    elif scheduler_type == 'OneCycleLR':
        return lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=config['training']['learning_rate'],
            total_steps=config['training']['total_steps']
        )
```

## Training Monitoring {#training-monitoring}

### 📈 TensorBoard Integration {#tensorboard-integration}

```python
# monitoring.py
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

class TrainingMonitor:
    def __init__(self, log_dir='logs'):
        self.writer = SummaryWriter(log_dir)
        self.metrics = {}
    
    def log_metrics(self, metrics, step):
        """Log training metrics"""
        for name, value in metrics.items():
            self.writer.add_scalar(name, value, step)
    
    def log_attention_weights(self, attention_weights, step):
        """Visualize attention weights"""
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(attention_weights[0].detach().cpu().numpy())
        plt.colorbar(im)
        self.writer.add_figure('attention_weights', fig, step)
        plt.close()
    
    def log_model_graph(self, model, input_sample):
        """Log model graph"""
        self.writer.add_graph(model, input_sample)
    
    def close(self):
        """Close writer"""
        self.writer.close()
```

### 🔍 Real-time Monitoring {#real-time-monitoring}

```python
# real_time_monitor.py
import time
import psutil
import GPUtil

class SystemMonitor:
    def __init__(self):
        self.start_time = time.time()
    
    def get_system_stats(self):
        """Get system statistics"""
        stats = {
            'cpu_usage': psutil.cpu_percent(),
            'memory_usage': psutil.virtual_memory().percent,
            'gpu_usage': self._get_gpu_usage(),
            'training_time': time.time() - self.start_time
        }
        return stats
    
    def _get_gpu_usage(self):
        """Get GPU usage"""
        gpus = GPUtil.getGPUs()
        if gpus:
            return {
                'gpu_memory': gpus[0].memoryUtil * 100,
                'gpu_load': gpus[0].load * 100
            }
        return {'gpu_memory': 0, 'gpu_load': 0}
```

### 🧰 Integrated Training Monitoring Utilities {#integrated-training-monitoring}

Leverage built-in utilities to monitor hardware, record enhanced logs, and generate visual reports.

```python
# Integrated monitoring with project utilities
from utils.hardware_monitor import HardwareMonitor
from utils.enhanced_logger import EnhancedTrainingLogger
from utils.training_visualizer import TrainingVisualizer

# Initialize monitors
hardware_monitor = HardwareMonitor(
    log_dir="./logs/hardware",
    enable_gpu_monitoring=True,
    enable_cpu_monitoring=True,
)

logger = EnhancedTrainingLogger(log_dir="./logs/training")
visualizer = TrainingVisualizer(log_dir="./logs/hardware")

# Start training session
hardware_monitor.start_training()

for epoch in range(num_epochs):
    hardware_monitor.start_epoch(epoch + 1)

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        hardware_monitor.start_batch(batch_idx + 1)

        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Log detailed batch metrics
        logger.log_batch_metrics({
            'loss/train': loss.item(),
            'lr': optimizer.param_groups[0]['lr'],
        }, step=global_step)

        hardware_monitor.end_batch()

    # Epoch end
    epoch_summary = hardware_monitor.end_epoch()
    logger.log_epoch_summary(epoch_summary)

# Finish training
hardware_monitor.end_training()

# Generate complete visual report
visualizer.generate_complete_report()
```

Tip: You can run the demonstration script directly: `python examples/training_with_monitoring.py`.

See also: [PDEBench Integration]({{ site.baseurl }}/pages/pdebench-integration/) for end-to-end experiments with monitoring enabled.

## Troubleshooting {#troubleshooting}

### 🚨 Common Issues {#common-issues}

#### Memory Issues {#memory-issues}

```python
# memory_optimization.py
def optimize_memory():
    """Memory optimization strategies"""
    
    # 1. Clear cache
    torch.cuda.empty_cache()
    
    # 2. Use gradient checkpointing
    model.gradient_checkpointing_enable()
    
    # 3. Reduce batch size
    config['training']['batch_size'] = 16
    
    # 4. Use mixed precision
    config['training']['use_amp'] = True
```

#### Convergence Issues {#convergence-issues}

```python
# convergence_debug.py
def debug_convergence(model, train_loader):
    """Debug convergence issues"""
    
    # 1. Check gradient norms
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            total_norm += p.grad.data.norm(2).item() ** 2
    total_norm = total_norm ** (1. / 2)
    
    print(f"Gradient norm: {total_norm}")
    
    # 2. Check learning rate
    for param_group in optimizer.param_groups:
        print(f"Current LR: {param_group['lr']}")
    
    # 3. Monitor loss components
    outputs = model(sample_batch)
    loss, loss_components = criterion(outputs, targets)
    
    for name, value in loss_components.items():
        print(f"{name}: {value.item()}")
```

## Best Practices {#best-practices}

### 💡 Training Tips {#training-tips}

1. Data Quality
   - Ensure high-quality, diverse training data
   - Use proper data augmentation
   - Validate data preprocessing pipeline

2. Model Configuration
   - Start with proven architectures
   - Gradually increase model complexity
   - Use appropriate initialization

3. Training Strategy
   - Monitor training/validation curves
   - Use early stopping to prevent overfitting
   - Save model checkpoints regularly

4. Resource Management
   - Monitor GPU memory usage
   - Use distributed training for large models
   - Optimize data loading pipeline

### 🎯 Performance Optimization {#performance-optimization}

```python
# performance_tips.py
def optimize_training_performance():
    """Performance optimization tips"""
    
    # 1. Data loading optimization
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=4,  # Parallel data loading
        pin_memory=True,  # Faster GPU transfer
        persistent_workers=True  # Keep workers alive
    )
    
    # 2. Model compilation (PyTorch 2.0+)
    model = torch.compile(model)
    
    # 3. Efficient attention computation
    with torch.backends.cuda.sdp_kernel(
        enable_flash=True,  # Flash attention
        enable_math=False,
        enable_mem_efficient=False
    ):
        outputs = model(inputs)
```

### 📊 Experiment Management {#experiment-management}

```python
# experiment_management.py
import mlflow
import json

class ExperimentTracker:
    def __init__(self, experiment_name):
        mlflow.set_experiment(experiment_name)
    
    def start_run(self, config):
        """Start experiment run"""
        mlflow.start_run()
        
        # Log parameters
        for key, value in config.items():
            if isinstance(value, dict):
                mlflow.log_params({f"{key}.{k}": v for k, v in value.items()})
            else:
                mlflow.log_param(key, value)
    
    def log_metrics(self, metrics, step):
        """Log metrics"""
        for name, value in metrics.items():
            mlflow.log_metric(name, value, step)
    
    def log_artifacts(self, artifact_path):
        """Log artifacts"""
        mlflow.log_artifacts(artifact_path)
    
    def end_run(self):
        """End experiment run"""
        mlflow.end_run()
```

---

## Related Links {#related-links}

- [Loss Functions]({{ site.baseurl }}/pages/loss-functions/) - Understanding loss function design
- [SVD Loss Functions]({{ site.baseurl }}/pages/svd-loss-functions/) - Special loss functions
- [Multi-Loss Strategy]({{ site.baseurl }}/pages/multi-loss-strategy/) - Multi-loss configuration
- [Hyperparameter Tuning]({{ site.baseurl }}/pages/hyperparameter-tuning/) - Automated parameter optimization
- [Troubleshooting]({{ site.baseurl }}/pages/troubleshooting/) - Solving training issues
- [FAQ]({{ site.baseurl }}/pages/faq/) - Frequently asked questions

For more information and support, please visit our FAQ or Troubleshooting pages.
