---
layout: default
title: Troubleshooting
description: Troubleshooting and problem solving guide for VIVTransformer
nav_order: 18
parent: Getting Help
permalink: /pages/troubleshooting/
---

# Troubleshooting Guide

This document provides solutions and debugging techniques for common issues in the VIVTransformer project.

## 📋 Table of Contents

- [Installation Issues](#installation-issues)
- [Configuration Issues](#configuration-issues)
- [Training Issues](#training-issues)
- [Inference Issues](#inference-issues)
- [Performance Issues](#performance-issues)
- [Memory Issues](#memory-issues)
- [Attention Mechanism Issues](#attention-mechanism-issues)
- [Data Processing Issues](#data-processing-issues)
- [Debugging Tools](#debugging-tools)
- [Common Error Codes](#common-error-codes)

## Installation Issues

### ❌ Dependency Installation Failed

**Problem Description**: Error occurs when running `pip install -r requirements.txt`

**Common Causes**:
1. Incompatible Python version
2. CUDA version mismatch
3. Network connection issues
4. Insufficient permissions

**Solutions**:

```bash
# 1. Check Python version (requires 3.8+)
python --version

# 2. Upgrade pip
python -m pip install --upgrade pip

# 3. Use mirror source
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/

# 4. Install key dependencies step by step
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers
pip install -r requirements.txt

# 5. Use conda environment (recommended)
conda create -n vivtransformer python=3.9
conda activate vivtransformer
pip install -r requirements.txt
```

### ❌ CUDA Related Errors

**Problem Description**: `RuntimeError: CUDA out of memory` or `CUDA device not found`

**Solutions**:

```python
# Check CUDA availability
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"GPU count: {torch.cuda.device_count()}")

# Force CPU usage if CUDA unavailable
device = torch.device('cpu')
model = model.to(device)

# Clear GPU memory
torch.cuda.empty_cache()
```

**CUDA Version Compatibility**:
```bash
# Check system CUDA version
nvcc --version

# Install corresponding PyTorch version
# CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# CPU version
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

## Configuration Issues

### ❌ Configuration File Format Error

**Problem Description**: `yaml.scanner.ScannerError` or configuration parsing failure

**Solutions**:

```python
# Validate YAML format
import yaml

def validate_config(config_path):
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        print("Configuration file format is correct")
        return config
    except yaml.YAMLError as e:
        print(f"YAML format error: {e}")
        return None
    except Exception as e:
        print(f"Configuration file error: {e}")
        return None

# Usage example
config = validate_config('config/default_config.yaml')
if config is None:
    print("Please check configuration file format")
```

**Common YAML Format Issues**:
```yaml
# ❌ Error: Inconsistent indentation
model:
  name: vivtransformer
   hidden_size: 512  # Indentation error

# ✅ Correct: Use consistent 2-space indentation
model:
  name: vivtransformer
  hidden_size: 512

# ❌ Error: No space after colon
learning_rate:0.001

# ✅ Correct: Space after colon
learning_rate: 0.001

# ❌ Error: Special characters in string without quotes
data_path: C:\Users\data  # Windows path

# ✅ Correct: Use quotes or forward slashes
data_path: "C:\\Users\\data"
# or
data_path: C:/Users/data
```

### ❌ Parameter Validation Failed

**Problem Description**: `ValueError: Invalid parameter value`

**Solutions**:

```python
# Parameter validation tool
def validate_model_config(config):
    """Validate model configuration parameters"""
    errors = []
    
    # Check required parameters
    required_params = ['d_model', 'n_heads', 'n_layers']
    for param in required_params:
        if param not in config:
            errors.append(f"Missing required parameter: {param}")
    
    # Check parameter ranges
    if 'd_model' in config:
        if config['d_model'] <= 0 or config['d_model'] % config.get('n_heads', 8) != 0:
            errors.append("d_model must be positive and divisible by n_heads")
    
    if 'learning_rate' in config:
        if not (1e-6 <= config['learning_rate'] <= 1.0):
            errors.append("learning_rate must be between 1e-6 and 1.0")
    
    if errors:
        raise ValueError("\n".join(errors))
    
    return True
    
# Usage example
try:
    validate_model_config(config['model'])
    print("Configuration validation passed")
except ValueError as e:
    print(f"Configuration validation failed: {e}")
```

## Training Issues

### ❌ Training Does Not Converge

**Problem Description**: Loss function does not decrease or oscillates severely

**Possible Causes**:
1. Learning rate too large or too small
2. Inappropriate batch size
3. Gradient explosion or vanishing
4. Data preprocessing issues

**Solutions**:

```python
# 1. Learning rate debugging
import matplotlib.pyplot as plt

def find_learning_rate(model, dataloader, init_lr=1e-8, final_lr=10):
    """Learning rate range test"""
    lrs = []
    losses = []
    
    optimizer = torch.optim.Adam(model.parameters(), lr=init_lr)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=1.1)
    
    for batch_idx, (data, target) in enumerate(dataloader):
        optimizer.zero_grad()
        output = model(data)
        loss = F.mse_loss(output, target)
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        
        lrs.append(optimizer.param_groups[0]['lr'])
        losses.append(loss.item())
        
        if optimizer.param_groups[0]['lr'] > final_lr:
            break
    
    # Plot learning rate vs loss curve
    plt.figure(figsize=(10, 6))
    plt.semilogx(lrs, losses)
    plt.xlabel('Learning Rate')
    plt.ylabel('Loss')
    plt.title('Learning Rate Range Test')
    plt.show()
    
    return lrs, losses

# 2. Gradient monitoring
def monitor_gradients(model):
    """Monitor gradient statistics"""
    total_norm = 0
    param_count = 0
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
            param_count += 1
            
            # Check gradient anomalies
            if torch.isnan(param.grad).any():
                print(f"Warning: {name} contains NaN gradients")
            if torch.isinf(param.grad).any():
                print(f"Warning: {name} contains infinite gradients")
    
    total_norm = total_norm ** (1. / 2)
    print(f"Gradient norm: {total_norm:.6f}")
    
    return total_norm

# 3. Loss function debugging
class DebugLoss(nn.Module):
    def __init__(self, base_loss):
        super().__init__()
        self.base_loss = base_loss
        self.loss_history = []
    
    def forward(self, pred, target):
        loss = self.base_loss(pred, target)
        
        # Record loss statistics
        self.loss_history.append(loss.item())
        
        # Check for anomalies
        if torch.isnan(loss):
            print("Warning: Loss is NaN")
            print(f"Prediction stats: min={pred.min():.6f}, max={pred.max():.6f}, mean={pred.mean():.6f}")
            print(f"Target stats: min={target.min():.6f}, max={target.max():.6f}, mean={target.mean():.6f}")
        
        return loss
```

### ❌ Slow Training Speed

**Problem Description**: Model inference takes too long

**Solutions**:

```python
# 1. Model optimization
import torch.jit

# TorchScript compilation
model.eval()
scripted_model = torch.jit.script(model)

# Or use trace
sample_input = torch.randn(1, 100, 512)
traced_model = torch.jit.trace(model, sample_input)

# 2. Inference acceleration
@torch.no_grad()
def fast_inference(model, data):
    """Fast inference"""
    model.eval()
    
    # Use half precision
    if torch.cuda.is_available():
        model = model.half()
        data = data.half()
    
    # Batch inference
    results = []
    batch_size = 32
    
    for i in range(0, len(data), batch_size):
        batch = data[i:i + batch_size]
        output = model(batch)
        results.append(output)
    
    return torch.cat(results, dim=0)

# 3. Memory optimization
def memory_efficient_inference(model, data, chunk_size=1000):
    """Memory efficient inference"""
    model.eval()
    results = []
    
    with torch.no_grad():
        for i in range(0, data.size(1), chunk_size):
            chunk = data[:, i:i + chunk_size]
            output = model(chunk)
            results.append(output.cpu())  # Move to CPU to free GPU memory
            torch.cuda.empty_cache()
    
    return torch.cat(results, dim=1)
```

## Performance Issues

### ❌ Slow Training Speed

**Solutions**:

```python
# 1. Data loading optimization
from torch.utils.data import DataLoader

# Optimize DataLoader
dataloader = DataLoader(
    dataset,
    batch_size=32,
    num_workers=4,  # Multi-process loading
    pin_memory=True,  # Pin memory
    persistent_workers=True,  # Persistent workers
    prefetch_factor=2  # Prefetch factor
)

# 2. Mixed precision training
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

# 3. Gradient accumulation
accumulation_steps = 4

for i, (data, target) in enumerate(dataloader):
    with autocast():
        output = model(data)
        loss = criterion(output, target) / accumulation_steps
    
    scaler.scale(loss).backward()
    
    if (i + 1) % accumulation_steps == 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
```

## Memory Issues

### ❌ GPU Memory Insufficient

**Solutions**:

```python
# 1. Memory monitoring
def monitor_gpu_memory():
    """Monitor GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        cached = torch.cuda.memory_reserved() / 1024**3
        print(f"GPU Memory - Allocated: {allocated:.2f}GB, Cached: {cached:.2f}GB")

# 2. Gradient checkpointing
from torch.utils.checkpoint import checkpoint

class CheckpointedTransformerLayer(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer
    
    def forward(self, x):
        return checkpoint(self.layer, x)

# 3. Model parallelism
class ModelParallelVIVTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        # Place different layers on different GPUs
        self.embedding = nn.Linear(config.input_dim, config.d_model).to('cuda:0')
        self.transformer_layers = nn.ModuleList([
            TransformerLayer(config).to(f'cuda:{i % torch.cuda.device_count()}')
            for i in range(config.n_layers)
        ])
        self.output_layer = nn.Linear(config.d_model, config.output_dim).to('cuda:0')
    
    def forward(self, x):
        x = x.to('cuda:0')
        x = self.embedding(x)
        
        for i, layer in enumerate(self.transformer_layers):
            device = f'cuda:{i % torch.cuda.device_count()}'
            x = x.to(device)
            x = layer(x)
        
        x = x.to('cuda:0')
        return self.output_layer(x)
```

## Attention Mechanism Issues

### ❌ Attention Weight Anomalies

**Solutions**:

```python
# 1. Attention visualization
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_attention(attention_weights, save_path=None):
    """Visualize attention weights"""
    # attention_weights: [batch_size, n_heads, seq_len, seq_len]
    
    batch_size, n_heads, seq_len, _ = attention_weights.shape
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    for head in range(min(8, n_heads)):
        attention_map = attention_weights[0, head].detach().cpu().numpy()
        
        sns.heatmap(
            attention_map,
            ax=axes[head],
            cmap='Blues',
            cbar=True,
            square=True
        )
        axes[head].set_title(f'Head {head + 1}')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

# 2. Attention pattern analysis
def analyze_attention_patterns(attention_weights):
    """Analyze attention patterns"""
    # Calculate attention entropy
    entropy = -torch.sum(attention_weights * torch.log(attention_weights + 1e-8), dim=-1)
    
    # Calculate attention concentration
    max_attention = torch.max(attention_weights, dim=-1)[0]
    
    # Statistics
    stats = {
        'entropy_mean': entropy.mean().item(),
        'entropy_std': entropy.std().item(),
        'max_attention_mean': max_attention.mean().item(),
        'max_attention_std': max_attention.std().item(),
    }
    
    print("Attention Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value:.4f}")
    
    return stats

# 3. Attention regularization
class RegularizedAttention(nn.Module):
    def __init__(self, attention_module, entropy_weight=0.01):
        super().__init__()
        self.attention = attention_module
        self.entropy_weight = entropy_weight
    
    def forward(self, x):
        output, attention_weights = self.attention(x)
        
        # Calculate entropy regularization loss
        entropy = -torch.sum(attention_weights * torch.log(attention_weights + 1e-8), dim=-1)
        entropy_loss = -entropy.mean()  # Negative entropy, encourage diversity
        
        return output, attention_weights, entropy_loss * self.entropy_weight
```

## Data Processing Issues

### ❌ Data Loading Error

**Solutions**:

```python
# 1. Data validation
def validate_data(data_path):
    """Validate data file"""
    try:
        data = torch.load(data_path)
        
        # Check data type
        if not isinstance(data, (torch.Tensor, dict, list)):
            raise ValueError(f"Unsupported data type: {type(data)}")
        
        # Check data shape
        if isinstance(data, torch.Tensor):
            print(f"Data shape: {data.shape}")
            print(f"Data type: {data.dtype}")
            print(f"Data range: [{data.min():.4f}, {data.max():.4f}]")
            
            # Check for anomalies
            if torch.isnan(data).any():
                print("Warning: Data contains NaN values")
            if torch.isinf(data).any():
                print("Warning: Data contains infinite values")
        
        return True
        
    except Exception as e:
        print(f"Data validation failed: {e}")
        return False

# 2. Data preprocessing pipeline
class DataPreprocessor:
    def __init__(self, config):
        self.config = config
        self.scaler = None
    
    def fit_transform(self, data):
        """Fit and transform data"""
        # Standardization
        if self.config.get('normalize', True):
            self.scaler = StandardScaler()
            data = self.scaler.fit_transform(data)
        
        # Outlier handling
        if self.config.get('clip_outliers', True):
            data = self.clip_outliers(data)
        
        # Fill missing values
        if self.config.get('fill_missing', True):
            data = self.fill_missing_values(data)
        
        return torch.tensor(data, dtype=torch.float32)
    
    def clip_outliers(self, data, std_threshold=3):
        """Clip outliers"""
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        
        lower_bound = mean - std_threshold * std
        upper_bound = mean + std_threshold * std
        
        return np.clip(data, lower_bound, upper_bound)
    
    def fill_missing_values(self, data, method='mean'):
        """Fill missing values"""
        if method == 'mean':
            mask = np.isnan(data)
            data[mask] = np.nanmean(data, axis=0)[mask[0]]
        elif method == 'forward_fill':
            data = pd.DataFrame(data).fillna(method='ffill').values
        
        return data
```

## Debugging Tools

### 🔧 Debugging Toolkit

```python
# 1. Model debugger
class ModelDebugger:
    def __init__(self, model):
        self.model = model
        self.hooks = []
        self.activations = {}
        self.gradients = {}
    
    def register_hooks(self):
        """Register debugging hooks"""
        def forward_hook(name):
            def hook(module, input, output):
                self.activations[name] = {
                    'input_shape': input[0].shape if input else None,
                    'output_shape': output.shape if hasattr(output, 'shape') else None,
                    'output_mean': output.mean().item() if hasattr(output, 'mean') else None,
                    'output_std': output.std().item() if hasattr(output, 'std') else None,
                }
            return hook
        
        def backward_hook(name):
            def hook(module, grad_input, grad_output):
                if grad_output[0] is not None:
                    self.gradients[name] = {
                        'grad_norm': grad_output[0].norm().item(),
                        'grad_mean': grad_output[0].mean().item(),
                        'grad_std': grad_output[0].std().item(),
                    }
            return hook
        
        for name, module in self.model.named_modules():
            if len(list(module.children())) == 0:  # Leaf modules
                self.hooks.append(module.register_forward_hook(forward_hook(name)))
                self.hooks.append(module.register_backward_hook(backward_hook(name)))
    
    def print_debug_info(self):
        """Print debugging information"""
        print("=== Activation Statistics ===")
        for name, stats in self.activations.items():
            print(f"{name}: {stats}")
        
        print("\n=== Gradient Statistics ===")
        for name, stats in self.gradients.items():
            print(f"{name}: {stats}")
    
    def remove_hooks(self):
        """Remove hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

# 2. Training monitor
class TrainingMonitor:
    def __init__(self, log_dir='./logs'):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        self.metrics = defaultdict(list)
        self.start_time = time.time()
    
    def log_metrics(self, epoch, **kwargs):
        """Log metrics"""
        self.metrics['epoch'].append(epoch)
        self.metrics['timestamp'].append(time.time() - self.start_time)
        
        for key, value in kwargs.items():
            self.metrics[key].append(value)
    
    def plot_metrics(self, metrics_to_plot=None):
        """Plot metric curves"""
        if metrics_to_plot is None:
            metrics_to_plot = ['train_loss', 'val_loss']
        
        fig, axes = plt.subplots(len(metrics_to_plot), 1, figsize=(10, 6 * len(metrics_to_plot)))
        if len(metrics_to_plot) == 1:
            axes = [axes]
        
        for i, metric in enumerate(metrics_to_plot):
            if metric in self.metrics:
                axes[i].plot(self.metrics['epoch'], self.metrics[metric])
                axes[i].set_title(f'{metric.title()}')
                axes[i].set_xlabel('Epoch')
                axes[i].set_ylabel(metric)
                axes[i].grid(True)
        
        plt.tight_layout()
        plt.savefig(self.log_dir / 'training_metrics.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_logs(self):
        """Save logs"""
        log_file = self.log_dir / 'training_log.json'
        with open(log_file, 'w') as f:
            json.dump(dict(self.metrics), f, indent=2)
```

## Common Error Codes

### 📋 Error Code Reference

| Error Code | Error Description | Possible Cause | Solution |
|------------|------------------|-----------------|----------|
| VIV-001 | Configuration file parsing failed | YAML format error | Check configuration file syntax |
| VIV-002 | Model initialization failed | Incompatible parameters | Validate model configuration parameters |
| VIV-003 | Data loading failed | Incorrect file path or unsupported format | Check data file path and format |
| VIV-004 | CUDA out of memory | Batch size too large or model too big | Reduce batch size or use gradient accumulation |
| VIV-005 | Gradient explosion | Learning rate too high or unstable model | Lower learning rate or add gradient clipping |
| VIV-006 | Attention computation error | Sequence length mismatch | Check input sequence dimensions |
| VIV-007 | Loss function returns NaN | Numerical instability | Check data preprocessing and model output |
| VIV-008 | Model save failed | Insufficient disk space or permission issues | Check storage space and file permissions |

### 🚨 Emergency Recovery

```python
# Emergency recovery script
def emergency_recovery(checkpoint_dir, config_path):
    """Emergency training recovery"""
    try:
        # 1. Load latest checkpoint
        checkpoints = list(Path(checkpoint_dir).glob('*.pth'))
        if not checkpoints:
            print("No checkpoint files found")
            return None
        
        latest_checkpoint = max(checkpoints, key=lambda x: x.stat().st_mtime)
        print(f"Loading checkpoint: {latest_checkpoint}")
        
        # 2. Load model state
        checkpoint = torch.load(latest_checkpoint, map_location='cpu')
        
        # 3. Rebuild model
        config = load_config(config_path)
        model = VIVTransformer(config)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # 4. Validate model
        model.eval()
        test_input = torch.randn(1, 100, config.input_dim)
        with torch.no_grad():
            output = model(test_input)
        
        print(f"Model recovery successful, output shape: {output.shape}")
        return model, checkpoint
        
    except Exception as e:
        print(f"Recovery failed: {e}")
        return None

# Usage example
model, checkpoint = emergency_recovery('./checkpoints', './config/default_config.yaml')
if model is not None:
    print(f"Resumed training from epoch {checkpoint['epoch']}")
```

## 📞 Getting Help

If the above solutions cannot resolve your issue, please get help through the following channels:

1. **GitHub Issues**: Submit detailed issue reports in the project repository
2. **Discussion Forum**: Participate in community discussions and share experiences
3. **Documentation**: Refer to complete API documentation and user guides
4. **Example Code**: Check example code in the project

When submitting issues, please include the following information:
- Complete error stack trace
- Configuration file used
- Python and dependency package versions
- Hardware environment information
- Minimal code example to reproduce the issue
