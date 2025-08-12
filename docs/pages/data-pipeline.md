---
layout: default
title: Data Pipeline
description: Design and implementation of the data processing pipeline
permalink: /pages/data-pipeline/
---

# Data Stream Processing {#data-stream-processing}

This document details the data processing workflow of the VIVTransformer project, including data loading, preprocessing, augmentation, and batching strategies.

## 📋 Table of Contents {#table-of-contents}

- [Data Stream Overview](#data-stream-overview)
- [Data Loading](#data-loading)
- [Data Preprocessing](#data-preprocessing)
- [Data Augmentation](#data-augmentation)
- [Batching Strategies](#batching-strategies)
- [Memory Optimization](#memory-optimization)
- [Distributed Processing](#distributed-processing)

## Data Stream Overview {#data-stream-overview}

### 🔄 Complete Data Workflow {#complete-data-workflow}

```
Raw Data → Data Loading → Preprocessing → Data Augmentation → Batching → Model Training
    ↓         ↓           ↓             ↓                ↓          ↓
  .pt files  Dataset    Normalization Random Transform  DataLoader Training Loop
```

### 📊 Data Characteristics {#data-characteristics}

| Property | Value | Description |
|----------|-------|-------------|
| **Input Dimension** | 400 | Original feature dimension |
| **Output Dimension** | 40000 | Target prediction dimension |
| **Sequence Length** | 49 | Time step length |
| **Data Type** | float32 | Floating-point precision |
| **Storage Format** | PyTorch Tensor | .pt file format |

## Data Loading {#data-loading}

### 📁 Dataset Class Implementation {#dataset-class-implementation}

```python
class VIVDataset(Dataset):
    """VIV Dataset Class"""
    
    def __init__(self, data_path, crop_size=None, use_augmentation=False):
        self.data = torch.load(data_path)
        self.crop_size = crop_size
        self.use_augmentation = use_augmentation
        
        # Data validation
        self._validate_data()
        
        # Pre-compute statistics
        self._compute_statistics()
    
    def __len__(self):
        return len(self.data['input'])
    
    def __getitem__(self, idx):
        input_data = self.data['input'][idx]
        target_data = self.data['target'][idx]
        
        # Apply preprocessing
        input_data = self._preprocess(input_data)
        target_data = self._preprocess(target_data)
        
        # Apply data augmentation
        if self.use_augmentation:
            input_data, target_data = self._augment(input_data, target_data)
        
        return input_data, target_data
```

### 🔍 Data Validation {#data-validation}

```python
def _validate_data(self):
    """Validate data integrity and format"""
    
    # Check required fields
    required_keys = ['input', 'target']
    for key in required_keys:
        if key not in self.data:
            raise ValueError(f"Missing required key: {key}")
    
    # Check data shapes
    input_shape = self.data['input'].shape
    target_shape = self.data['target'].shape
    
    if len(input_shape) != 3:  # [N, seq_len, input_dim]
        raise ValueError(f"Invalid input shape: {input_shape}")
    
    if len(target_shape) != 3:  # [N, seq_len, output_dim]
        raise ValueError(f"Invalid target shape: {target_shape}")
    
    # Check data types
    if self.data['input'].dtype != torch.float32:
        self.data['input'] = self.data['input'].float()
    
    if self.data['target'].dtype != torch.float32:
        self.data['target'] = self.data['target'].float()
    
    print(f"✅ Data validation passed: {input_shape} → {target_shape}")
```

### 📈 Statistics Computation {#statistics-computation}

```python
def _compute_statistics(self):
    """Compute dataset statistics"""
    
    # Input data statistics
    self.input_mean = self.data['input'].mean(dim=(0, 1))
    self.input_std = self.data['input'].std(dim=(0, 1))
    
    # Target data statistics
    self.target_mean = self.data['target'].mean(dim=(0, 1))
    self.target_std = self.data['target'].std(dim=(0, 1))
    
    # Data ranges
    self.input_min = self.data['input'].min()
    self.input_max = self.data['input'].max()
    self.target_min = self.data['target'].min()
    self.target_max = self.data['target'].max()
    
    print(f"📊 Data Statistics:")
    print(f"   Input range: [{self.input_min:.4f}, {self.input_max:.4f}]")
    print(f"   Target range: [{self.target_min:.4f}, {self.target_max:.4f}]")
```

## Data Preprocessing {#data-preprocessing}

### 🔧 Normalization Processing {#normalization-processing}

```python
def _preprocess(self, data):
    """Data preprocessing pipeline"""
    
    # 1. Normalization
    if hasattr(self, 'normalize') and self.normalize:
        data = self._normalize(data)
    
    # 2. Cropping
    if self.crop_size is not None:
        data = self._crop(data)
    
    # 3. Padding
    if hasattr(self, 'pad_size') and self.pad_size:
        data = self._pad(data)
    
    return data

def _normalize(self, data):
    """Data normalization"""
    # Z-score normalization
    mean = data.mean(dim=-1, keepdim=True)
    std = data.std(dim=-1, keepdim=True)
    return (data - mean) / (std + 1e-8)

def _crop(self, data):
    """Data cropping"""
    if data.shape[-1] > self.crop_size[0]:
        start_idx = torch.randint(0, data.shape[-1] - self.crop_size[0] + 1, (1,))
        data = data[..., start_idx:start_idx + self.crop_size[0]]
    return data

def _pad(self, data):
    """Data padding"""
    if data.shape[-1] < self.pad_size:
        pad_width = self.pad_size - data.shape[-1]
        padding = torch.zeros(*data.shape[:-1], pad_width)
        data = torch.cat([data, padding], dim=-1)
    return data
```

### 🎯 Feature Engineering {#feature-engineering}

```python
class FeatureEngineer:
    """Feature Engineering Class"""
    
    @staticmethod
    def add_temporal_features(data):
        """Add temporal features"""
        seq_len = data.shape[1]
        
        # Time position encoding
        time_pos = torch.arange(seq_len).float() / seq_len
        time_pos = time_pos.unsqueeze(0).unsqueeze(-1)
        time_pos = time_pos.expand(data.shape[0], -1, 1)
        
        # Concatenate temporal features
        data = torch.cat([data, time_pos], dim=-1)
        return data
    
    @staticmethod
    def add_statistical_features(data):
        """Add statistical features"""
        # Sliding window statistics
        window_size = 5
        
        # Moving average
        moving_avg = F.avg_pool1d(
            data.transpose(1, 2), 
            kernel_size=window_size, 
            stride=1, 
            padding=window_size//2
        ).transpose(1, 2)
        
        # Moving standard deviation
        moving_std = torch.zeros_like(moving_avg)
        for i in range(data.shape[1]):
            start = max(0, i - window_size//2)
            end = min(data.shape[1], i + window_size//2 + 1)
            moving_std[:, i] = data[:, start:end].std(dim=1)
        
        # Concatenate statistical features
        data = torch.cat([data, moving_avg, moving_std], dim=-1)
        return data
```

## Data Augmentation {#data-augmentation}

### 🎲 Augmentation Strategies {#augmentation-strategies}

```python
def _augment(self, input_data, target_data):
    """Data augmentation pipeline"""
    
    # Randomly select augmentation strategies
    augmentations = [
        self._add_noise,
        self._time_shift,
        self._amplitude_scale,
        self._frequency_mask
    ]
    
    # Randomly apply 1-2 augmentations
    num_augs = torch.randint(1, 3, (1,)).item()
    selected_augs = torch.randperm(len(augmentations))[:num_augs]
    
    for aug_idx in selected_augs:
        input_data, target_data = augmentations[aug_idx](input_data, target_data)
    
    return input_data, target_data

def _add_noise(self, input_data, target_data):
    """Add Gaussian noise"""
    noise_level = 0.01
    noise = torch.randn_like(input_data) * noise_level
    input_data = input_data + noise
    return input_data, target_data

def _time_shift(self, input_data, target_data):
    """Time shifting"""
    max_shift = 3
    shift = torch.randint(-max_shift, max_shift + 1, (1,)).item()
    
    if shift > 0:
        input_data = torch.cat([input_data[:, shift:], input_data[:, :shift]], dim=1)
        target_data = torch.cat([target_data[:, shift:], target_data[:, :shift]], dim=1)
    elif shift < 0:
        input_data = torch.cat([input_data[:, shift:], input_data[:, :shift]], dim=1)
        target_data = torch.cat([target_data[:, shift:], target_data[:, :shift]], dim=1)
    
    return input_data, target_data

def _amplitude_scale(self, input_data, target_data):
    """Amplitude scaling"""
    scale_range = (0.8, 1.2)
    scale = torch.uniform(*scale_range, (1,)).item()
    input_data = input_data * scale
    return input_data, target_data

def _frequency_mask(self, input_data, target_data):
    """Frequency masking"""
    mask_ratio = 0.1
    mask_size = int(input_data.shape[-1] * mask_ratio)
    
    start_idx = torch.randint(0, input_data.shape[-1] - mask_size + 1, (1,)).item()
    input_data[..., start_idx:start_idx + mask_size] = 0
    
    return input_data, target_data
```

## Batching Strategies {#batching-strategies}

### 📦 DataLoader Configuration {#dataloader-configuration}

```python
def get_data_loaders(config):
    """Create data loaders"""
    
    # Create dataset
    dataset = VIVDataset(
        data_path=config['data']['path'],
        crop_size=config['data'].get('crop_size'),
        use_augmentation=config['data'].get('use_augmentation', False)
    )
    
    # Data splitting
    train_size = int(0.7 * len(dataset))
    valid_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - valid_size
    
    train_dataset, valid_dataset, test_dataset = random_split(
        dataset, [train_size, valid_size, test_size]
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=True,
        num_workers=config['data'].get('num_workers', 4),
        pin_memory=config['data'].get('pin_memory', True),
        drop_last=True
    )
    
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False,
        num_workers=config['data'].get('num_workers', 4),
        pin_memory=config['data'].get('pin_memory', True)
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False,
        num_workers=config['data'].get('num_workers', 4),
        pin_memory=config['data'].get('pin_memory', True)
    )
    
    return train_loader, valid_loader, test_loader
```

### ⚡ Dynamic Batch Size {#dynamic-batch-size}

```python
class DynamicBatchSampler:
    """Dynamic batch size sampler"""
    
    def __init__(self, dataset, max_batch_size, max_memory_gb=8):
        self.dataset = dataset
        self.max_batch_size = max_batch_size
        self.max_memory_gb = max_memory_gb
        self.current_batch_size = max_batch_size
    
    def adjust_batch_size(self):
        """Adjust batch size based on GPU memory usage"""
        if torch.cuda.is_available():
            memory_used = torch.cuda.memory_allocated() / 1e9
            memory_ratio = memory_used / self.max_memory_gb
            
            if memory_ratio > 0.9:
                self.current_batch_size = max(16, self.current_batch_size // 2)
            elif memory_ratio < 0.5:
                self.current_batch_size = min(self.max_batch_size, self.current_batch_size * 2)
    
    def __iter__(self):
        indices = torch.randperm(len(self.dataset))
        
        for i in range(0, len(indices), self.current_batch_size):
            batch_indices = indices[i:i + self.current_batch_size]
            yield batch_indices.tolist()
            
            # Dynamically adjust batch size
            self.adjust_batch_size()
```

## Memory Optimization {#memory-optimization}

### 💾 Memory Management Strategies {#memory-management-strategies}

```python
class MemoryOptimizedDataset(Dataset):
    """Memory-optimized dataset"""
    
    def __init__(self, data_path, cache_size=1000):
        self.data_path = data_path
        self.cache_size = cache_size
        self.cache = {}
        self.access_count = {}
        
        # Only load metadata
        self.metadata = self._load_metadata()
    
    def __getitem__(self, idx):
        # Check cache
        if idx in self.cache:
            self.access_count[idx] += 1
            return self.cache[idx]
        
        # Load data
        data = self._load_single_item(idx)
        
        # Update cache
        self._update_cache(idx, data)
        
        return data
    
    def _update_cache(self, idx, data):
        """Update cache using LRU strategy"""
        if len(self.cache) >= self.cache_size:
            # Remove least recently used item
            lru_idx = min(self.access_count, key=self.access_count.get)
            del self.cache[lru_idx]
            del self.access_count[lru_idx]
        
        self.cache[idx] = data
        self.access_count[idx] = 1
```

### 🔄 Prefetching Strategy {#prefetching-strategy}

```python
class PrefetchDataLoader:
    """Prefetching data loader"""
    
    def __init__(self, dataloader, device, prefetch_factor=2):
        self.dataloader = dataloader
        self.device = device
        self.prefetch_factor = prefetch_factor
        self.stream = torch.cuda.Stream()
    
    def __iter__(self):
        first = True
        
        for next_input, next_target in self.dataloader:
            with torch.cuda.stream(self.stream):
                next_input = next_input.to(self.device, non_blocking=True)
                next_target = next_target.to(self.device, non_blocking=True)
            
            if not first:
                yield input, target
            else:
                first = False
            
            torch.cuda.current_stream().wait_stream(self.stream)
            input, target = next_input, next_target
        
        yield input, target
```

## Distributed Processing {#distributed-processing}

### 🌐 Distributed Data Loading {#distributed-data-loading}

```python
def setup_distributed_data(config, rank, world_size):
    """Setup distributed data loading"""
    
    # Create distributed sampler
    dataset = VIVDataset(config['data']['path'])
    sampler = DistributedSampler(
        dataset, 
        num_replicas=world_size, 
        rank=rank,
        shuffle=True
    )
    
    # Create data loader
    dataloader = DataLoader(
        dataset,
        batch_size=config['data']['batch_size'] // world_size,
        sampler=sampler,
        num_workers=config['data'].get('num_workers', 4),
        pin_memory=True
    )
    
    return dataloader, sampler
```

### 📊 Data Parallel Strategy {#data-parallel-strategy}

```python
class DataParallelStrategy:
    """Data parallel strategy"""
    
    @staticmethod
    def split_batch(batch, num_gpus):
        """Split batch across multiple GPUs"""
        input_data, target_data = batch
        batch_size = input_data.shape[0]
        
        # Calculate batch size per GPU
        per_gpu_batch_size = batch_size // num_gpus
        
        # Split data
        input_splits = torch.split(input_data, per_gpu_batch_size)
        target_splits = torch.split(target_data, per_gpu_batch_size)
        
        return list(zip(input_splits, target_splits))
    
    @staticmethod
    def gather_results(results):
        """Gather multi-GPU results"""
        if isinstance(results[0], torch.Tensor):
            return torch.cat(results, dim=0)
        elif isinstance(results[0], dict):
            gathered = {}
            for key in results[0].keys():
                gathered[key] = torch.cat([r[key] for r in results], dim=0)
            return gathered
        else:
            return results
```

## 🔧 Configuration Examples {#configuration-examples}

### 📝 Complete Data Configuration {#complete-data-configuration}

```yaml
data:
  # Basic configuration
  path: "data/viv_dataset.pt"
  batch_size: 128
  num_workers: 4
  pin_memory: true
  
  # Preprocessing configuration
  normalize: true
  crop_size: [400]
  pad_size: null
  
  # Data augmentation configuration
  use_augmentation: true
  augmentation:
    noise_level: 0.01
    time_shift_range: 3
    amplitude_scale_range: [0.8, 1.2]
    frequency_mask_ratio: 0.1
  
  # Memory optimization configuration
  cache_size: 1000
  prefetch_factor: 2
  max_memory_gb: 8
  
  # Distributed configuration
  distributed: false
  world_size: 1
  rank: 0
```

---

**💡 Tip**: Data processing is the foundation of deep learning projects. A well-designed data pipeline can significantly improve training efficiency and model performance.

---

*Need help? Check the [FAQ](faq) or [Troubleshooting](troubleshooting) pages.*
