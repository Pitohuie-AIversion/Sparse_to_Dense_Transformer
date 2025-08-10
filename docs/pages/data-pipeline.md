---
layout: doc
title: Data Pipeline
description: 数据处理管道的设计和实现
permalink: /pages/data-pipeline/
---

# 数据流处理 {#数据流处理}

本文档详细介绍VIVTransformer项目的数据处理流程，包括数据加载、预处理、增强和批处理策略。

## 📋 目录 {#目录}

- [数据流概览](#数据流概览)
- [数据加载](#数据加载)
- [数据预处理](#数据预处理)
- [数据增强](#数据增强)
- [批处理策略](#批处理策略)
- [内存优化](#内存优化)
- [分布式处理](#分布式处理)

## 数据流概览 {#数据流概览}

### 🔄 完整数据流程 {#完整数据流程}

```
原始数据 → 数据加载 → 预处理 → 数据增强 → 批处理 → 模型训练
    ↓         ↓        ↓        ↓        ↓        ↓
  .pt文件   Dataset   标准化    随机变换   DataLoader  训练循环
```

### 📊 数据特征 {#数据特征}

| 属性 | 值 | 说明 |
|------|----|----- |
| **输入维度** | 400 | 原始特征维度 |
| **输出维度** | 40000 | 目标预测维度 |
| **序列长度** | 49 | 时间步长度 |
| **数据类型** | float32 | 浮点数精度 |
| **存储格式** | PyTorch Tensor | .pt文件格式 |

## 数据加载 {#数据加载}

### 📁 数据集类实现 {#数据集类实现}

```python
class VIVDataset(Dataset):
    """VIV数据集类"""
    
    def __init__(self, data_path, crop_size=None, use_augmentation=False):
        self.data = torch.load(data_path)
        self.crop_size = crop_size
        self.use_augmentation = use_augmentation
        
        # 数据验证
        self._validate_data()
        
        # 预计算统计信息
        self._compute_statistics()
    
    def __len__(self):
        return len(self.data['input'])
    
    def __getitem__(self, idx):
        input_data = self.data['input'][idx]
        target_data = self.data['target'][idx]
        
        # 应用预处理
        input_data = self._preprocess(input_data)
        target_data = self._preprocess(target_data)
        
        # 应用数据增强
        if self.use_augmentation:
            input_data, target_data = self._augment(input_data, target_data)
        
        return input_data, target_data
```

### 🔍 数据验证 {#数据验证}

```python
def _validate_data(self):
    """验证数据完整性和格式"""
    
    # 检查必要字段
    required_keys = ['input', 'target']
    for key in required_keys:
        if key not in self.data:
            raise ValueError(f"Missing required key: {key}")
    
    # 检查数据形状
    input_shape = self.data['input'].shape
    target_shape = self.data['target'].shape
    
    if len(input_shape) != 3:  # [N, seq_len, input_dim]
        raise ValueError(f"Invalid input shape: {input_shape}")
    
    if len(target_shape) != 3:  # [N, seq_len, output_dim]
        raise ValueError(f"Invalid target shape: {target_shape}")
    
    # 检查数据类型
    if self.data['input'].dtype != torch.float32:
        self.data['input'] = self.data['input'].float()
    
    if self.data['target'].dtype != torch.float32:
        self.data['target'] = self.data['target'].float()
    
    print(f"✅ 数据验证通过: {input_shape} → {target_shape}")
```

### 📈 统计信息计算 {#统计信息计算}

```python
def _compute_statistics(self):
    """计算数据集统计信息"""
    
    # 输入数据统计
    self.input_mean = self.data['input'].mean(dim=(0, 1))
    self.input_std = self.data['input'].std(dim=(0, 1))
    
    # 目标数据统计
    self.target_mean = self.data['target'].mean(dim=(0, 1))
    self.target_std = self.data['target'].std(dim=(0, 1))
    
    # 数据范围
    self.input_min = self.data['input'].min()
    self.input_max = self.data['input'].max()
    self.target_min = self.data['target'].min()
    self.target_max = self.data['target'].max()
    
    print(f"📊 数据统计信息:")
    print(f"   输入范围: [{self.input_min:.4f}, {self.input_max:.4f}]")
    print(f"   目标范围: [{self.target_min:.4f}, {self.target_max:.4f}]")
```

## 数据预处理 {#数据预处理}

### 🔧 标准化处理 {#标准化处理}

```python
def _preprocess(self, data):
    """数据预处理流程"""
    
    # 1. 标准化
    if hasattr(self, 'normalize') and self.normalize:
        data = self._normalize(data)
    
    # 2. 裁剪
    if self.crop_size is not None:
        data = self._crop(data)
    
    # 3. 填充
    if hasattr(self, 'pad_size') and self.pad_size:
        data = self._pad(data)
    
    return data

def _normalize(self, data):
    """数据标准化"""
    # Z-score标准化
    mean = data.mean(dim=-1, keepdim=True)
    std = data.std(dim=-1, keepdim=True)
    return (data - mean) / (std + 1e-8)

def _crop(self, data):
    """数据裁剪"""
    if data.shape[-1] > self.crop_size[0]:
        start_idx = torch.randint(0, data.shape[-1] - self.crop_size[0] + 1, (1,))
        data = data[..., start_idx:start_idx + self.crop_size[0]]
    return data

def _pad(self, data):
    """数据填充"""
    if data.shape[-1] < self.pad_size:
        pad_width = self.pad_size - data.shape[-1]
        padding = torch.zeros(*data.shape[:-1], pad_width)
        data = torch.cat([data, padding], dim=-1)
    return data
```

### 🎯 特征工程 {#特征工程}

```python
class FeatureEngineer:
    """特征工程类"""
    
    @staticmethod
    def add_temporal_features(data):
        """添加时间特征"""
        seq_len = data.shape[1]
        
        # 时间位置编码
        time_pos = torch.arange(seq_len).float() / seq_len
        time_pos = time_pos.unsqueeze(0).unsqueeze(-1)
        time_pos = time_pos.expand(data.shape[0], -1, 1)
        
        # 拼接时间特征
        data = torch.cat([data, time_pos], dim=-1)
        return data
    
    @staticmethod
    def add_statistical_features(data):
        """添加统计特征"""
        # 滑动窗口统计
        window_size = 5
        
        # 滑动平均
        moving_avg = F.avg_pool1d(
            data.transpose(1, 2), 
            kernel_size=window_size, 
            stride=1, 
            padding=window_size//2
        ).transpose(1, 2)
        
        # 滑动标准差
        moving_std = torch.zeros_like(moving_avg)
        for i in range(data.shape[1]):
            start = max(0, i - window_size//2)
            end = min(data.shape[1], i + window_size//2 + 1)
            moving_std[:, i] = data[:, start:end].std(dim=1)
        
        # 拼接统计特征
        data = torch.cat([data, moving_avg, moving_std], dim=-1)
        return data
```

## 数据增强 {#数据增强}

### 🎲 增强策略 {#增强策略}

```python
def _augment(self, input_data, target_data):
    """数据增强流程"""
    
    # 随机选择增强策略
    augmentations = [
        self._add_noise,
        self._time_shift,
        self._amplitude_scale,
        self._frequency_mask
    ]
    
    # 随机应用1-2种增强
    num_augs = torch.randint(1, 3, (1,)).item()
    selected_augs = torch.randperm(len(augmentations))[:num_augs]
    
    for aug_idx in selected_augs:
        input_data, target_data = augmentations[aug_idx](input_data, target_data)
    
    return input_data, target_data

def _add_noise(self, input_data, target_data):
    """添加高斯噪声"""
    noise_level = 0.01
    noise = torch.randn_like(input_data) * noise_level
    input_data = input_data + noise
    return input_data, target_data

def _time_shift(self, input_data, target_data):
    """时间偏移"""
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
    """幅度缩放"""
    scale_range = (0.8, 1.2)
    scale = torch.uniform(*scale_range, (1,)).item()
    input_data = input_data * scale
    return input_data, target_data

def _frequency_mask(self, input_data, target_data):
    """频域掩码"""
    mask_ratio = 0.1
    mask_size = int(input_data.shape[-1] * mask_ratio)
    
    start_idx = torch.randint(0, input_data.shape[-1] - mask_size + 1, (1,)).item()
    input_data[..., start_idx:start_idx + mask_size] = 0
    
    return input_data, target_data
```

## 批处理策略 {#批处理策略}

### 📦 DataLoader配置 {#dataloader配置}

```python
def get_data_loaders(config):
    """创建数据加载器"""
    
    # 创建数据集
    dataset = VIVDataset(
        data_path=config['data']['path'],
        crop_size=config['data'].get('crop_size'),
        use_augmentation=config['data'].get('use_augmentation', False)
    )
    
    # 数据分割
    train_size = int(0.7 * len(dataset))
    valid_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - valid_size
    
    train_dataset, valid_dataset, test_dataset = random_split(
        dataset, [train_size, valid_size, test_size]
    )
    
    # 创建数据加载器
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

### ⚡ 动态批大小 {#动态批大小}

```python
class DynamicBatchSampler:
    """动态批大小采样器"""
    
    def __init__(self, dataset, max_batch_size, max_memory_gb=8):
        self.dataset = dataset
        self.max_batch_size = max_batch_size
        self.max_memory_gb = max_memory_gb
        self.current_batch_size = max_batch_size
    
    def adjust_batch_size(self):
        """根据GPU内存使用调整批大小"""
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
            
            # 动态调整批大小
            self.adjust_batch_size()
```

## 内存优化 {#内存优化}

### 💾 内存管理策略 {#内存管理策略}

```python
class MemoryOptimizedDataset(Dataset):
    """内存优化的数据集"""
    
    def __init__(self, data_path, cache_size=1000):
        self.data_path = data_path
        self.cache_size = cache_size
        self.cache = {}
        self.access_count = {}
        
        # 只加载元数据
        self.metadata = self._load_metadata()
    
    def __getitem__(self, idx):
        # 检查缓存
        if idx in self.cache:
            self.access_count[idx] += 1
            return self.cache[idx]
        
        # 加载数据
        data = self._load_single_item(idx)
        
        # 更新缓存
        self._update_cache(idx, data)
        
        return data
    
    def _update_cache(self, idx, data):
        """更新缓存，使用LRU策略"""
        if len(self.cache) >= self.cache_size:
            # 移除最少使用的项
            lru_idx = min(self.access_count, key=self.access_count.get)
            del self.cache[lru_idx]
            del self.access_count[lru_idx]
        
        self.cache[idx] = data
        self.access_count[idx] = 1
```

### 🔄 预取策略 {#预取策略}

```python
class PrefetchDataLoader:
    """预取数据加载器"""
    
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

## 分布式处理 {#分布式处理}

### 🌐 分布式数据加载 {#分布式数据加载}

```python
def setup_distributed_data(config, rank, world_size):
    """设置分布式数据加载"""
    
    # 创建分布式采样器
    dataset = VIVDataset(config['data']['path'])
    sampler = DistributedSampler(
        dataset, 
        num_replicas=world_size, 
        rank=rank,
        shuffle=True
    )
    
    # 创建数据加载器
    dataloader = DataLoader(
        dataset,
        batch_size=config['data']['batch_size'] // world_size,
        sampler=sampler,
        num_workers=config['data'].get('num_workers', 4),
        pin_memory=True
    )
    
    return dataloader, sampler
```

### 📊 数据并行策略 {#数据并行策略}

```python
class DataParallelStrategy:
    """数据并行策略"""
    
    @staticmethod
    def split_batch(batch, num_gpus):
        """将批次分割到多个GPU"""
        input_data, target_data = batch
        batch_size = input_data.shape[0]
        
        # 计算每个GPU的批大小
        per_gpu_batch_size = batch_size // num_gpus
        
        # 分割数据
        input_splits = torch.split(input_data, per_gpu_batch_size)
        target_splits = torch.split(target_data, per_gpu_batch_size)
        
        return list(zip(input_splits, target_splits))
    
    @staticmethod
    def gather_results(results):
        """收集多GPU结果"""
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

## 🔧 配置示例 {#配置示例}

### 📝 完整数据配置 {#完整数据配置}

```yaml
data:
  # 基础配置
  path: "data/viv_dataset.pt"
  batch_size: 128
  num_workers: 4
  pin_memory: true
  
  # 预处理配置
  normalize: true
  crop_size: [400]
  pad_size: null
  
  # 数据增强配置
  use_augmentation: true
  augmentation:
    noise_level: 0.01
    time_shift_range: 3
    amplitude_scale_range: [0.8, 1.2]
    frequency_mask_ratio: 0.1
  
  # 内存优化配置
  cache_size: 1000
  prefetch_factor: 2
  max_memory_gb: 8
  
  # 分布式配置
  distributed: false
  world_size: 1
  rank: 0
```

---

**💡 提示**: 数据处理是深度学习项目的基础，合理的数据流程设计能够显著提升训练效率和模型性能。

---

*需要帮助？查看 [FAQ](faq) 或 [故障排除](troubleshooting) 页面。*
