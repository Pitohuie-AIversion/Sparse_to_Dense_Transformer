# 统一数据适配器 (Unified Data Adapter)

## 概述

统一数据适配器是一个智能的数据处理组件，能够自动检测和处理不同类型的数据集格式，为深度学习模型提供统一的数据接口。

## 支持的数据格式

### 1. 压力场数据集 (20x20→200x200)
- **格式**: PyTorch `.pt` 文件
- **输入维度**: 400 (20×20)
- **输出维度**: 40000 (200×200)
- **用途**: 压力场预测任务
- **数据结构**: 包含 `in_pressure` 和 `pressure` 键的字典

### 2. PDEBench数据集
- **格式**: HDF5 `.h5`/`.hdf5` 文件
- **支持的PDE**: Darcy Flow, Navier-Stokes, Shallow Water等
- **可配置分辨率**: 32×32, 64×64, 128×128等
- **用途**: 偏微分方程求解任务

## 核心功能

### 自动数据类型检测
```python
from data.unified_adapter import UnifiedDataAdapter

# 自动检测数据类型
adapter = UnifiedDataAdapter(config)
print(f"检测到数据类型: {adapter.data_type}")
```

### 统一数据集创建
```python
from data.unified_adapter import create_unified_datasets

# 创建数据集（自动适配格式）
datasets = create_unified_datasets(config)
train_loader = datasets['train']
val_loader = datasets['val']
test_loader = datasets['test']
info = datasets['info']
```

## 配置示例

### 压力场数据集配置
```python
config = {
    'data': {
        'path': 'path/to/merged_all_pressures_separated_normalized.pt',
        'batch_size': 32,
        'normalize': True,
        'num_workers': 4,
        'pin_memory': True
    }
}
```

### PDEBench数据集配置
```python
config = {
    'data': {
        'path': 'path/to/2D_DarcyFlow_beta0.01_Train.hdf5',
        'batch_size': 32,
        'normalize': True
    },
    'current_pde': 'darcy_flow',
    'pdebench': {
        'pde_configs': {
            'darcy_flow': {
                'sequence_length': 1,
                'spatial_resolution': [128, 128],
                'input_dim': 16384,
                'output_dim': 16384
            }
        }
    }
}
```

## 使用方法

### 1. 基本使用
```python
# 导入模块
from data.unified_adapter import create_unified_datasets
from utils.config import load_config

# 加载配置
config = load_config('configs/your_config.yaml')

# 创建数据集
datasets = create_unified_datasets(config)

# 获取数据信息
info = datasets['info']
print(f"数据类型: {info.get('data_type')}")
print(f"输入维度: {info.get('input_dim')}")
print(f"输出维度: {info.get('output_dim')}")

# 使用数据加载器
for batch in datasets['train']:
    inputs, targets = batch[0], batch[1]
    # 进行训练...
```

### 2. 在训练循环中使用
```python
def train_model(config):
    # 创建数据集
    datasets = create_unified_datasets(config)
    
    # 获取数据信息
    info = datasets['info']
    input_dim = info['input_dim']
    output_dim = info['output_dim']
    
    # 创建模型
    model = create_model(input_dim, output_dim)
    
    # 训练循环
    for epoch in range(num_epochs):
        for batch in datasets['train']:
            # 处理批次数据
            if isinstance(batch, (list, tuple)):
                inputs, targets = batch[0], batch[1]
            elif isinstance(batch, dict):
                inputs = batch['input']
                targets = batch['target']
            
            # 训练步骤
            loss = train_step(model, inputs, targets)
```

## 数据信息结构

### 压力场数据集信息
```python
info = {
    'data_type': 'pressure_field',
    'input_dim': 400,
    'output_dim': 40000,
    'input_shape': [20, 20],
    'output_shape': [200, 200],
    'total_samples': 880,
    'train_samples': 616,
    'val_samples': 132,
    'test_samples': 132
}
```

### PDEBench数据集信息
```python
info = {
    'pde_type': 'darcy_flow',
    'spatial_resolution': [128, 128],
    'input_dim': 16384,
    'sequence_length': 1,
    'train_samples': 7000,
    'val_samples': 1500,
    'test_samples': 1500
}
```

## 优势特点

1. **自动检测**: 无需手动指定数据格式，自动识别数据类型
2. **统一接口**: 不同数据格式使用相同的API
3. **灵活配置**: 支持各种数据处理参数配置
4. **高效加载**: 优化的数据加载和预处理流程
5. **易于扩展**: 可轻松添加新的数据格式支持

## 测试和验证

运行测试脚本验证功能：
```bash
# 测试统一适配器
python test_unified_adapter.py

# 运行使用示例
python example_unified_usage.py
```

## 注意事项

1. 确保数据文件路径正确且文件存在
2. 对于PDEBench数据，需要正确配置PDE类型和参数
3. 根据硬件资源调整批次大小和工作进程数
4. 大型数据集建议启用数据归一化

## 扩展支持

要添加新的数据格式支持，请：
1. 在 `_detect_data_type()` 方法中添加检测逻辑
2. 实现相应的 `_create_xxx_datasets()` 方法
3. 更新 `create_datasets()` 方法的分发逻辑

---

通过统一数据适配器，您可以轻松处理不同格式的数据集，专注于模型开发而无需担心数据处理的复杂性。