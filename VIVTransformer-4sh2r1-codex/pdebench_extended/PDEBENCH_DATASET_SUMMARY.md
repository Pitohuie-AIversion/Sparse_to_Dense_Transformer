# PDEBench数据集适配器总结

## 概述

PDEBench数据集适配器是一个专门用于处理偏微分方程（PDE）基准数据集的Python模块。该适配器支持多种PDE类型的数据加载、预处理和批次处理，为深度学习模型提供统一的数据接口。

## 支持的数据集类型

### 1. PDE方程类型
- **Navier-Stokes方程** (`ns_incom`): 不可压缩流体动力学
- **Darcy流方程** (`darcy_flow`): 多孔介质中的流体流动
- **浅水方程** (`shallow_water`): 浅水波动力学
- **对流方程** (`advection`): 物质传输方程
- **Burgers方程** (`burgers`): 非线性偏微分方程
- **反应扩散方程** (`reaction_diffusion`): 化学反应和扩散过程

### 2. 数据格式支持
- **HDF5格式** (`.h5`, `.hdf5`): 科学计算标准格式
- **PyTorch格式** (`.pt`): PyTorch张量序列化格式

## 数据结构

### 输入数据维度
- **原始数据形状**: `[N, T, H, W]` 或 `[N, T, H, W, C]`
  - `N`: 样本数量
  - `T`: 时间步数
  - `H`: 空间高度
  - `W`: 空间宽度
  - `C`: 通道数（可选）

### 输出数据维度
- **处理后形状**: `[B, T-1, H*W*C]`
  - `B`: 批次大小
  - `T-1`: 序列长度（输入序列长度减1）
  - `H*W*C`: 展平的空间维度

## 数据处理流程

### 1. 数据加载
```python
# 支持的数据键名
- 'data': 主要数据存储键
- 'tensor': 张量数据键
- 自动检测数值数据键（排除坐标和参数键）
```

### 2. 数据预处理
- **维度检查**: 确保数据至少为4维
- **通道处理**: 自动添加通道维度（如果缺失）
- **分辨率适配**: 自动检测并适配空间分辨率

### 3. 序列截取策略

#### 训练模式
- **随机截取**: 从完整时间序列中随机选择起始点
- **动态长度**: 根据指定的`sequence_length`截取子序列

#### 验证/测试模式
- **固定截取**: 从序列开始位置截取
- **一致性保证**: 确保验证和测试结果的可重复性

#### 序列不足处理
- **边缘填充**: 使用`np.pad`的`edge`模式填充不足的时间步

### 4. 数据分割
- **训练集**: 70%的数据
- **验证集**: 15%的数据
- **测试集**: 15%的数据

### 5. 归一化处理
- **Z-score标准化**: `(x - mean) / std`
- **训练集统计**: 使用训练集计算均值和标准差
- **全局应用**: 将训练集统计量应用到所有分割

## 核心类和方法

### PDEBenchDataset类
```python
class PDEBenchDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        pde_type: str = "ns_incom",
        split: str = "train",
        sequence_length: int = 49,
        spatial_resolution: Optional[List[int]] = None,
        normalize: bool = True,
        transform: Optional[Any] = None,
        target_transform: Optional[Any] = None
    )
```

#### 关键方法
- `_load_data()`: 加载HDF5或PyTorch格式数据
- `_preprocess_data()`: 数据预处理和维度检查
- `_create_splits()`: 创建训练/验证/测试分割
- `_prepare_sequence()`: 序列截取和准备
- `_reshape_spatial_data()`: 空间维度展平
- `get_data_info()`: 获取数据集详细信息

### PDEBenchDataLoader类
```python
class PDEBenchDataLoader(DataLoader):
    def __init__(
        self,
        dataset: PDEBenchDataset,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 4,
        pin_memory: bool = True
    )
```

#### 特殊功能
- **自定义批次整理**: `_collate_fn`方法处理变长序列
- **数据信息集成**: 自动收集和提供数据集元信息

## 便捷函数

### 创建数据集
```python
train_dataset, val_dataset, test_dataset = create_pdebench_datasets(
    data_path="path/to/data.h5",
    pde_type="darcy_flow",
    sequence_length=49,
    spatial_resolution=[64, 64],
    normalize=True
)
```

### 创建数据加载器
```python
train_loader, val_loader, test_loader = create_pdebench_loaders(
    data_path="path/to/data.h5",
    pde_type="ns_incom",
    batch_size=32,
    sequence_length=49,
    num_workers=4
)
```

## 数据输出格式

### 单个数据项
```python
inputs, targets, time_steps = dataset[0]
# inputs: torch.Tensor [T-1, H*W*C] - 输入序列
# targets: torch.Tensor [T-1, H*W*C] - 目标序列
# time_steps: torch.Tensor [T-1] - 时间步信息
```

### 批次数据
```python
inputs, targets, time_steps = next(iter(dataloader))
# inputs: torch.Tensor [B, T-1, H*W*C] - 批次输入
# targets: torch.Tensor [B, T-1, H*W*C] - 批次目标
# time_steps: torch.Tensor [B, T-1] - 批次时间步
```

## 配置参数

### 关键参数说明
- `sequence_length`: 序列长度，控制时间维度的截取长度
- `spatial_resolution`: 空间分辨率，自动检测或手动指定
- `normalize`: 是否进行数据归一化
- `pde_type`: PDE类型标识符
- `split`: 数据分割类型（train/val/test）

### 性能参数
- `batch_size`: 批次大小，影响内存使用和训练效率
- `num_workers`: 数据加载工作进程数
- `pin_memory`: 是否使用固定内存（GPU训练推荐）

## 特殊处理机制

### 1. 稳态vs时间演化
- **稳态问题**（如Darcy Flow）: 输入和目标为同一时间步
- **时间演化问题**: 输入为前n-1步，目标为后n-1步

### 2. 元数据处理
- **坐标信息**: 自动提取x、y、t坐标
- **参数信息**: 保存PDE相关参数（如雷诺数）
- **属性保存**: 维护原始数据的元属性

### 3. 错误处理
- **文件验证**: 检查数据文件存在性
- **格式验证**: 支持的文件格式检查
- **维度验证**: 数据维度合理性检查
- **键名自适应**: 自动检测HDF5数据键名

## 使用示例

### 基本使用
```python
from pdebench_adapter import create_pdebench_loaders

# 创建数据加载器
train_loader, val_loader, test_loader = create_pdebench_loaders(
    data_path="data/darcy_flow_beta1.0_train.h5",
    pde_type="darcy_flow",
    batch_size=32,
    sequence_length=1,  # Darcy Flow为稳态问题
    spatial_resolution=[128, 128]
)

# 获取数据信息
info = train_loader.get_data_info()
print(f"输入维度: {info['input_dim']}")
print(f"样本数量: {info['n_samples']}")

# 训练循环
for inputs, targets, time_steps in train_loader:
    # inputs: [B, T, D], targets: [B, T, D]
    # 模型训练代码
    pass
```

### 高级配置
```python
from pdebench_adapter import PDEBenchDataset, PDEBenchDataLoader
import torch.nn as nn

# 自定义变换
transform = nn.Sequential(
    nn.Dropout(0.1),
    nn.LayerNorm(16384)  # 128*128
)

# 创建自定义数据集
dataset = PDEBenchDataset(
    data_path="data/ns_incom_train.h5",
    pde_type="ns_incom",
    split="train",
    sequence_length=49,
    normalize=True,
    transform=transform
)

# 创建数据加载器
loader = PDEBenchDataLoader(
    dataset=dataset,
    batch_size=16,
    shuffle=True,
    num_workers=8,
    pin_memory=True
)
```

## 与统一适配器的集成

PDEBench适配器与之前创建的统一数据适配器完全兼容，可以通过统一接口调用：

```python
from unified_adapter import UnifiedDataAdapter

# 自动检测并使用PDEBench适配器
adapter = UnifiedDataAdapter()
loader = adapter.create_dataloader(
    data_path="data/darcy_flow.h5",
    batch_size=32
)
```

## 总结

PDEBench数据集适配器提供了一个完整、灵活且高效的解决方案，用于处理各种偏微分方程的数值模拟数据。其主要优势包括：

1. **多格式支持**: HDF5和PyTorch格式
2. **多PDE类型**: 支持6种主要PDE类型
3. **智能截取**: 灵活的序列截取策略
4. **自动适配**: 空间分辨率和数据格式自动检测
5. **标准化处理**: 完整的数据预处理流程
6. **高性能**: 多进程数据加载和内存优化
7. **易于使用**: 简洁的API和便捷函数
8. **可扩展性**: 支持自定义变换和配置

该适配器为PDE相关的深度学习研究提供了坚实的数据基础设施支持。