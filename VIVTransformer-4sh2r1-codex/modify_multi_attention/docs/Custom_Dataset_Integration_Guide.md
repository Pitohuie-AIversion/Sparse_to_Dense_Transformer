# 自定义数据集集成指南

本指南详细说明如何将新的数据集集成到VIVTransformer系统中。

## 概述

自定义数据集适配器提供了一个灵活的框架，支持多种数据格式和预处理选项，特别针对科学计算数据进行了优化。

## 支持的数据格式

- **PyTorch格式** (`.pt`): 标准的PyTorch张量文件
- **HDF5格式** (`.h5`, `.hdf5`): 科学计算常用的层次化数据格式
- **Pickle格式** (`.pkl`): Python对象序列化格式
- **NumPy格式** (`.npy`, `.npz`): NumPy数组格式

## 快速开始

### 1. 准备数据文件

确保你的数据文件符合以下要求之一：

#### PyTorch格式 (.pt)
```python
# 数据结构示例
data = {
    'pressure': torch.tensor(...),  # 主要数据 [N, T, H, W] 或 [N, T, H, W, C]
    'metadata': {...}               # 可选的元数据
}
torch.save(data, 'your_data.pt')
```

#### HDF5格式 (.h5/.hdf5)
```python
import h5py

with h5py.File('your_data.h5', 'w') as f:
    f.create_dataset('data', data=your_array)  # [N, T, H, W] 或 [N, T, H, W, C]
    f.attrs['description'] = 'Your dataset description'
```

### 2. 创建配置文件

复制并修改 `configs/custom_dataset_config.yaml`：

```yaml
# Data settings
data:
  use_custom_dataset: true  # 启用自定义数据集
  use_pdebench: false       # 禁用PDEBench
  batch_size: 64
  normalize: true

# 自定义数据集配置
custom_dataset:
  data_root: "/path/to/your/data"  # 数据根目录
  current_dataset: "your_dataset"  # 当前使用的数据集类型
  
  dataset_configs:
    your_dataset:
      data_file: "your_data.pt"           # 数据文件名
      spatial_resolution: [200, 200]      # 空间分辨率
      input_size: 128                     # 输入区域大小（中心裁剪）
      sequence_length: 48                 # 序列长度
      normalization_method: "standard"    # 归一化方法
      data_key: "pressure"                # 数据键名
      target_key: null                    # 目标数据键名
      split_ratios: [0.7, 0.15, 0.15]    # 数据分割比例
```

### 3. 运行训练

```bash
# 使用自定义数据集配置运行训练
python main.py --config configs/custom_dataset_config.yaml
```

## 详细配置说明

### 数据集配置参数

| 参数 | 类型 | 说明 | 默认值 |
|------|------|------|--------|
| `data_file` | str | 数据文件名 | 必需 |
| `spatial_resolution` | List[int] | 空间分辨率 [H, W] | [128, 128] |
| `input_size` | int/null | 输入区域大小（中心裁剪） | null（不裁剪） |
| `sequence_length` | int | 序列长度 | 48 |
| `normalization_method` | str | 归一化方法 | "standard" |
| `data_key` | str/null | 数据文件中的数据键名 | null（自动检测） |
| `target_key` | str/null | 目标数据键名 | null（时间序列预测） |
| `split_ratios` | List[float] | 数据分割比例 | [0.7, 0.15, 0.15] |

### 归一化方法

- **"standard"**: 标准化 (mean=0, std=1)
- **"minmax"**: 最小-最大归一化 (范围[0,1])
- **"none"**: 不进行归一化

### 数据形状要求

支持的数据形状：
- `[N, T, H, W]`: N个样本，T个时间步，H×W空间分辨率
- `[N, T, H, W, C]`: 包含C个通道的数据
- `[N, H, W]`: 单时间步数据（会自动扩展）

## 高级用法

### 1. 多数据集配置

可以在同一个配置文件中定义多个数据集：

```yaml
custom_dataset:
  current_dataset: "dataset_a"  # 当前使用的数据集
  
  dataset_configs:
    dataset_a:
      data_file: "data_a.pt"
      # ... 其他配置
    
    dataset_b:
      data_file: "data_b.h5"
      # ... 其他配置
```

### 2. 自定义数据键

对于复杂的数据文件结构，可以指定具体的数据键：

```yaml
dataset_configs:
  complex_dataset:
    data_file: "complex_data.pt"
    data_key: "velocity_field"     # 输入数据键
    target_key: "pressure_field"   # 目标数据键
```

### 3. 不同文件格式示例

#### HDF5格式
```yaml
hdf5_dataset:
  data_file: "simulation.h5"
  data_key: "velocity"
  target_key: "pressure"
  normalization_method: "minmax"
```

#### NumPy格式
```yaml
numpy_dataset:
  data_file: "data.npz"
  data_key: "input_data"
  normalization_method: "standard"
```

## 测试和验证

### 运行测试脚本

```bash
# 测试自定义数据集适配器
python test_custom_dataset.py
```

测试脚本会验证：
- 数据集类的基本功能
- 数据加载器的批次处理
- 自适应加载器的集成
- 不同文件格式的支持

### 调试数据加载

```python
from data.custom_dataset_adapter import CustomDataset

# 创建数据集实例
dataset = CustomDataset(
    data_path="your_data.pt",
    dataset_type="your_type",
    split="train"
)

# 检查数据集信息
info = dataset.get_data_info()
print(f"数据集信息: {info}")

# 检查单个样本
inputs, targets, time_steps = dataset[0]
print(f"输入形状: {inputs.shape}")
print(f"目标形状: {targets.shape}")
```

## 常见问题

### Q1: 数据维度不匹配

**问题**: `ValueError: 数据维度不足，期望至少3维`

**解决**: 确保数据至少是3维的 `[N, H, W]` 或更高维度。

### Q2: 找不到数据键

**问题**: `KeyError: 无法确定数据键`

**解决**: 在配置中明确指定 `data_key` 参数。

### Q3: 内存不足

**问题**: 加载大数据集时内存不足

**解决**: 
- 减少 `batch_size`
- 设置 `num_workers: 0`
- 使用 `pin_memory: false`

### Q4: 序列长度不匹配

**问题**: 数据的时间步数与配置不匹配

**解决**: 调整配置中的 `sequence_length` 参数，或者数据会自动填充/截取。

## 性能优化建议

1. **数据预处理**: 预先将数据转换为合适的格式和分辨率
2. **批次大小**: 根据GPU内存调整批次大小
3. **工作进程**: 在Windows上建议设置 `num_workers: 0`
4. **数据格式**: HDF5格式通常比Pickle格式加载更快
5. **归一化**: 预先计算归一化参数可以加速训练

## 扩展开发

### 添加新的数据格式支持

在 `CustomDataset` 类中添加新的加载方法：

```python
def _load_your_format_data(self):
    """加载你的自定义格式数据"""
    # 实现你的数据加载逻辑
    pass
```

然后在 `_load_data` 方法中添加格式检测：

```python
def _load_data(self):
    suffix = self.data_path.suffix.lower()
    
    if suffix == '.your_ext':
        self._load_your_format_data()
    # ... 其他格式
```

### 自定义预处理

可以继承 `CustomDataset` 类来实现特定的预处理逻辑：

```python
class YourCustomDataset(CustomDataset):
    def _preprocess_data(self):
        super()._preprocess_data()
        # 添加你的自定义预处理逻辑
        pass
```

## 总结

自定义数据集适配器提供了一个强大而灵活的框架，可以轻松集成各种格式的科学计算数据。通过合适的配置，你可以快速将新的数据集集成到VIVTransformer系统中，并开始训练实验。

如果遇到问题，请参考测试脚本和示例配置，或者查看日志输出获取详细的错误信息。