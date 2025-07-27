# 自定义数据集快速开始指南

本指南将帮助您快速开始使用 VIVTransformer 的新自定义数据集功能。

## 🚀 快速开始

### 1. 准备数据

确保您的数据文件符合以下格式之一：
- **PyTorch格式** (`.pt`): 推荐格式，加载速度最快
- **HDF5格式** (`.h5`, `.hdf5`): 适合大型数据集
- **Pickle格式** (`.pkl`): Python原生序列化格式
- **NumPy格式** (`.npy`, `.npz`): 适合数值数组数据

### 2. 数据结构要求

您的数据文件应包含以下键：
```python
{
    'input_data': torch.Tensor,    # 形状: [N, T, spatial_dims]
    'target_data': torch.Tensor,   # 形状: [N, T, spatial_dims]
    'time_steps': torch.Tensor     # 形状: [N, T] (可选)
}
```

其中：
- `N`: 样本数量
- `T`: 时间步数
- `spatial_dims`: 空间维度（如 128*128=16384）

### 3. 创建配置文件

复制并修改 `configs/custom_dataset_config.yaml`：

```yaml
# 启用自定义数据集
data:
  use_custom_dataset: true
  use_pdebench: false

# 自定义数据集配置
custom_dataset:
  data_root: "path/to/your/data"  # 修改为您的数据路径
  dataset_type: "flow_field"      # 数据集类型标识
  file_paths:
    - "your_data_file.pt"         # 修改为您的数据文件名
  
  # 数据参数
  spatial_resolution: [128, 128]  # 修改为您的空间分辨率
  sequence_length: 50             # 修改为您的序列长度
  
  # 数据键映射
  data_keys:
    input: "input_data"           # 输入数据的键名
    target: "target_data"         # 目标数据的键名
    time: "time_steps"            # 时间步的键名（可选）
```

### 4. 运行测试

验证数据集配置是否正确：

```bash
python test_custom_dataset.py
```

### 5. 开始训练

使用示例脚本开始训练：

```bash
python examples/custom_dataset_example.py
```

或者使用主训练脚本：

```bash
python main.py --config configs/custom_dataset_config.yaml
```

## 📋 配置参数详解

### 必需参数

| 参数 | 描述 | 示例 |
|------|------|------|
| `data_root` | 数据文件根目录 | `"./data/custom"` |
| `file_paths` | 数据文件路径列表 | `["data1.pt", "data2.pt"]` |
| `spatial_resolution` | 空间分辨率 | `[128, 128]` |
| `sequence_length` | 时间序列长度 | `50` |

### 可选参数

| 参数 | 描述 | 默认值 |
|------|------|--------|
| `normalization_method` | 归一化方法 | `"minmax"` |
| `center_crop_size` | 中心裁剪尺寸 | `null` |
| `data_split_ratios` | 数据分割比例 | `[0.7, 0.2, 0.1]` |
| `input_dim` | 输入维度 | 自动计算 |
| `output_dim` | 输出维度 | 自动计算 |

## 🔧 高级用法

### 多文件数据集

```yaml
custom_dataset:
  file_paths:
    - "train_data.pt"
    - "validation_data.pt"
    - "test_data.pt"
```

### 自定义数据键

```yaml
custom_dataset:
  data_keys:
    input: "velocity_field"     # 自定义输入数据键名
    target: "pressure_field"    # 自定义目标数据键名
    time: "timestamps"          # 自定义时间键名
```

### 数据增强

```yaml
training:
  data_augmentation:
    enabled: true
    rotation: true
    flip: true
    noise_level: 0.01
```

## 🐛 常见问题

### Q: 数据加载失败
**A:** 检查以下项目：
1. 文件路径是否正确
2. 数据文件是否包含必需的键
3. 数据形状是否符合要求

### Q: 内存不足
**A:** 尝试以下解决方案：
1. 减小批次大小 (`batch_size`)
2. 启用中心裁剪 (`center_crop_size`)
3. 使用数据增强减少数据量

### Q: 训练速度慢
**A:** 优化建议：
1. 使用 PyTorch 格式 (`.pt`) 文件
2. 增加 `num_workers` 参数
3. 使用 GPU 训练

## 📊 性能优化

### 数据格式选择
- **最快**: PyTorch (`.pt`)
- **平衡**: HDF5 (`.h5`)
- **兼容**: NumPy (`.npy`)

### 批次大小建议
- **小数据集**: 32-64
- **中等数据集**: 16-32
- **大数据集**: 8-16

### 内存优化
```yaml
training:
  batch_size: 16              # 根据GPU内存调整
  num_workers: 4              # 根据CPU核心数调整
  pin_memory: true            # GPU训练时启用
```

## 🔗 相关文档

- [详细集成指南](Custom_Dataset_Integration_Guide.md)
- [配置文件示例](configs/custom_dataset_config.yaml)
- [测试脚本](test_custom_dataset.py)
- [训练示例](examples/custom_dataset_example.py)

## 💡 提示

1. **数据预处理**: 建议预先将数据转换为 PyTorch 格式以获得最佳性能
2. **配置验证**: 使用测试脚本验证配置正确性
3. **渐进训练**: 从小数据集开始，逐步扩展到完整数据集
4. **监控资源**: 注意内存和GPU使用情况，适时调整参数

---

🎉 **恭喜！** 您现在可以使用自定义数据集训练 VIVTransformer 模型了！