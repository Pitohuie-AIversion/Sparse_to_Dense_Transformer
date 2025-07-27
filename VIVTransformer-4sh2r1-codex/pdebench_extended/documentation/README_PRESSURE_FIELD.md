# 压力场重建系统使用指南

本文档介绍如何使用新开发的压力场数据适配器来处理您的20x20→200x200压力场重建任务。

## 📋 目录

- [系统概述](#系统概述)
- [快速开始](#快速开始)
- [详细使用说明](#详细使用说明)
- [配置文件说明](#配置文件说明)
- [API参考](#api参考)
- [示例代码](#示例代码)
- [常见问题](#常见问题)
- [性能优化](#性能优化)

## 🎯 系统概述

### 核心功能

本系统专门为您的压力场重建任务设计，具有以下特点：

- **输入**: 20x20压力场数据（从200x200中心截取）
- **输出**: 200x200完整压力场数据
- **任务类型**: 空间超分辨率重建
- **上采样因子**: 10倍（20→200）
- **物理领域**: 流体动力学

### 系统架构

```
原始数据(.pt) → 压力场适配器 → 统一数据接口 → 训练/推理
     ↓              ↓              ↓           ↓
  20x20输入      数据预处理      标准化接口    模型训练
  200x200输出    数据增强        批处理       结果评估
```

### 主要组件

1. **压力场适配器** (`data/pressure_field_adapter.py`)
   - 专门处理20x20→200x200数据格式
   - 支持多种归一化方法
   - 内置数据增强功能
   - 物理约束验证

2. **统一数据适配器** (`data/unified_adapter.py`)
   - 自动检测数据格式
   - 无缝集成到PDEBench系统
   - 支持多种数据源

3. **训练脚本** (`train_pressure_field.py`)
   - 完整的训练流程
   - Transformer模型实现
   - 多种损失函数
   - 实时监控和可视化

## 🚀 快速开始

### 1. 环境准备

确保您已安装必要的依赖：

```bash
pip install torch torchvision torchaudio
pip install numpy matplotlib scipy scikit-learn
pip install tensorboard tqdm pyyaml
pip install h5py  # 如果需要处理PDEBench数据
```

### 2. 数据准备

确保您的数据文件格式正确：

```python
# 您的数据文件应该包含：
data = {
    'in_pressure': torch.Tensor,    # 形状: [N, 400] (20x20展平)
    'pressure': torch.Tensor,       # 形状: [N, 40000] (200x200展平)
    'time_steps': torch.Tensor      # 形状: [N]
}
```

### 3. 运行示例

```bash
# 1. 运行数据处理示例
python examples/pressure_field_example.py

# 2. 开始训练
python train_pressure_field.py --config configs/pressure_field_training.yaml --data_path "your_data_path.pt"
```

## 📖 详细使用说明

### 数据加载和预处理

#### 基本用法

```python
from data.pressure_field_adapter import create_pressure_field_datasets

# 创建数据集
train_loader, val_loader, test_loader = create_pressure_field_datasets(
    data_path="path/to/your/data.pt",
    batch_size=16,
    normalize=True,
    normalize_method='minmax',
    augmentation=True,
    num_workers=4
)

# 检查数据
for inputs, targets, time_steps in train_loader:
    print(f"输入形状: {inputs.shape}")      # [16, 400]
    print(f"目标形状: {targets.shape}")     # [16, 40000]
    print(f"时间步形状: {time_steps.shape}") # [16]
    break
```

#### 高级用法

```python
from data.pressure_field_adapter import PressureFieldDataset

# 创建自定义数据集
dataset = PressureFieldDataset(
    data_path="path/to/your/data.pt",
    split='train',
    normalize=True,
    normalize_method='zscore',
    augmentation=True,
    physics_validation=True
)

# 获取数据信息
info = dataset.get_data_info()
print(f"数据类型: {info['data_type']}")
print(f"上采样因子: {info['upsampling_factor']}")

# 可视化样本
dataset.visualize_sample(0, save_path="sample_0.png")
```

### 数据分析

```python
from data.pressure_field_adapter import analyze_pressure_field_data

# 分析数据特性
analysis = analyze_pressure_field_data("path/to/your/data.pt")

print("数据统计信息:")
print(f"输入数据范围: [{analysis['input_statistics']['min']:.4f}, {analysis['input_statistics']['max']:.4f}]")
print(f"输出数据范围: [{analysis['output_statistics']['min']:.4f}, {analysis['output_statistics']['max']:.4f}]")
print(f"总样本数: {analysis['total_samples']}")
```

### 统一数据接口

```python
from data.unified_adapter import UnifiedDataAdapter

# 配置
config = {
    'data': {
        'path': "path/to/your/data.pt",
        'batch_size': 16,
        'normalize': True,
        'num_workers': 4
    }
}

# 创建适配器
adapter = UnifiedDataAdapter(config)

# 自动检测数据类型
data_type = adapter.detect_data_type(config['data']['path'])
print(f"检测到的数据类型: {data_type}")  # 应该输出: pressure_field

# 创建数据集
datasets = adapter.create_datasets()
```

## ⚙️ 配置文件说明

### 主要配置项

#### 数据配置 (`data`)

```yaml
data:
  path: "path/to/your/data.pt"  # 数据文件路径
  input_dim: 400                # 输入维度 (20x20)
  output_dim: 40000             # 输出维度 (200x200)
  batch_size: 16                # 批次大小
  normalize: true               # 是否归一化
  normalize_method: "minmax"    # 归一化方法
  
  # 数据增强
  augmentation:
    enabled: true
    flip_horizontal: 0.5        # 水平翻转概率
    flip_vertical: 0.5          # 垂直翻转概率
    rotation_angle: 15          # 最大旋转角度
    noise_prob: 0.3             # 添加噪声概率
```

#### 模型配置 (`model`)

```yaml
model:
  type: "transformer"
  transformer:
    d_model: 512                # 模型维度
    nhead: 8                    # 注意力头数
    num_encoder_layers: 6       # 编码器层数
    dropout: 0.1                # Dropout率
    
    # 输出投影
    output_projection:
      type: "mlp"
      hidden_dims: [1024, 2048, 40000]
```

#### 训练配置 (`training`)

```yaml
training:
  epochs: 200
  learning_rate: 1e-4
  optimizer:
    type: "adamw"
    betas: [0.9, 0.999]
  
  scheduler:
    type: "cosine_annealing"
    T_max: 200
    eta_min: 1e-6
  
  early_stopping:
    enabled: true
    patience: 20
```

#### 损失函数配置 (`loss`)

```yaml
loss:
  primary:
    type: "mse"
    weight: 1.0
  
  auxiliary:
    gradient:
      enabled: true
      weight: 0.05              # 梯度损失权重
    
    frequency:
      enabled: true
      weight: 0.02              # 频域损失权重
    
    physics:
      enabled: true
      weight: 0.01              # 物理约束损失权重
```

## 🔧 API参考

### PressureFieldDataset

```python
class PressureFieldDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        split: str = 'train',           # 'train', 'val', 'test'
        normalize: bool = True,
        normalize_method: str = 'minmax', # 'minmax', 'zscore'
        augmentation: bool = False,
        physics_validation: bool = True
    )
    
    def get_data_info(self) -> Dict[str, Any]
    def visualize_sample(self, idx: int, save_path: Optional[str] = None)
```

### 便捷函数

```python
# 创建数据集
create_pressure_field_datasets(
    data_path: str,
    batch_size: int = 32,
    normalize: bool = True,
    normalize_method: str = 'minmax',
    augmentation: bool = False,
    num_workers: int = 4,
    pin_memory: bool = True
) -> Tuple[PressureFieldDataLoader, PressureFieldDataLoader, PressureFieldDataLoader]

# 分析数据
analyze_pressure_field_data(data_path: str) -> Dict[str, Any]
```

## 💡 示例代码

### 完整训练示例

```python
import torch
import yaml
from data.pressure_field_adapter import create_pressure_field_datasets
from train_pressure_field import PressureFieldTrainer

# 1. 加载配置
with open('configs/pressure_field_training.yaml', 'r') as f:
    config = yaml.safe_load(f)

# 2. 修改数据路径
config['data']['path'] = "path/to/your/data.pt"

# 3. 创建训练器
trainer = PressureFieldTrainer(config)

# 4. 开始训练
trainer.train()
```

### 数据预处理示例

```python
from data.pressure_field_adapter import PressureFieldDataset
import matplotlib.pyplot as plt

# 创建数据集
dataset = PressureFieldDataset(
    data_path="your_data.pt",
    split='train',
    normalize=True,
    augmentation=True
)

# 比较原始和增强数据
fig, axes = plt.subplots(2, 2, figsize=(10, 8))

for i in range(2):
    input_tensor, output_tensor, time_step = dataset[i]
    
    # 重塑为2D
    input_2d = input_tensor.numpy().reshape(20, 20)
    output_2d = output_tensor.numpy().reshape(200, 200)
    
    # 绘制
    axes[0, i].imshow(input_2d, cmap='coolwarm')
    axes[0, i].set_title(f'输入 {i+1} (20x20)')
    
    axes[1, i].imshow(output_2d, cmap='coolwarm')
    axes[1, i].set_title(f'输出 {i+1} (200x200)')

plt.tight_layout()
plt.savefig('data_samples.png', dpi=300)
plt.show()
```

### 模型推理示例

```python
import torch
from train_pressure_field import PressureFieldTransformer

# 加载训练好的模型
checkpoint = torch.load('checkpoints/pressure_field/best.pth')
config = checkpoint['config']

model = PressureFieldTransformer(config)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# 推理
with torch.no_grad():
    # 假设有一个20x20的输入
    input_data = torch.randn(1, 400)  # 批次大小为1
    output = model(input_data)        # 输出形状: [1, 40000]
    
    # 重塑为2D
    output_2d = output.view(200, 200)
    
    # 可视化结果
    plt.imshow(output_2d.numpy(), cmap='coolwarm')
    plt.title('预测的压力场 (200x200)')
    plt.colorbar()
    plt.show()
```

## ❓ 常见问题

### Q1: 数据文件格式不正确怎么办？

**A**: 确保您的`.pt`文件包含以下键：
- `in_pressure`: 形状为`[N, 400]`的张量
- `pressure`: 形状为`[N, 40000]`的张量  
- `time_steps`: 形状为`[N]`的张量

```python
# 检查数据格式
data = torch.load('your_data.pt')
print("数据键:", list(data.keys()))
print("输入形状:", data['in_pressure'].shape)
print("输出形状:", data['pressure'].shape)
```

### Q2: 内存不足怎么办？

**A**: 尝试以下方法：
1. 减小批次大小
2. 减少工作进程数
3. 关闭数据增强
4. 使用梯度累积

```yaml
data:
  batch_size: 8        # 减小批次大小
  num_workers: 2       # 减少工作进程
  augmentation:
    enabled: false     # 关闭数据增强

training:
  gradient_accumulation_steps: 4  # 梯度累积
```

### Q3: 训练速度太慢怎么办？

**A**: 优化建议：
1. 使用GPU训练
2. 启用混合精度训练
3. 增加工作进程数
4. 使用更大的批次大小

```yaml
global:
  device: "cuda"
  mixed_precision: true

data:
  batch_size: 32
  num_workers: 8
  pin_memory: true
```

### Q4: 如何调整模型大小？

**A**: 修改模型配置：

```yaml
model:
  transformer:
    d_model: 256          # 减小模型维度
    nhead: 4              # 减少注意力头数
    num_encoder_layers: 4 # 减少层数
    dim_feedforward: 1024 # 减小前馈网络维度
```

## 🚀 性能优化

### 训练优化

1. **使用混合精度训练**
```yaml
global:
  mixed_precision: true
```

2. **优化数据加载**
```yaml
data:
  num_workers: 8
  pin_memory: true
  drop_last: true
```

3. **梯度累积**
```yaml
training:
  gradient_accumulation_steps: 4
```

### 模型优化

1. **使用更高效的激活函数**
```yaml
model:
  transformer:
    activation: "gelu"  # 比ReLU更平滑
```

2. **调整学习率调度**
```yaml
training:
  scheduler:
    type: "cosine_annealing"
    T_max: 200
    eta_min: 1e-6
```

### 内存优化

1. **梯度检查点**
```python
# 在模型中启用梯度检查点
from torch.utils.checkpoint import checkpoint

class PressureFieldTransformer(nn.Module):
    def forward(self, x):
        # 使用梯度检查点
        x = checkpoint(self.transformer_encoder, x)
        return x
```

2. **数据类型优化**
```python
# 使用半精度浮点数
model = model.half()
inputs = inputs.half()
```

## 📊 监控和可视化

### TensorBoard监控

```bash
# 启动TensorBoard
tensorboard --logdir=logs/pressure_field
```

### 自定义可视化

```python
import matplotlib.pyplot as plt
from data.pressure_field_adapter import PressureFieldDataset

def visualize_training_progress(model, dataset, device):
    model.eval()
    with torch.no_grad():
        # 获取一个样本
        inputs, targets, _ = dataset[0]
        inputs = inputs.unsqueeze(0).to(device)
        
        # 预测
        outputs = model(inputs)
        
        # 重塑为2D
        input_2d = inputs.cpu().view(20, 20)
        target_2d = targets.view(200, 200)
        output_2d = outputs.cpu().view(200, 200)
        
        # 可视化
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(input_2d, cmap='coolwarm')
        axes[0].set_title('输入 (20x20)')
        
        axes[1].imshow(target_2d, cmap='coolwarm')
        axes[1].set_title('真实输出 (200x200)')
        
        axes[2].imshow(output_2d, cmap='coolwarm')
        axes[2].set_title('预测输出 (200x200)')
        
        plt.tight_layout()
        plt.savefig('training_progress.png', dpi=300)
        plt.show()
```

---

## 📞 支持和反馈

如果您在使用过程中遇到问题或有改进建议，请：

1. 检查本文档的常见问题部分
2. 查看示例代码
3. 检查配置文件设置
4. 确保数据格式正确

**祝您使用愉快！** 🎉