# VIVTransformer 多尺度超分辨率重构

## 概述

本项目实现了基于VIVTransformer的多尺度超分辨率重构功能，能够从低分辨率输入重构出高分辨率输出，实现"局部到整体"的升阶预测。

### 核心功能

- **多尺度数据处理**: 支持将高分辨率数据（如128×128）按整数倍缩小作为输入
- **超分辨率重构**: 从低分辨率输入预测回原始高分辨率
- **多种缩放因子**: 支持2×、4×、8×等不同缩放比例
- **灵活的下采样方法**: 平均池化、双线性插值、最近邻插值
- **完整的训练流程**: 包含训练、验证、测试和可视化

### 技术特点

- 基于Transformer架构的超分辨率重构
- 支持PDEBench数据集
- 自动维度计算和配置
- 多GPU训练支持
- 丰富的可视化功能

## 项目结构

```
pdebench_extended/
├── data/
│   ├── multiscale_adapter.py      # 多尺度数据适配器
│   └── dataloader.py              # 数据加载器（已更新）
├── configs/
│   └── multiscale_config.yaml     # 多尺度配置文件
├── train_multiscale.py            # 多尺度训练脚本
├── test_multiscale.py             # 多尺度测试脚本
├── run_multiscale.py              # 运行脚本
└── README_MultiScale.md           # 本文档
```

## 快速开始

### 1. 环境准备

确保已安装以下依赖：

```bash
pip install torch torchvision
pip install numpy matplotlib
pip install pyyaml tqdm
pip install h5py scikit-learn
```

### 2. 数据准备

将PDEBench数据文件放置在适当位置，例如：
```
data/pdebench/darcy_flow_beta_0.01.h5
```

### 3. 运行实验

#### 方法一：使用运行脚本（推荐）

```bash
# 运行2倍缩放实验（64×64 → 128×128）
python run_multiscale.py --scale_factor 2 --data_path data/pdebench/darcy_flow_beta_0.01.h5

# 运行4倍缩放实验（32×32 → 128×128）
python run_multiscale.py --scale_factor 4 --data_path data/pdebench/darcy_flow_beta_0.01.h5

# 运行8倍缩放实验（16×16 → 128×128）
python run_multiscale.py --scale_factor 8 --data_path data/pdebench/darcy_flow_beta_0.01.h5
```

#### 方法二：使用配置文件

```bash
# 生成配置文件
python run_multiscale.py --scale_factor 2 --generate_config_only

# 使用配置文件运行
python run_multiscale.py --config configs/multiscale_config_2x.yaml
```

#### 方法三：直接运行训练脚本

```bash
python train_multiscale.py
```

### 4. 测试功能

```bash
# 运行功能测试
python run_multiscale.py --test_only --config configs/multiscale_config.yaml

# 或直接运行测试脚本
python test_multiscale.py
```

## 配置说明

### 主要配置参数

```yaml
data:
  use_multiscale: true              # 启用多尺度模式
  scale_factor: 2                   # 缩放因子
  original_resolution: [128, 128]   # 原始分辨率
  downsampling_method: "average"    # 下采样方法
  data_path: "data/pdebench/darcy_flow_beta_0.01.h5"
  pde_type: "darcy_flow"
  sequence_length: 1
  normalize: true
  batch_size: 16

model:
  input_dim: 4096    # 自动计算：64×64×1
  output_dim: 16384  # 自动计算：128×128×1
  d_model: 512
  nhead: 8
  num_encoder_layers: 6
  num_decoder_layers: 6
```

### 缩放因子与维度对应关系

| 缩放因子 | 输入分辨率 | 输出分辨率 | 输入维度 | 输出维度 | 压缩比 |
|---------|-----------|-----------|---------|---------|-------|
| 2×      | 64×64     | 128×128   | 4,096   | 16,384  | 4.0   |
| 4×      | 32×32     | 128×128   | 1,024   | 16,384  | 16.0  |
| 8×      | 16×16     | 128×128   | 256     | 16,384  | 64.0  |

## 使用示例

### 示例1：基础2倍缩放实验

```bash
# 生成配置并运行
python run_multiscale.py --scale_factor 2 --data_path data/pdebench/darcy_flow_beta_0.01.h5
```

这将：
1. 创建配置文件 `configs/multiscale_config_2x.yaml`
2. 加载128×128的原始数据
3. 将数据下采样到64×64作为输入
4. 训练模型从64×64重构到128×128
5. 保存训练结果和可视化

### 示例2：高压缩比实验

```bash
# 8倍缩放实验（高压缩比）
python run_multiscale.py --scale_factor 8 --data_path data/pdebench/darcy_flow_beta_0.01.h5
```

这将进行16×16到128×128的超分辨率重构，压缩比达到64倍。

### 示例3：自定义配置

```yaml
# 创建自定义配置文件 my_config.yaml
data:
  use_multiscale: true
  scale_factor: 4
  original_resolution: [128, 128]
  downsampling_method: "bilinear"  # 使用双线性插值
  batch_size: 32
  
model:
  d_model: 1024  # 增大模型容量
  num_encoder_layers: 8
  
training:
  epochs: 200
  learning_rate: 0.0005
```

```bash
python run_multiscale.py --config my_config.yaml
```

## 输出结果

### 目录结构

```
results/multiscale_2x/
├── checkpoints/
│   ├── best_model.pth
│   ├── latest_checkpoint.pth
│   └── checkpoint_epoch_*.pth
├── visualizations/
│   ├── training_curves.png
│   ├── prediction_sample_0.png
│   └── ...
├── test_results.json
└── ...

logs/multiscale_2x/
└── training_*.log
```

### 可视化结果

1. **训练曲线**: 显示训练和验证损失变化
2. **预测结果**: 对比预测输出与真实目标
3. **误差分析**: 显示预测误差分布

### 性能指标

- **MSE**: 均方误差
- **MAE**: 平均绝对误差
- **PSNR**: 峰值信噪比
- **SSIM**: 结构相似性指数

## 高级功能

### 1. 多种下采样方法

```yaml
data:
  downsampling_method: "center_crop"  # 中心截取（推荐）
  # downsampling_method: "average"    # 平均池化
  # downsampling_method: "bilinear"   # 双线性插值
  # downsampling_method: "nearest"    # 最近邻插值
```

**下采样方法说明：**
- **center_crop**: 从原数据集中心截取低分辨率区域，保持原始数据的局部细节
- **average**: 平均池化下采样，对局部区域取平均值
- **bilinear**: 双线性插值下采样，平滑的尺寸缩放
- **nearest**: 最近邻插值下采样，保持原始像素值

### 2. 多GPU训练

```yaml
model:
  use_data_parallel: true  # 启用多GPU训练
```

### 3. 学习率调度

```yaml
training:
  scheduler:
    type: "StepLR"        # 阶梯式衰减
    step_size: 30
    gamma: 0.5
    # type: "CosineAnnealingLR"  # 余弦退火
    # type: "ReduceLROnPlateau"  # 自适应衰减
```

### 4. 早停机制

```yaml
training:
  early_stopping:
    patience: 15      # 容忍轮数
    min_delta: 0.0001 # 最小改善
```

## 故障排除

### 常见问题

1. **内存不足**
   - 减小批次大小：`batch_size: 8`
   - 减小模型大小：`d_model: 256`
   - 启用梯度检查点：`gradient_checkpointing: true`

2. **数据文件不存在**
   - 检查数据路径是否正确
   - 确保数据文件格式为HDF5

3. **训练不收敛**
   - 调整学习率：`learning_rate: 0.0001`
   - 增加训练轮数：`epochs: 200`
   - 检查数据归一化：`normalize: true`

4. **维度不匹配**
   - 确保原始分辨率能被缩放因子整除
   - 检查配置文件中的维度设置

### 调试模式

```yaml
debug:
  enabled: true
  options:
    profile_memory: true
    check_gradients: true
```

## 扩展功能

### 1. 添加新的PDE类型

在 `multiscale_adapter.py` 中添加对新PDE类型的支持：

```python
# 支持新的PDE类型
pde_type = "new_pde_type"
```

### 2. 自定义损失函数

```python
class CustomLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()
    
    def forward(self, pred, target):
        return self.mse(pred, target) + 0.1 * self.l1(pred, target)
```

### 3. 添加新的评估指标

```python
def calculate_psnr(pred, target):
    mse = torch.mean((pred - target) ** 2)
    psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
    return psnr
```

## 性能优化

### 1. 数据加载优化

```yaml
data:
  num_workers: 8      # 增加工作进程
  pin_memory: true    # 固定内存
  prefetch_factor: 2  # 预取因子
```

### 2. 模型优化

```yaml
compute:
  mixed_precision: true      # 混合精度训练
  gradient_checkpointing: true  # 梯度检查点
```

### 3. 内存管理

```yaml
compute:
  max_memory_usage: "80%"
  clear_cache_every: 10
```

## 贡献指南

欢迎贡献代码和改进建议！请遵循以下步骤：

1. Fork 项目
2. 创建功能分支
3. 提交更改
4. 创建 Pull Request

## 许可证

本项目采用 MIT 许可证。

## 联系方式

如有问题或建议，请通过以下方式联系：

- 项目Issues
- 邮件联系

---

**注意**: 本文档持续更新中，请关注最新版本。