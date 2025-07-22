# PDEBench数据集集成指南

本指南介绍如何在多注意力机制项目中使用PDEBench数据集进行偏微分方程求解实验。

## 概述

PDEBench是一个综合性的偏微分方程基准数据集，包含多种类型的PDE问题：
- Navier-Stokes方程（不可压缩流体）
- Darcy流方程
- 浅水方程
- 对流方程
- Burgers方程
- 反应扩散方程

## 安装依赖

首先安装必要的依赖包：

```bash
pip install -r requirements.txt
```

新增的依赖包包括：
- `h5py`: 用于读取HDF5格式的数据文件
- `scipy`: 科学计算库
- `sklearn`: 机器学习工具
- `seaborn`: 数据可视化
- `tqdm`: 进度条显示

## 数据准备

### 1. 下载PDEBench数据集

访问 [PDEBench GitHub仓库](https://github.com/pdebench/PDEBench) 下载所需的数据文件。

### 2. 数据目录结构

将下载的数据文件放置在以下目录结构中：

```
data/
└── pdebench/
    ├── ns_incom_inhom_2d.h5          # Navier-Stokes不可压缩流
    ├── darcy_flow_2d.h5              # Darcy流方程
    ├── shallow_water_2d.h5           # 浅水方程
    ├── advection_2d.h5               # 对流方程
    ├── burgers_2d.h5                 # Burgers方程
    └── reaction_diffusion_2d.h5      # 反应扩散方程
```

## 配置文件

### 1. 基本配置

在配置文件中启用PDEBench数据集：

```yaml
data:
  use_pdebench: true
  batch_size: 16
  num_workers: 4
  normalize: true
  use_augmentation: false

current_pde: "ns_incom"  # 当前使用的PDE类型
```

### 2. PDEBench特定配置

```yaml
pdebench:
  data_root: "./data/pdebench"
  pde_configs:
    ns_incom:
      data_file: "ns_incom_inhom_2d.h5"
      spatial_resolution: [64, 64]
      sequence_length: 49
      input_dim: 4096
      output_dim: 4096
      description: "Navier-Stokes incompressible flow"
    
    darcy_flow:
      data_file: "darcy_flow_2d.h5"
      spatial_resolution: [32, 32]
      sequence_length: 1
      input_dim: 1024
      output_dim: 1024
      description: "Darcy flow equation"
    
    shallow_water:
      data_file: "shallow_water_2d.h5"
      spatial_resolution: [128, 128]
      sequence_length: 40
      input_dim: 16384
      output_dim: 16384
      description: "Shallow water equations"
```

## 使用方法

### 1. 快速开始

运行示例脚本：

```bash
python examples/pdebench_example.py
```

### 2. 完整训练

使用PDEBench配置文件进行训练：

```bash
python main.py -c configs/pdebench_config.yaml
```

### 3. 指定PDE类型

修改配置文件中的 `current_pde` 字段来切换不同的PDE类型：

```yaml
current_pde: "darcy_flow"  # 切换到Darcy流方程
```

## 代码结构

### 新增文件

1. **`data/pdebench_adapter.py`**: PDEBench数据集适配器
   - `PDEBenchDataset`: 数据集类
   - `PDEBenchDataLoader`: 数据加载器

2. **`configs/pdebench_config.yaml`**: PDEBench专用配置文件

3. **`examples/pdebench_example.py`**: 使用示例脚本

### 修改文件

1. **`data/dataloader.py`**: 添加PDEBench支持
   - `get_pdebench_loaders()`: PDEBench数据加载器
   - `get_adaptive_loaders()`: 自适应数据加载器

2. **`main.py`**: 更新为使用自适应数据加载器

3. **`requirements.txt`**: 添加新依赖

## 数据格式说明

### 输入数据格式

PDEBench数据集使用HDF5格式存储，包含以下字段：
- `data`: 主要数据数组，形状为 `[N, T, H, W, C]`
  - N: 样本数量
  - T: 时间步数
  - H, W: 空间维度
  - C: 通道数（如速度分量）

### 数据预处理

1. **空间重塑**: 将2D空间数据展平为1D向量
2. **序列准备**: 提取输入序列和目标序列
3. **归一化**: 可选的数据归一化
4. **数据分割**: 70%训练，15%验证，15%测试

## 评估指标

PDEBench支持以下评估指标：

- **相对L2误差**: `||pred - true||_2 / ||true||_2`
- **均方误差 (MSE)**: `mean((pred - true)^2)`
- **平均绝对误差 (MAE)**: `mean(|pred - true|)`
- **能量谱误差**: 频域分析误差
- **结构相似性 (SSIM)**: 图像质量评估

## 可视化

项目支持以下可视化功能：

1. **训练曲线**: 损失和指标随时间变化
2. **预测结果**: 真实值vs预测值对比
3. **误差分布**: 空间误差分布图
4. **注意力权重**: 注意力机制可视化

## 性能优化

### 内存优化

1. **批次大小**: 根据GPU内存调整批次大小
2. **数据加载**: 使用多进程加载数据
3. **梯度累积**: 对于大模型使用梯度累积

### 计算优化

1. **混合精度**: 使用FP16训练
2. **模型并行**: 多GPU训练支持
3. **检查点**: 定期保存模型状态

## 故障排除

### 常见问题

1. **数据文件未找到**
   ```
   FileNotFoundError: PDEBench数据文件不存在
   ```
   解决方案: 检查数据文件路径和文件名

2. **内存不足**
   ```
   CUDA out of memory
   ```
   解决方案: 减小批次大小或使用梯度累积

3. **维度不匹配**
   ```
   RuntimeError: size mismatch
   ```
   解决方案: 检查模型输入输出维度配置

### 调试技巧

1. **数据检查**: 使用示例脚本验证数据加载
2. **小规模测试**: 先在小数据集上测试
3. **日志分析**: 查看详细的训练日志

## 扩展功能

### 添加新的PDE类型

1. 在 `pdebench_config.yaml` 中添加新配置
2. 确保数据文件格式兼容
3. 调整模型输入输出维度

### 自定义数据增强

1. 在 `PDEBenchDataset` 中添加变换
2. 考虑PDE的物理约束
3. 验证增强后的数据质量

## 参考资源

- [PDEBench论文](https://arxiv.org/abs/2210.07182)
- [PDEBench GitHub](https://github.com/pdebench/PDEBench)
- [数据下载链接](https://darus.uni-stuttgart.de/dataset.xhtml?persistentId=doi:10.18419/darus-2986)
- [官方文档](https://pdebench.readthedocs.io/)

## 许可证

PDEBench数据集遵循其原始许可证。请在使用前查看相关许可条款。