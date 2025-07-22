# PDEBench 数据集构建与训练快速指南

## 概述

本指南将帮助您快速开始使用 PDEBench 数据集构建系统，为 Transformer 模型训练准备数据。系统支持多种 PDE 类型和注意力机制的组合实验。

## 🚀 快速开始

### 1. 环境准备

```bash
# 确保Python环境已安装
python --version

# 安装依赖包
pip install -r requirements.txt

# 验证关键包
python -c "import torch, numpy, h5py, yaml, matplotlib"
```

### 2. 数据文件准备

确保以下数据文件存在于正确路径：

```
data/pdebench/
├── 2D_DarcyFlow_beta1.0_Train.hdf5
├── 2D_DarcyFlow_beta1.0_Valid.hdf5
├── 2D_DarcyFlow_beta1.0_Test.hdf5
├── 2D_CFD_Rand_M1.0_Eta1e-08_Zeta1e-08_periodic_Train.hdf5
├── 2D_CFD_Rand_M1.0_Eta1e-08_Zeta1e-08_periodic_Valid.hdf5
├── 2D_CFD_Rand_M1.0_Eta1e-08_Zeta1e-08_periodic_Test.hdf5
└── ...
```

### 3. 使用批处理脚本（推荐）

**Windows 用户：**
```cmd
run_experiments.bat
```

**Linux/Mac 用户：**
```bash
python experiment_manager.py -c configs/multi_pde_training_config.yaml
```

### 4. 手动操作步骤

#### 步骤 1: 验证数据文件
```bash
python build_training_dataset.py --validate-only
```

#### 步骤 2: 构建数据集
```bash
# 构建单个数据集
python build_training_dataset.py -p darcy_flow

# 构建所有数据集
python build_training_dataset.py -p all
```

#### 步骤 3: 运行训练
```bash
# 单个实验
set CURRENT_PDE=darcy_flow
set ATTENTION_TYPE=MultiHeadAttention
python main.py -c configs/multi_pde_training_config.yaml

# 批量实验
python experiment_manager.py -c configs/multi_pde_training_config.yaml
```

## 📊 支持的配置

### PDE 类型
- **darcy_flow**: 2D Darcy Flow
- **ns_compressible**: 2D Compressible Navier-Stokes
- **shallow_water**: 2D Shallow Water
- **burgers_1d**: 1D Burgers
- **reaction_diffusion**: 2D Reaction-Diffusion

### 注意力机制
- **MultiHeadAttention**: 标准多头注意力
- **ECAAttention**: 高效通道注意力
- **SEAttention**: Squeeze-and-Excitation 注意力
- **CBAM**: 卷积块注意力模块
- **CoordinateAttention**: 坐标注意力

## 🔧 配置文件说明

### 主配置文件: `multi_pde_training_config.yaml`

```yaml
# 全局设置
global:
  device: "cuda"  # 或 "cpu"
  result_dir: "results"
  
# 数据参数
data:
  batch_size: 32
  num_workers: 4
  
# 模型架构
model:
  d_model: 256
  num_heads: 8
  num_layers: 6
  
# 训练参数
training:
  epochs: 100
  learning_rate: 0.001
```

### 数据集特定配置

每个 PDE 类型都有专门的配置部分，包括：
- 数据文件路径
- 输入/输出维度
- 序列长度
- 空间分辨率
- 物理参数

## 📁 输出结构

```
results/
├── experiment_YYYYMMDD_HHMMSS/
│   ├── experiment_report.md
│   ├── results_summary.json
│   ├── results_comparison.csv
│   ├── visualizations/
│   │   ├── validation_loss_heatmap.png
│   │   ├── training_time_comparison.png
│   │   └── loss_distribution_boxplot.png
│   └── individual_results/
│       ├── darcy_flow_MultiHeadAttention/
│       ├── ns_compressible_ECAAttention/
│       └── ...
└── logs/
    ├── build_dataset.log
    └── training.log
```

## 🎯 使用场景

### 场景 1: 快速验证单个模型
```bash
# 1. 验证数据
python build_training_dataset.py --validate-only

# 2. 构建 Darcy Flow 数据集
python build_training_dataset.py -p darcy_flow

# 3. 训练单个模型
set CURRENT_PDE=darcy_flow
set ATTENTION_TYPE=MultiHeadAttention
python main.py -c configs/multi_pde_training_config.yaml
```

### 场景 2: 比较不同注意力机制
```bash
# 运行所有注意力机制在 Darcy Flow 上的实验
python experiment_manager.py -c configs/multi_pde_training_config.yaml --pde-filter darcy_flow
```

### 场景 3: 完整的消融研究
```bash
# 运行所有 PDE 类型和注意力机制的组合
python experiment_manager.py -c configs/multi_pde_training_config.yaml
```

## 🔍 故障排除

### 常见问题

1. **数据文件未找到**
   ```
   错误: FileNotFoundError: 数据文件不存在
   解决: 检查 configs/multi_pde_training_config.yaml 中的文件路径
   ```

2. **内存不足**
   ```
   错误: CUDA out of memory
   解决: 减少 batch_size 或使用 CPU 训练
   ```

3. **依赖包缺失**
   ```
   错误: ModuleNotFoundError
   解决: pip install -r requirements.txt
   ```

### 调试模式

```bash
# 启用详细日志
export PYTHONPATH=.
export LOG_LEVEL=DEBUG
python build_training_dataset.py -p darcy_flow --verbose
```

### 性能优化

1. **数据加载优化**
   - 增加 `num_workers`
   - 使用 SSD 存储数据
   - 启用数据预加载

2. **训练优化**
   - 使用混合精度训练
   - 调整学习率调度器
   - 启用梯度累积

## 📈 结果分析

### 自动生成的报告

实验完成后，系统会自动生成：

1. **实验报告** (`experiment_report.md`)
   - 实验配置总结
   - 性能对比表格
   - 最佳模型推荐

2. **可视化图表**
   - 验证损失热力图
   - 训练时间对比
   - 损失分布箱线图

3. **数据文件**
   - JSON 格式的详细结果
   - CSV 格式的对比数据

### 手动分析

```bash
# 分析现有结果
python experiment_manager.py -c configs/multi_pde_training_config.yaml --analyze-only

# 生成自定义报告
python analysis/generate_report.py --result-dir results/experiment_YYYYMMDD_HHMMSS
```

## 🔄 扩展指南

### 添加新的 PDE 类型

1. 在 `multi_pde_training_config.yaml` 中添加配置
2. 更新 `build_training_dataset.py` 中的数据加载逻辑
3. 在 `experiment_manager.py` 中注册新的 PDE 类型

### 添加新的注意力机制

1. 在 `mymodels/components/attention_factory.py` 中注册
2. 更新配置文件中的 `attention_types` 列表
3. 测试兼容性

## 📞 支持

如果遇到问题：

1. 查看日志文件 (`logs/` 目录)
2. 检查配置文件格式
3. 验证数据文件完整性
4. 参考 `DATASET_FORMAT_ANALYSIS.md` 了解数据格式

---

**祝您训练顺利！** 🎉