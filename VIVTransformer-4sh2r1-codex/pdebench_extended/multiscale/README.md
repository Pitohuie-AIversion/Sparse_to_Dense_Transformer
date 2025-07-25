# 多尺度超分辨率重构模块

这个模块提供了完整的多尺度超分辨率重构功能，支持从低分辨率输入重构到高分辨率输出。

## 📁 项目结构

```
multiscale/
├── __init__.py                    # 模块初始化文件
├── README.md                      # 本文档
├── data/                          # 数据处理模块
│   ├── __init__.py
│   └── multiscale_adapter.py      # 多尺度数据适配器
├── configs/                       # 配置文件
│   └── multiscale_config.yaml     # 多尺度配置文件
├── training/                      # 训练模块
│   ├── __init__.py
│   └── train_multiscale.py        # 多尺度训练脚本
├── testing/                       # 测试模块
│   ├── __init__.py
│   └── test_multiscale.py         # 多尺度测试脚本
├── examples/                      # 示例和运行脚本
│   ├── __init__.py
│   └── run_multiscale.py          # 多尺度运行脚本
└── docs/                          # 文档
    └── README_MultiScale.md       # 详细文档
```

## 🚀 快速开始

### 1. 基本使用

```bash
# 运行2倍超分辨率实验
python multiscale/examples/run_multiscale.py --scale_factor 2 --data_path data/pdebench/darcy_flow_beta_0.01.h5

# 运行4倍超分辨率实验
python multiscale/examples/run_multiscale.py --scale_factor 4 --data_path data/pdebench/darcy_flow_beta_0.01.h5

# 使用配置文件运行
python multiscale/examples/run_multiscale.py --config multiscale/configs/multiscale_config.yaml
```

### 2. 仅运行测试

```bash
# 测试多尺度功能
python multiscale/examples/run_multiscale.py --test_only

# 直接运行测试脚本
python multiscale/testing/test_multiscale.py
```

### 3. 直接训练

```bash
# 直接运行训练脚本
python multiscale/training/train_multiscale.py --config multiscale/configs/multiscale_config.yaml
```

## 🔧 核心功能

### 数据处理 (`data/`)
- **MultiScaleDataset**: 多尺度数据集类
- **MultiScaleDataLoader**: 多尺度数据加载器
- **智能下采样**: 支持average、bilinear、nearest方法
- **自动维度计算**: 根据缩放因子自动计算输入输出维度

### 训练系统 (`training/`)
- **MultiScaleTrainer**: 完整的训练流程
- **多GPU支持**: 自动检测和使用多GPU
- **智能检查点**: 自动保存最佳模型
- **可视化**: 训练过程可视化和预测结果可视化

### 测试系统 (`testing/`)
- **功能测试**: 数据集和数据加载器测试
- **多尺度测试**: 不同缩放因子测试
- **可视化测试**: 数据可视化功能测试
- **性能分析**: 数据分析和性能评估

### 配置系统 (`configs/`)
- **灵活配置**: 支持多种缩放因子和参数
- **自适应参数**: 根据缩放因子自动调整批次大小和学习率
- **实验管理**: 支持多个实验配置

## 📊 支持的缩放因子

| 缩放因子 | 输入分辨率 | 输出分辨率 | 输入维度 | 输出维度 | 压缩比 |
|---------|-----------|-----------|----------|----------|--------|
| 2x      | 64×64     | 128×128   | 4,096    | 16,384   | 4.0    |
| 4x      | 32×32     | 128×128   | 1,024    | 16,384   | 16.0   |
| 8x      | 16×16     | 128×128   | 256      | 16,384   | 64.0   |
| 16x     | 8×8       | 128×128   | 64       | 16,384   | 256.0  |

## 🎯 主要特性

1. **模块化设计**: 清晰的目录结构，便于维护和扩展
2. **完整工作流**: 从数据处理到训练测试的完整流程
3. **灵活配置**: 支持多种实验配置和参数调整
4. **可视化支持**: 丰富的可视化功能
5. **性能优化**: 内存效率和计算优化
6. **错误处理**: 完善的错误处理和日志记录

## 📖 详细文档

更多详细信息请参考：
- [详细使用指南](docs/README_MultiScale.md)
- [API文档](data/multiscale_adapter.py)
- [配置说明](configs/multiscale_config.yaml)

## 🔗 集成说明

这个模块已经集成到主项目的数据加载系统中：
- `data/dataloader.py` 中的 `get_adaptive_loaders` 函数会自动检测多尺度配置
- 当配置中 `use_multiscale: true` 时，会自动使用多尺度数据加载器

## 🤝 贡献

欢迎贡献代码和建议！请确保：
1. 遵循现有的代码风格
2. 添加适当的测试
3. 更新相关文档

## 📄 许可证

本项目采用与主项目相同的许可证。