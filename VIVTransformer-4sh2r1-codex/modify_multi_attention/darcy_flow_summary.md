# Darcy Flow 数据加载和可视化总结

## 概述

我们成功加载并可视化了 Darcy Flow HDF5 数据集。以下是详细的分析和结果。

## 数据集信息

**文件路径**: `X:/2025/Graduation_project/report/VIVTransformer-4sh2r1-codex/modify_multi_attention/PDEBench/pdebench/data_download/data/2D/DarcyFlow/2D_DarcyFlow_beta0.1_Train.hdf5`

### HDF5文件结构

- **nu**: 形状 (10000, 128, 128), 数据类型 float32 - 扩散系数
- **tensor**: 形状 (10000, 1, 128, 128), 数据类型 float32 - 解场
- **x-coordinate**: 形状 (128,), 数据类型 float32 - X坐标
- **y-coordinate**: 形状 (128,), 数据类型 float32 - Y坐标

### 数据统计信息

#### Tensor数据 (解场)
- **形状**: (10000, 1, 128, 128)
- **数值范围**: [0.000018, 0.588051]
- **均值**: 0.024394
- **标准差**: 0.040709

#### Nu数据 (扩散系数)
- **形状**: (10000, 128, 128)
- **数值范围**: [0.100000, 1.000000]
- **均值**: 0.541578
- **标准差**: 0.449922

## 可视化结果

我们生成了以下可视化文件：

### 单样本可视化
1. **darcy_flow_sample_0.png** - 样本0的tensor和nu并排显示
2. **darcy_flow_tensor_sample_0.png** - 样本0的解场单独显示
3. **darcy_flow_nu_sample_0.png** - 样本0的扩散系数单独显示

4. **darcy_flow_sample_100.png** - 样本100的tensor和nu并排显示
5. **darcy_flow_tensor_sample_100.png** - 样本100的解场单独显示
6. **darcy_flow_nu_sample_100.png** - 样本100的扩散系数单独显示

### 多样本对比
7. **darcy_flow_multiple_samples.png** - 6个不同样本的对比可视化

### 其他可视化文件
8. **darcy_flow_visualization.png** - 使用测试脚本生成的可视化
9. **darcy_visualization_sample_0.png** - 交互式可视化脚本生成的结果

## 样本数据分析

### 样本0
- **Tensor范围**: [0.000197, 0.073899]
- **Nu范围**: [0.100000, 1.000000]

### 样本100
- **Tensor范围**: [0.000133, 0.023483]
- **Nu范围**: [0.100000, 1.000000]

## 使用的工具和脚本

### 1. 现有的PDEBench可视化脚本
- **文件**: `visualize_pdes.py`
- **功能**: 使用内置的`visualize_darcy`函数
- **命令**: `python visualize_pdes.py --data_path [path] --pde_name darcy --param 0.1`
- **输出**: `2D_DarcyFlow.pdf`

### 2. 交互式数据加载脚本
- **文件**: `load_and_visualize_darcy.py`
- **功能**: 详细的数据结构分析和可视化
- **特点**: 支持中文显示，提供详细的数据统计信息

### 3. 简化的批量可视化脚本
- **文件**: `simple_darcy_visualizer.py`
- **功能**: 批量生成高质量的可视化图片
- **特点**: 使用非交互式后端，适合批量处理

## 数据特征观察

1. **扩散系数 (Nu)**: 在0.1到1.0之间变化，呈现复杂的空间分布模式
2. **解场 (Tensor)**: 数值较小，主要集中在0到0.1之间，显示了流体在不同扩散系数下的分布
3. **空间分辨率**: 128x128网格，提供了足够的空间细节
4. **样本数量**: 10000个样本，为机器学习提供了充足的训练数据

## 技术要点

1. **HDF5文件格式**: 高效的科学数据存储格式
2. **数据访问**: 使用h5py库进行数据读取
3. **可视化**: 使用matplotlib进行2D热图可视化
4. **颜色映射**: 
   - Tensor数据使用'viridis'色彩映射
   - Nu数据使用'plasma'色彩映射

## 结论

Darcy Flow数据集已成功加载和可视化。数据结构清晰，包含了完整的输入（扩散系数nu）和输出（解场tensor）信息，适合用于偏微分方程求解的机器学习研究。生成的可视化文件清楚地展示了不同样本的数据特征和空间分布模式。