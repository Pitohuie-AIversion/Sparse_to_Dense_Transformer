# PDEBench数据集格式详细分析

## 概述

本文档详细阐述PDEBench中2D Darcy Flow和Compressible Navier-Stokes方程数据集的数据格式、存储结构和字段定义。

## 1. 2D Darcy Flow数据集

### 1.1 基本信息
- **方程类型**: 2D Darcy Flow（达西流方程）
- **文件格式**: HDF5 (`.hdf5`)
- **典型文件名**: `2D_DarcyFlow_beta{参数值}_Train.hdf5`
- **示例**: `2D_DarcyFlow_beta0.01_Train.hdf5`, `2D_DarcyFlow_beta1.0_Train.hdf5`

### 1.2 HDF5文件结构

#### 主要数据字段
```
2D_DarcyFlow_beta{X}_Train.hdf5
├── tensor          # 主要解场数据 (batch, t, x, y, channel)
├── nu              # 扩散系数场 (batch, t, x, y, channel)
├── x-coordinate    # x坐标网格
├── y-coordinate    # y坐标网格
└── attributes:
    └── beta        # PDE参数β值
```

#### 数据维度说明
- **tensor**: 形状为 `(batch, t, x, y, channel)`
  - `batch`: 批次大小（样本数量）
  - `t`: 时间步数（对于稳态问题通常为1）
  - `x, y`: 空间网格分辨率
  - `channel`: 通道数（通常为1，表示压力场或流场）

- **nu**: 扩散系数场，形状与tensor相同
  - 表示材料的渗透性分布
  - 是Darcy方程中的关键物理参数

#### 典型配置
- **空间分辨率**: 常见为32×32, 64×64, 128×128
- **时间序列长度**: 1（稳态问题）
- **数据类型**: `float32`
- **序列长度**: `sequence_length = 1`

### 1.3 物理意义
- **tensor字段**: 表示流体压力场或速度势
- **nu字段**: 表示多孔介质的渗透性分布
- **beta参数**: 控制边界条件或源项的强度

## 2. Compressible Navier-Stokes数据集

### 2.1 基本信息
- **方程类型**: Compressible Navier-Stokes（可压缩纳维-斯托克斯方程）
- **文件格式**: HDF5 (`.h5`)
- **典型文件名**: `{维度}D_CFD_{类型}_M{马赫数}_Eta{粘性系数}_Zeta{体粘性}__{边界条件}_{分辨率}_Train.hdf5`
- **示例**: 
  - `2D_CFD_Rand_M0.1_Eta1e-8_Zeta1e-8_periodic_512_Train.hdf5`
  - `3D_CFD_Rand_M1.0_Eta1e-8_Zeta1e-8_periodic_Train.hdf5`

### 2.2 HDF5文件结构

#### 主要数据字段
```
{X}D_CFD_*.hdf5
├── density         # 密度场 (batch, t, x, y[, z])
├── pressure        # 压力场 (batch, t, x, y[, z])
├── Vx              # x方向速度分量 (batch, t, x, y[, z])
├── Vy              # y方向速度分量 (batch, t, x, y[, z]) [2D/3D]
├── Vz              # z方向速度分量 (batch, t, x, y, z) [仅3D]
├── x-coordinate    # x坐标网格
├── y-coordinate    # y坐标网格 [2D/3D]
├── z-coordinate    # z坐标网格 [仅3D]
├── t-coordinate    # 时间坐标
└── attributes:
    ├── eta         # 动力粘性系数
    ├── zeta        # 体粘性系数
    └── M           # 马赫数 [2D/3D]
```

#### 数据维度说明
- **所有物理场**: 形状为 `(batch, t, x, y[, z])`
  - `batch`: 批次大小
  - `t`: 时间步数（时间演化问题，通常为49或更多）
  - `x, y, z`: 空间网格分辨率

#### 典型配置
- **1D CFD**: 
  - 分辨率: 1024点
  - 时间步: 通常49步
  - 字段: density, pressure, Vx
  
- **2D CFD**: 
  - 分辨率: 64×64, 128×128, 512×512
  - 时间步: 通常49步
  - 字段: density, pressure, Vx, Vy
  
- **3D CFD**: 
  - 分辨率: 64×64×64
  - 时间步: 通常49步
  - 字段: density, pressure, Vx, Vy, Vz

### 2.3 物理意义
- **density**: 流体密度分布
- **pressure**: 压力场分布
- **Vx, Vy, Vz**: 速度场的各个分量
- **eta**: 动力粘性系数（影响粘性耗散）
- **zeta**: 体粘性系数（影响压缩性效应）
- **M**: 马赫数（流速与声速的比值）

## 3. 数据加载和处理

### 3.1 数据读取示例

#### Darcy Flow数据读取
```python
import h5py
import numpy as np

# 读取Darcy Flow数据
with h5py.File('2D_DarcyFlow_beta1.0_Train.hdf5', 'r') as f:
    # 主要解场
    data = np.array(f['tensor'], dtype=np.float32)  # (batch, t, x, y, channel)
    # 扩散系数
    nu = np.array(f['nu'], dtype=np.float32)        # (batch, t, x, y, channel)
    # 坐标
    x_coord = np.array(f['x-coordinate'])
    y_coord = np.array(f['y-coordinate'])
    # 参数
    beta = f.attrs['beta']
    
print(f"Data shape: {data.shape}")
print(f"Nu shape: {nu.shape}")
print(f"Beta parameter: {beta}")
```

#### Compressible NS数据读取
```python
# 读取Compressible NS数据
with h5py.File('2D_CFD_Rand_M0.1_Eta1e-8_Zeta1e-8_periodic_512_Train.hdf5', 'r') as f:
    # 物理场
    density = np.array(f['density'], dtype=np.float32)   # (batch, t, x, y)
    pressure = np.array(f['pressure'], dtype=np.float32) # (batch, t, x, y)
    vx = np.array(f['Vx'], dtype=np.float32)            # (batch, t, x, y)
    vy = np.array(f['Vy'], dtype=np.float32)            # (batch, t, x, y)
    
    # 坐标
    x_coord = np.array(f['x-coordinate'])
    y_coord = np.array(f['y-coordinate'])
    t_coord = np.array(f['t-coordinate'])
    
    # 参数
    eta = f.attrs['eta']
    zeta = f.attrs['zeta']
    M = f.attrs['M']
    
print(f"Density shape: {density.shape}")
print(f"Pressure shape: {pressure.shape}")
print(f"Velocity shapes: Vx={vx.shape}, Vy={vy.shape}")
print(f"Parameters: eta={eta}, zeta={zeta}, M={M}")
```

### 3.2 数据预处理

#### 维度变换
- **Darcy Flow**: 通常需要将 `(batch, t, x, y, channel)` 展平为 `(batch, x*y*channel)`
- **Compressible NS**: 可能需要将多个物理场合并或分别处理

#### 归一化
- 每个物理场通常需要独立归一化
- 可以使用最大-最小归一化或标准化

## 4. 数据集特点对比

| 特征 | 2D Darcy Flow | Compressible Navier-Stokes |
|------|---------------|-----------------------------|
| **问题类型** | 稳态椭圆型PDE | 时间演化双曲-抛物型PDE |
| **时间依赖性** | 无（稳态） | 有（时间演化） |
| **主要物理量** | 压力场、渗透性 | 密度、压力、速度 |
| **典型分辨率** | 32×32, 128×128 | 64×64, 512×512 |
| **时间步数** | 1 | 49+ |
| **数据复杂度** | 相对简单 | 复杂（多物理场耦合） |
| **计算成本** | 低 | 高 |
| **应用领域** | 多孔介质流动 | 可压缩流体力学 |

## 5. 使用建议

### 5.1 模型输入输出设计
- **Darcy Flow**: 输入渗透性场，输出压力场
- **Compressible NS**: 输入初始状态，输出时间序列或最终状态

### 5.2 数据增强策略
- 空间旋转和翻转
- 参数插值生成新样本
- 多尺度训练

### 5.3 评估指标
- L2相对误差
- 物理守恒量检查
- 长时间稳定性（对于NS方程）

## 6. 注意事项

1. **内存管理**: CFD数据集较大，需要注意内存使用
2. **数据质量**: 检查NaN值和异常值
3. **物理一致性**: 确保预处理不破坏物理约束
4. **边界条件**: 注意不同数据集的边界条件设置
5. **单位系统**: 确保物理量的单位一致性

## 7. 扩展信息

### 7.1 相关文件
- 配置文件: `config_Darcy.yaml`, `config_2DCFD.yaml`
- 可视化脚本: `visualize_pdes.py`
- 数据生成: `Data_Merge.py`

### 7.2 数据下载
```bash
# 下载Darcy Flow数据
python download_direct.py --root_folder ./data --pde_name darcy

# 下载2D CFD数据
python download_direct.py --root_folder ./data --pde_name 2d_cfd
```

---

*本文档基于PDEBench项目的源代码和数据结构分析生成，详细信息请参考项目官方文档。*