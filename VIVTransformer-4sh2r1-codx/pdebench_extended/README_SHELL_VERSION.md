# VIVTransformer Shell版本运行指南

本文档提供了在Linux/macOS环境下运行VIVTransformer项目的完整指南。

## 📁 项目结构

```
pdebench_extended/
├── run_training.sh              # 主训练脚本（Shell版本）
├── quick_start.sh               # 快速启动脚本
├── train_pressure_field.py     # Python训练脚本
├── configs/
│   └── pressure_field_training.yaml  # 训练配置文件
├── data/                        # 数据目录
└── ...
```

## 🚀 快速开始

### 1. 环境准备

确保系统已安装：
- Python 3.7+
- Git
- NVIDIA驱动（如果使用GPU）

### 2. 一键启动

```bash
# 进入项目目录
cd x:/2025/Graduation_project/report/VIVTransformer-4sh2r1-codx/pdebench_extended

# 运行快速启动脚本（首次运行）
bash quick_start.sh

# 运行训练脚本
bash run_training.sh
```

## 📋 详细使用说明

### 快速启动脚本 (quick_start.sh)

首次运行时使用，用于：
- 环境检查和依赖安装
- 数据下载和准备
- 目录结构创建

**数据下载选项：**
1. **从GitHub克隆PDEBench仓库**（推荐）
   - 自动克隆完整仓库
   - 包含所有数据集
   
2. **下载预处理数据集**
   - 生成适合训练的数据格式
   - 包含20x20→200x200压力场数据
   
3. **生成示例数据**
   - 用于快速测试
   - 小规模数据集

### 主训练脚本 (run_training.sh)

用于实际训练，提供以下功能：

#### 环境检查
- ✅ Python版本检测
- ✅ PyTorch安装验证
- ✅ GPU状态监控
- ✅ 依赖包检查

#### 数据管理
- 🔍 自动搜索数据文件（.pt, .hdf5, .h5）
- 📁 支持多种数据路径
- 🎯 智能数据文件定位

#### 运行模式
1. **开始训练** - 完整训练流程
2. **启动TensorBoard** - 实时监控训练进度
3. **测试模式** - 快速验证（1个epoch）
4. **查看配置** - 显示训练配置信息

## 🔧 配置说明

### 数据路径配置

配置文件已更新为正确的数据路径：
```yaml
data:
  path: "x:/2025/Graduation_project/report/VIVTransformer-4sh2r1-codx/pdebench_extended/data/PDEBench/pdebench/data_download/data/2D/DarcyFlow/2D_DarcyFlow_beta0.1_Train.hdf5"
```

### GPU配置

脚本会自动：
- 检测可用GPU
- 显示GPU状态（内存使用、温度等）
- 选择最优GPU进行训练

## 📊 训练监控

### TensorBoard

```bash
# 启动TensorBoard（在训练过程中或之后）
tensorboard --logdir=logs --port=6006

# 访问监控界面
# http://localhost:6006
```

### 日志文件

- **训练日志**: `logs/` 目录
- **模型检查点**: `checkpoints/` 目录
- **输出结果**: `outputs/` 目录

## 🛠️ 故障排除

### 常见问题

1. **权限问题**
   ```bash
   chmod +x run_training.sh
   chmod +x quick_start.sh
   ```

2. **Python路径问题**
   ```bash
   # 确保使用正确的Python版本
   which python3
   python3 --version
   ```

3. **数据文件未找到**
   ```bash
   # 重新运行快速启动脚本
   bash quick_start.sh
   ```

4. **GPU不可用**
   ```bash
   # 检查NVIDIA驱动
   nvidia-smi
   
   # 检查CUDA
   python3 -c "import torch; print(torch.cuda.is_available())"
   ```

### 依赖安装

```bash
# 安装基础依赖
pip install torch torchvision torchaudio
pip install numpy scipy matplotlib pyyaml h5py tqdm tensorboard

# 或使用requirements.txt
pip install -r requirements.txt
```

## 🎯 使用示例

### 完整训练流程

```bash
# 1. 首次设置
bash quick_start.sh
# 选择选项1：从GitHub克隆PDEBench仓库

# 2. 开始训练
bash run_training.sh
# 选择选项1：开始训练

# 3. 监控训练（新终端）
tensorboard --logdir=logs
```

### 快速测试

```bash
# 生成示例数据并快速测试
bash quick_start.sh  # 选择选项3：生成示例数据
bash run_training.sh # 选择选项3：测试模式
```

## 📈 性能优化

### GPU优化
- 脚本自动选择最空闲的GPU
- 支持多GPU环境
- 智能内存管理

### 训练优化
- 混合精度训练
- 梯度累积
- 学习率调度
- 早停机制

## 🔗 相关文件

- `run_training_pycharm.bat` - PyCharm版本（Windows）
- `train_pressure_field.py` - Python训练脚本
- `configs/pressure_field_training.yaml` - 训练配置
- `TRAINING_GUIDE.md` - 详细训练指南

## 📞 技术支持

如遇问题，请检查：
1. 系统环境是否满足要求
2. 数据文件是否正确下载
3. 配置文件路径是否正确
4. 日志文件中的错误信息

---

**注意**: 本Shell版本与Windows批处理版本功能相同，但针对Linux/macOS环境进行了优化。