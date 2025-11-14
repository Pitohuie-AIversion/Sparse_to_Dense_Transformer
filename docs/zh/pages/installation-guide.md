---
layout: default
title: 安装指南
nav_order: 3
description: "VIVTransformer 环境设置和依赖项安装"
permalink: /zh/pages/installation-guide/
lang: zh
ref: installation-guide
---

# 安装指南

本指南将帮助您设置 VIVTransformer 的运行环境。

## 📋 系统要求

### 硬件要求

- **CPU**: 多核处理器（推荐Intel i7或AMD Ryzen 7以上）
- **内存**: 最少8GB，推荐16GB或更多
- **存储**: 至少10GB可用空间
- **GPU**: NVIDIA GPU（推荐RTX 3060或更高）

### 软件要求

- **操作系统**: Linux (Ubuntu 20.04+), macOS (10.15+), Windows 10/11
- **Python**: 3.8, 3.9, 3.10, 或 3.11
- **CUDA**: 11.8 或更高版本（用于GPU支持）
- **Git**: 用于代码版本控制

## 🚀 快速安装

### 1. 克隆仓库

```bash
git clone https://github.com/Pitohuie-AIversion/Sparse_to_Dense_Transformer.git
cd Sparse_to_Dense_Transformer
```

### 2. 创建虚拟环境

#### 使用 conda（推荐）

```bash
conda create -n vivtransformer python=3.10
conda activate vivtransformer
```

#### 使用 venv

```bash
python -m venv vivtransformer-env
source vivtransformer-env/bin/activate  # Linux/Mac
# 或者
vivtransformer-env\Scripts\activate  # Windows
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

### 4. 验证安装

```bash
python -c "import vivtransformer; print('VIVTransformer installed successfully!')"
```

## 📦 详细安装步骤

### Python 环境设置

#### 检查 Python 版本

```bash
python --version
# 应该显示 Python 3.8+
```

#### 安装 pip（如果未安装）

```bash
python -m ensurepip --upgrade
```

### CUDA 安装（可选但推荐）

#### 检查 NVIDIA GPU

```bash
nvidia-smi
```

#### 安装 CUDA Toolkit

1. 访问 [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)
2. 选择您的操作系统和版本
3. 按照官方指南安装

#### 验证 CUDA 安装

```bash
nvcc --version
```

### PyTorch 安装

#### 有 CUDA 支持

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### CPU 版本

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### 依赖项说明

#### 核心依赖

- **torch**: PyTorch 深度学习框架
- **numpy**: 数值计算
- **pandas**: 数据处理
- **matplotlib**: 可视化
- **scikit-learn**: 机器学习工具
- **tqdm**: 进度条
- **tensorboard**: 训练监控
- **wandb**: 实验跟踪（可选）

#### 开发依赖

- **pytest**: 测试框架
- **black**: 代码格式化
- **flake8**: 代码检查
- **mypy**: 类型检查

## 🔧 高级配置

### 环境变量

```bash
# 设置 CUDA 设备
export CUDA_VISIBLE_DEVICES=0

# 设置随机种子
export PYTHONHASHSEED=42

# 设置 PyTorch 后端
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
```

### 配置文件

创建 `config.yaml` 文件：

```yaml
# 数据配置
data:
  batch_size: 32
  num_workers: 4
  pin_memory: true

# 模型配置
model:
  d_model: 128
  n_heads: 8
  n_layers: 6
  dropout: 0.1

# 训练配置
training:
  learning_rate: 1e-4
  n_epochs: 100
  early_stopping_patience: 10
```

## 🧪 测试安装

### 运行示例

```bash
python examples/basic_usage.py
```

### 运行测试

```bash
pytest tests/ -v
```

## 🐛 常见问题

### CUDA 相关问题

**问题**: `RuntimeError: CUDA out of memory`

**解决方案**: 
- 减小批大小
- 清理 GPU 内存：`torch.cuda.empty_cache()`
- 检查其他进程是否占用 GPU

**问题**: `CUDA capability sm_86 is not supported`

**解决方案**: 升级 PyTorch 到支持您的 GPU 架构的版本

### 依赖冲突

**问题**: 包版本冲突

**解决方案**: 
- 使用干净的虚拟环境
- 按照 requirements.txt 精确安装版本

### 权限问题

**问题**: 安装时权限被拒绝

**解决方案**: 
- 使用 `--user` 标志：`pip install --user package_name`
- 或者使用虚拟环境

## 📞 获取帮助

如果安装过程中遇到问题：

1. 查看 [故障排除指南]({{ site.baseurl }}/zh/pages/troubleshooting/)
2. 在 [GitHub Issues](https://github.com/Pitohuie-AIversion/Sparse_to_Dense_Transformer/issues) 报告问题
3. 加入我们的社区讨论

---

<div style="text-align: center; margin-top: 2rem;">
  <p><strong>下一步：</strong> <a href="{{ site.baseurl }}/zh/pages/quick-start-tutorial/">快速开始教程</a></p>
</div>