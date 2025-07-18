---
layout: default
title: Installation Guide
parent: Getting Started
nav_order: 2
description: "详细的VIVTransformer安装指南"
permalink: /pages/installation-guide/
---

# 安装指南 {#安装指南}

本指南将详细介绍如何在不同环境中安装和配置VIVTransformer项目。

## 📋 目录 {#目录}

- [系统要求](#系统要求)
- [Python环境配置](#python环境配置)
- [项目安装](#项目安装)
- [GPU环境配置](#gpu环境配置)
- [验证安装](#验证安装)
- [常见问题](#常见问题)
- [卸载指南](#卸载指南)

## 系统要求 {#系统要求}

### 🖥️ 硬件要求 {#硬件要求}

| 组件 | 最低要求 | 推荐配置 | 说明 |
|------|----------|----------|------|
| **CPU** | Intel i5 / AMD Ryzen 5 | Intel i7 / AMD Ryzen 7 | 支持AVX指令集 |
| **内存** | 8GB RAM | 16GB+ RAM | 大型模型需要更多内存 |
| **存储** | 5GB 可用空间 | 20GB+ SSD | 包含数据集和模型存储 |
| **GPU** | 可选 | NVIDIA RTX 3060+ | 8GB+ VRAM推荐 |

### 🔧 软件要求 {#软件要求}

| 软件 | 版本要求 | 说明 |
|------|----------|------|
| **操作系统** | Windows 10+, Ubuntu 18.04+, macOS 10.15+ | 64位系统 |
| **Python** | 3.8 - 3.11 | 推荐使用3.9或3.10 |
| **Git** | 2.20+ | 用于克隆仓库 |
| **CUDA** | 11.8+ (可选) | GPU加速需要 |
| **cuDNN** | 8.6+ (可选) | 配合CUDA使用 |

## Python环境配置 {#python环境配置}

### 🐍 方法1：使用Conda (推荐) {#方法1-使用conda-推荐}

```bash
# 1. 安装Miniconda或Anaconda
# 下载地址：https://docs.conda.io/en/latest/miniconda.html

# 2. 创建虚拟环境
conda create -n vivtransformer python=3.9

# 3. 激活环境
conda activate vivtransformer

# 4. 验证Python版本
python --version
# 应该显示：Python 3.9.x
```

### 🔧 方法2：使用venv {#方法2-使用venv}

```bash
# 1. 创建虚拟环境
python -m venv vivtransformer-env

# 2. 激活环境
# Windows:
vivtransformer-env\Scripts\activate
# Linux/macOS:
source vivtransformer-env/bin/activate

# 3. 升级pip
python -m pip install --upgrade pip
```

### 🐳 方法3：使用Docker {#方法3-使用docker}

```bash
# 1. 安装Docker
# 下载地址：https://www.docker.com/get-started

# 2. 拉取预配置镜像
docker pull pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel

# 3. 运行容器
docker run -it --gpus all -v $(pwd):/workspace pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel bash
```

## 项目安装 {#项目安装}

### 📥 步骤1：克隆项目 {#步骤1-克隆项目}

```bash
# 克隆主仓库
git clone https://github.com/Pitohuie-AIversion/Sparse_to_Dense_Transformer.git

# 进入项目目录
cd Sparse_to_Dense_Transformer

# 查看项目结构
ls -la
```

### 📦 步骤2：安装依赖 {#步骤2-安装依赖}

#### 基础安装 {#基础安装}

```bash
# 安装核心依赖
pip install -r modify_multi_attention/requirements.txt

# 或者使用conda安装PyTorch
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

#### 开发环境安装 {#开发环境安装}

```bash
# 安装开发依赖（包含测试、文档工具等）
pip install -r requirements-dev.txt

# 安装预提交钩子
pre-commit install
```

#### 可选组件安装 {#可选组件安装}

```bash
# 安装可视化工具
pip install tensorboard wandb matplotlib seaborn

# 安装Jupyter支持
pip install jupyter ipykernel
python -m ipykernel install --user --name vivtransformer

# 安装性能分析工具
pip install line-profiler memory-profiler
```

### ✅ 步骤3：验证安装 {#步骤3-验证安装}

```bash
# 检查Python包
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import numpy; print(f'NumPy: {numpy.__version__}')"
python -c "import yaml; print('YAML: OK')"

# 检查CUDA支持
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA devices: {torch.cuda.device_count()}')"

# 运行快速测试
python -m modify_multi_attention.main --help
```

## GPU环境配置 {#gpu环境配置}

### 🎮 NVIDIA GPU设置 {#nvidia-gpu设置}

#### 1. 驱动安装 {#1-驱动安装}

```bash
# 检查当前驱动
nvidia-smi

# 如果没有输出，需要安装NVIDIA驱动
# Windows: 从NVIDIA官网下载
# Ubuntu: sudo apt install nvidia-driver-525
```

#### 2. CUDA安装 {#2-cuda安装}

```bash
# 检查CUDA版本
nvcc --version

# 如果没有安装，下载CUDA Toolkit
# 下载地址：https://developer.nvidia.com/cuda-downloads

# 验证CUDA安装
cd /usr/local/cuda/samples/1_Utilities/deviceQuery
sudo make
./deviceQuery
```

#### 3. cuDNN安装 {#3-cudnn安装}

```bash
# 下载cuDNN（需要NVIDIA开发者账号）
# 下载地址：https://developer.nvidia.com/cudnn

# 解压并复制文件
tar -xzvf cudnn-linux-x86_64-8.x.x.x_cudaX.X-archive.tar.xz
sudo cp cuda/include/cudnn*.h /usr/local/cuda/include
sudo cp cuda/lib64/libcudnn* /usr/local/cuda/lib64
sudo chmod a+r /usr/local/cuda/include/cudnn*.h /usr/local/cuda/lib64/libcudnn*
```

### 🔧 环境变量配置 {#环境变量配置}

#### Linux/macOS {#linux-macos}

```bash
# 添加到 ~/.bashrc 或 ~/.zshrc
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# 重新加载配置
source ~/.bashrc
```

#### Windows {#windows}

```powershell
# 添加到系统环境变量
$env:CUDA_PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8"
$env:PATH += ";$env:CUDA_PATH\bin"
```

## 验证安装 {#验证安装}

### 🧪 运行测试套件 {#运行测试套件}

```bash
# 运行单元测试
python -m pytest modify_multi_attention/tests/ -v

# 运行注意力机制测试
python modify_multi_attention/attention_test.py

# 运行端到端测试
python -m modify_multi_attention.main --loss_idx 0 --debug
```

### 📊 性能基准测试 {#性能基准测试}

```bash
# CPU性能测试
python -c "
import torch
import time
x = torch.randn(1000, 1000)
start = time.time()
y = torch.mm(x, x)
print(f'CPU计算时间: {time.time() - start:.4f}秒')
"

# GPU性能测试（如果可用）
python -c "
import torch
import time
if torch.cuda.is_available():
    x = torch.randn(1000, 1000).cuda()
    start = time.time()
    y = torch.mm(x, x)
    torch.cuda.synchronize()
    print(f'GPU计算时间: {time.time() - start:.4f}秒')
else:
    print('GPU不可用')
"
```

## 常见问题 {#常见问题}

### ❌ 安装问题 {#安装问题}

#### 问题1：pip安装失败 {#问题1-pip安装失败}

```bash
# 解决方案：升级pip和setuptools
pip install --upgrade pip setuptools wheel

# 使用国内镜像源
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/
```

#### 问题2：CUDA版本不匹配 {#问题2-cuda版本不匹配}

```bash
# 检查CUDA版本
nvidia-smi  # 查看驱动支持的最高CUDA版本
nvcc --version  # 查看安装的CUDA版本

# 安装匹配的PyTorch版本
# 访问：https://pytorch.org/get-started/locally/
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### 问题3：内存不足 {#问题3-内存不足}

```bash
# 减少批大小
# 修改 modify_multi_attention/configs/config.yaml
data:
  batch_size: 32  # 从128减少到32

# 使用梯度累积
training:
  gradient_accumulation_steps: 4
```

### 🔧 配置问题 {#配置问题}

#### 问题4：找不到配置文件 {#问题4-找不到配置文件}

```bash
# 检查配置文件路径
ls modify_multi_attention/configs/

# 使用绝对路径
python -m modify_multi_attention.main --config /absolute/path/to/config.yaml
```

#### 问题5：数据路径错误 {#问题5-数据路径错误}

```bash
# 检查数据文件
ls modify_multi_attention/data/

# 修改配置文件中的数据路径
# modify_multi_attention/configs/config.yaml
data:
  path: "modify_multi_attention/data/your_data.pt"
```

## 卸载指南 {#卸载指南}

### 🗑️ 完全卸载 {#完全卸载}

```bash
# 1. 停用虚拟环境
conda deactivate  # 或 deactivate

# 2. 删除虚拟环境
conda env remove -n vivtransformer
# 或删除venv目录
rm -rf vivtransformer-env/

# 3. 删除项目文件
rm -rf Sparse_to_Dense_Transformer/

# 4. 清理pip缓存
pip cache purge

# 5. 清理conda缓存
conda clean --all
```

### 🔄 重新安装 {#重新安装}

```bash
# 如果遇到问题，可以重新安装
# 1. 备份配置和数据
cp -r modify_multi_attention/configs/ ~/backup/
cp -r modify_multi_attention/data/ ~/backup/

# 2. 完全卸载（参考上面步骤）

# 3. 重新安装（从步骤1开始）
```

## 📞 获取帮助 {#获取帮助}

如果在安装过程中遇到问题：

- 📖 查看 [故障排除指南](troubleshooting.html)
- 🐛 提交 [GitHub Issue](https://github.com/Pitohuie-AIversion/Sparse_to_Dense_Transformer/issues)
- 💬 参与 [讨论区](https://github.com/Pitohuie-AIversion/Sparse_to_Dense_Transformer/discussions)
- 📧 发送邮件至：1748492875@qq.com

---

*最后更新：{{ site.time | date: "%Y-%m-%d" }}*