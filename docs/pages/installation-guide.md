---
layout: default
title: Installation Guide
permalink: /pages/installation-guide/
---

# Installation Guide

This guide provides detailed instructions on how to install and configure the VIVTransformer project in different environments.

## Table of Contents

- [System Requirements](#system-requirements)
- [Python Environment Setup](#python-environment-setup)
- [Project Installation](#project-installation)
- [GPU Environment Setup](#gpu-environment-setup)
- [Installation Verification](#installation-verification)
- [Common Issues](#common-issues)
- [Uninstallation Guide](#uninstallation-guide)

## System Requirements

### Hardware Requirements

| Component | Minimum | Recommended | Notes |
|-----------|---------|-------------|-------|
| **CPU** | Intel i5 / AMD Ryzen 5 | Intel i7 / AMD Ryzen 7 | AVX instruction set support |
| **Memory** | 8GB RAM | 16GB+ RAM | Large models require more memory |
| **Storage** | 5GB available space | 20GB+ SSD | Including datasets and model storage |
| **GPU** | Optional | NVIDIA RTX 3060+ | 8GB+ VRAM recommended |

### Software Requirements

| Software | Version | Notes |
|----------|---------|-------|
| **Operating System** | Windows 10+, Ubuntu 18.04+, macOS 10.15+ | 64-bit system |
| **Python** | 3.8 - 3.11 | Recommended 3.9 or 3.10 |
| **Git** | 2.20+ | For repository cloning |
| **CUDA** | 11.8+ (Optional) | Required for GPU acceleration |
| **cuDNN** | 8.6+ (Optional) | Used with CUDA |

## Python Environment Setup

### Method 1: Using Conda (Recommended)

```bash
# 1. Install Miniconda or Anaconda
# Download from: https://docs.conda.io/en/latest/miniconda.html

# 2. Create virtual environment
conda create -n vivtransformer python=3.9

# 3. Activate environment
conda activate vivtransformer

# 4. Verify Python version
python --version
# Should display: Python 3.9.x
```

### Method 2: Using venv

```bash
# 1. Create virtual environment
python -m venv vivtransformer-env

# 2. Activate environment
# Windows:
vivtransformer-env\Scripts\activate
# Linux/macOS:
source vivtransformer-env/bin/activate

# 3. Upgrade pip
python -m pip install --upgrade pip
```

### Method 3: Using Docker

```bash
# 1. Install Docker
# Download from: https://www.docker.com/get-started

# 2. Pull pre-configured image
docker pull pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel

# 3. Run container
docker run -it --gpus all -v $(pwd):/workspace pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel bash
```

## Project Installation

### Step 1: Clone Repository

```bash
# Clone main repository
git clone https://github.com/Pitohuie-AIversion/Sparse_to_Dense_Transformer.git

# Enter project directory
cd Sparse_to_Dense_Transformer

# View project structure
ls -la
```

### Step 2: Install Dependencies

#### Basic Installation

```bash
# Install core dependencies
pip install -r modify_multi_attention/requirements.txt

# Or install PyTorch using conda
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

#### Development Environment Installation

```bash
# Install development dependencies (including testing, documentation tools, etc.)
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install
```

#### Optional Components Installation

```bash
# Install visualization tools
pip install tensorboard wandb matplotlib seaborn

# Install Jupyter support
pip install jupyter ipykernel
python -m ipykernel install --user --name vivtransformer

# Install performance profiling tools
pip install line-profiler memory-profiler
```

### Step 3: Verify Installation

```bash
# Check Python packages
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import numpy; print(f'NumPy: {numpy.__version__}')"
python -c "import yaml; print('YAML: OK')"

# Check CUDA support
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA devices: {torch.cuda.device_count()}')"

# Run quick test
python -m modify_multi_attention.main --help
```

## GPU Environment Setup

### NVIDIA GPU Setup

#### 1. Driver Installation

```bash
# Check current driver
nvidia-smi

# If no output, install NVIDIA driver
# Windows: Download from NVIDIA website
# Ubuntu: sudo apt install nvidia-driver-525
```

#### 2. CUDA Installation

```bash
# Check CUDA version
nvcc --version

# If not installed, download CUDA Toolkit
# Download from: https://developer.nvidia.com/cuda-downloads

# Verify CUDA installation
cd /usr/local/cuda/samples/1_Utilities/deviceQuery
sudo make
./deviceQuery
```

#### 3. cuDNN Installation

```bash
# Download cuDNN (requires NVIDIA developer account)
# Download from: https://developer.nvidia.com/cudnn

# Extract and copy files
tar -xzvf cudnn-linux-x86_64-8.x.x.x_cudaX.X-archive.tar.xz
sudo cp cuda/include/cudnn*.h /usr/local/cuda/include
sudo cp cuda/lib64/libcudnn* /usr/local/cuda/lib64
sudo chmod a+r /usr/local/cuda/include/cudnn*.h /usr/local/cuda/lib64/libcudnn*
```

### Environment Variables Configuration

#### Linux/macOS

```bash
# Add to ~/.bashrc or ~/.zshrc
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Reload configuration
source ~/.bashrc
```

#### Windows

```powershell
# Add to system environment variables
$env:CUDA_PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8"
$env:PATH += ";$env:CUDA_PATH\bin"
```

## Installation Verification

### Run Test Suite

```bash
# Run unit tests
python -m pytest modify_multi_attention/tests/ -v

# Run attention mechanism tests
python modify_multi_attention/attention_test.py

# Run end-to-end tests
python -m modify_multi_attention.main --loss_idx 0 --debug
```

### Performance Benchmark Tests

```bash
# CPU performance test
python -c "
import torch
import time
x = torch.randn(1000, 1000)
start = time.time()
y = torch.mm(x, x)
print(f'CPU computation time: {time.time() - start:.4f} seconds')
"

# GPU performance test (if available)
python -c "
import torch
import time
if torch.cuda.is_available():
    x = torch.randn(1000, 1000).cuda()
    start = time.time()
    y = torch.mm(x, x)
    torch.cuda.synchronize()
    print(f'GPU computation time: {time.time() - start:.4f} seconds')
else:
    print('GPU not available')
"
```

## Common Issues

### Installation Problems

#### Issue 1: pip Installation Failure

```bash
# Solution: Upgrade pip and setuptools
pip install --upgrade pip setuptools wheel

# Use domestic mirror source
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/
```

#### Issue 2: CUDA Version Mismatch

```bash
# Check CUDA version
nvidia-smi  # Check maximum CUDA version supported by driver
nvcc --version  # Check installed CUDA version

# Install matching PyTorch version
# Visit: https://pytorch.org/get-started/locally/
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### Issue 3: Out of Memory

```bash
# Reduce batch size
# Modify modify_multi_attention/configs/config.yaml
data:
  batch_size: 32  # Reduce from 128 to 32

# Use gradient accumulation
training:
  gradient_accumulation_steps: 4
```

### Configuration Problems

#### Issue 4: Configuration File Not Found

```bash
# Check configuration file path
ls modify_multi_attention/configs/

# Use absolute path
python -m modify_multi_attention.main --config /absolute/path/to/config.yaml
```

#### Issue 5: Data Path Error

```bash
# Check data files
ls modify_multi_attention/data/

# Modify data path in configuration file
# modify_multi_attention/configs/config.yaml
data:
  path: "modify_multi_attention/data/your_data.pt"
```

## Uninstallation Guide

### Complete Uninstallation

```bash
# 1. Deactivate virtual environment
conda deactivate  # or deactivate

# 2. Remove virtual environment
conda env remove -n vivtransformer
# or remove venv directory
rm -rf vivtransformer-env/

# 3. Remove project files
rm -rf Sparse_to_Dense_Transformer/

# 4. Clear pip cache
pip cache purge

# 5. Clear conda cache
conda clean --all
```

### Reinstallation

```bash
# If you encounter problems, you can reinstall
# 1. Backup configuration and data
cp -r modify_multi_attention/configs/ ~/backup/
cp -r modify_multi_attention/data/ ~/backup/

# 2. Complete uninstallation (refer to steps above)

# 3. Reinstall (start from step 1)
```

## Getting Help

If you encounter problems during installation:

- 📖 Check [Troubleshooting Guide](troubleshooting)
- 🐛 Submit [GitHub Issue](https://github.com/Pitohuie-AIversion/Sparse_to_Dense_Transformer/issues)
- 💬 Join [Discussions](https://github.com/Pitohuie-AIversion/Sparse_to_Dense_Transformer/discussions)
- 📧 Send email to: 1748492875@qq.com