# 服务器部署指南

本指南帮助您在服务器上部署和运行多尺度VIVTransformer训练。

## 📋 目录

- [环境准备](#环境准备)
- [文件传输](#文件传输)
- [配置设置](#配置设置)
- [运行训练](#运行训练)
- [监控训练](#监控训练)
- [结果获取](#结果获取)
- [故障排除](#故障排除)

## 🔧 环境准备

### 1. 检查系统要求

```bash
# 检查GPU
nvidia-smi

# 检查Python版本
python --version  # 需要 Python 3.8+

# 检查CUDA版本
nvcc --version
```

### 2. 安装依赖

```bash
# 创建虚拟环境
conda create -n vivtransformer python=3.9
conda activate vivtransformer

# 安装PyTorch (根据CUDA版本选择)
# CUDA 11.8
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# 安装其他依赖
pip install -r requirements.txt
```

### 3. 验证安装

```bash
python -c "import torch; print(f'PyTorch版本: {torch.__version__}'); print(f'CUDA可用: {torch.cuda.is_available()}'); print(f'GPU数量: {torch.cuda.device_count()}')"
```

## 📁 文件传输

### 1. 上传项目文件

```bash
# 使用scp上传整个项目
scp -r ./pdebench_extended username@server_ip:/path/to/project/

# 或使用rsync (推荐)
rsync -avz --progress ./pdebench_extended/ username@server_ip:/path/to/project/pdebench_extended/
```

### 2. 上传数据文件

```bash
# 上传PDEBench数据
scp PDEBench/pdebench/data_download/2D_DarcyFlow_beta0.1_Train.hdf5 username@server_ip:/path/to/data/
```

### 3. 设置权限

```bash
# 在服务器上设置执行权限
chmod +x /path/to/project/pdebench_extended/*.py
chmod +x /path/to/project/pdebench_extended/*.sh
```

## ⚙️ 配置设置

### 1. 修改数据路径

编辑配置文件，更新数据路径：

```bash
# 编辑配置文件
vim /path/to/project/pdebench_extended/configs/multiscale_configs.json
```

将 `data_path` 更新为服务器上的实际路径：

```json
{
  "scale_factor_4": {
    "data": {
      "data_path": "/path/to/data/2D_DarcyFlow_beta0.1_Train.hdf5",
      ...
    }
  }
}
```

### 2. 创建服务器专用配置

```bash
# 创建服务器配置文件
cp configs/multiscale_configs.json configs/server_config.json
```

编辑 `server_config.json`，调整以下参数：

```json
{
  "server_high_performance": {
    "data": {
      "data_path": "/path/to/data/2D_DarcyFlow_beta0.1_Train.hdf5",
      "batch_size": 32,  # 根据GPU内存调整
      "num_workers": 8   # 根据CPU核心数调整
    },
    "training": {
      "num_epochs": 100,
      "learning_rate": 1e-4
    },
    "output_dir": "/path/to/results/server_training_results"
  }
}
```

## 🚀 运行训练

### 1. 快速开始

```bash
# 进入项目目录
cd /path/to/project/pdebench_extended

# 激活环境
conda activate vivtransformer

# 查看可用配置
python run_multiscale_training.py --list_presets

# 使用预设配置运行
python run_multiscale_training.py --preset scale_factor_4
```

### 2. 自定义参数运行

```bash
# 指定缩放因子和训练轮数
python run_multiscale_training.py --scale_factor 4 --num_epochs 50 --batch_size 16

# 使用自定义配置文件
python run_multiscale_training.py --config configs/server_config.json

# 指定GPU
python run_multiscale_training.py --preset scale_factor_4 --gpu 0
```

### 3. 后台运行

```bash
# 使用nohup后台运行
nohup python run_multiscale_training.py --preset server_config_high_performance > training.log 2>&1 &

# 或使用screen
screen -S vivtransformer_training
python run_multiscale_training.py --preset server_config_high_performance
# Ctrl+A, D 分离会话

# 重新连接screen会话
screen -r vivtransformer_training
```

### 4. 使用tmux (推荐)

```bash
# 创建tmux会话
tmux new-session -d -s vivtransformer

# 进入会话
tmux attach-session -t vivtransformer

# 运行训练
python run_multiscale_training.py --preset server_config_high_performance

# 分离会话: Ctrl+B, D
# 重新连接: tmux attach-session -t vivtransformer
```

## 📊 监控训练

### 1. 实时监控

```bash
# 查看训练日志
tail -f /path/to/results/training.log

# 监控GPU使用情况
watch -n 1 nvidia-smi

# 监控系统资源
htop
```

### 2. TensorBoard监控

```bash
# 在服务器上启动TensorBoard
tensorboard --logdir /path/to/results/tensorboard --host 0.0.0.0 --port 6006

# 在本地浏览器访问 (需要端口转发)
# ssh -L 6006:localhost:6006 username@server_ip
# 然后访问 http://localhost:6006
```

### 3. 远程监控脚本

创建监控脚本 `monitor_training.py`：

```python
#!/usr/bin/env python3
import time
import json
import psutil
import GPUtil
from pathlib import Path

def monitor_training(result_dir):
    """监控训练状态"""
    result_path = Path(result_dir)
    stats_file = result_path / 'training_stats.json'
    
    while True:
        # 检查训练状态
        if stats_file.exists():
            with open(stats_file, 'r') as f:
                stats = json.load(f)
            print(f"当前epoch: {len(stats.get('train_losses', []))}")
            if stats.get('train_losses'):
                print(f"最新训练损失: {stats['train_losses'][-1]:.6f}")
            if stats.get('val_losses'):
                print(f"最新验证损失: {stats['val_losses'][-1]:.6f}")
        
        # 系统资源
        print(f"CPU使用率: {psutil.cpu_percent()}%")
        print(f"内存使用率: {psutil.virtual_memory().percent}%")
        
        # GPU状态
        gpus = GPUtil.getGPUs()
        for gpu in gpus:
            print(f"GPU {gpu.id}: {gpu.memoryUtil*100:.1f}% 内存, {gpu.load*100:.1f}% 使用率")
        
        print("-" * 50)
        time.sleep(60)  # 每分钟检查一次

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        monitor_training(sys.argv[1])
    else:
        print("用法: python monitor_training.py <result_directory>")
```

## 📥 结果获取

### 1. 下载训练结果

```bash
# 下载整个结果目录
scp -r username@server_ip:/path/to/results/ ./local_results/

# 只下载重要文件
scp username@server_ip:/path/to/results/best_checkpoint.pth ./
scp username@server_ip:/path/to/results/training_stats.json ./
scp -r username@server_ip:/path/to/results/visualizations/ ./
```

### 2. 压缩后下载

```bash
# 在服务器上压缩
tar -czf training_results.tar.gz /path/to/results/

# 下载压缩文件
scp username@server_ip:/path/to/training_results.tar.gz ./

# 本地解压
tar -xzf training_results.tar.gz
```

## 🔧 故障排除

### 1. 常见错误

#### CUDA内存不足
```bash
# 减少批次大小
python run_multiscale_training.py --batch_size 4

# 或修改配置文件中的batch_size
```

#### 数据文件未找到
```bash
# 检查数据文件路径
ls -la /path/to/data/2D_DarcyFlow_beta0.1_Train.hdf5

# 更新配置文件中的data_path
```

#### 权限问题
```bash
# 设置正确的权限
chmod -R 755 /path/to/project/
chown -R username:usergroup /path/to/project/
```

### 2. 性能优化

#### GPU优化
```bash
# 设置GPU内存增长
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# 使用混合精度训练 (需要修改代码)
# 在训练脚本中添加 autocast 和 GradScaler
```

#### 数据加载优化
```json
{
  "data": {
    "num_workers": 8,     // 根据CPU核心数调整
    "pin_memory": true,   // GPU训练时启用
    "persistent_workers": true  // 保持worker进程
  }
}
```

### 3. 调试模式

```bash
# 干运行模式，只检查配置
python run_multiscale_training.py --preset scale_factor_4 --dry_run

# 使用小数据集测试
python run_multiscale_training.py --num_epochs 1 --batch_size 2
```

## 📝 最佳实践

1. **定期备份**: 设置定期备份检查点和结果
2. **监控资源**: 持续监控GPU和内存使用情况
3. **日志记录**: 保存详细的训练日志
4. **版本控制**: 记录使用的代码版本和配置
5. **实验管理**: 为每次实验创建唯一的输出目录

## 🔗 相关文件

- `train_configurable_multiscale.py`: 主训练脚本
- `run_multiscale_training.py`: 启动脚本
- `configs/multiscale_configs.json`: 预设配置
- `requirements.txt`: 依赖列表

## 📞 支持

如果遇到问题，请检查：
1. 训练日志文件
2. 系统资源使用情况
3. 配置文件设置
4. 数据文件完整性

---

**注意**: 请根据您的具体服务器环境调整路径和配置参数。