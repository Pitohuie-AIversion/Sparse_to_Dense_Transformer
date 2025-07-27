# 服务器端部署和运行指南

本指南帮助您在Linux服务器上解决`ModuleNotFoundError: No module named 'multiscale.data'`错误并成功运行VIVTransformer训练。

## 问题描述

在服务器环境中运行`train_configurable_multiscale.py`时，可能遇到以下错误：
```
ModuleNotFoundError: No module named 'multiscale.data'
```

这通常是由于Python模块路径配置问题导致的。

## 解决方案

### 方法1: 使用自动修复脚本（推荐）

1. **运行修复脚本**：
   ```bash
   cd /path/to/pdebench_extended
   python fix_server_import.py
   ```

2. **使用便捷运行脚本**：
   ```bash
   chmod +x run_server_training.sh
   ./run_server_training.sh
   ```

### 方法2: 手动设置环境

1. **设置Python路径**：
   ```bash
   export PYTHONPATH="$(pwd):$(pwd)/multiscale:$(pwd)/multiscale/data:$PYTHONPATH"
   ```

2. **验证文件结构**：
   确保以下文件存在：
   ```
   pdebench_extended/
   ├── multiscale/
   │   ├── __init__.py
   │   └── data/
   │       ├── __init__.py
   │       └── multiscale_adapter.py
   ├── mymodels/
   │   ├── __init__.py
   │   └── transformer.py
   └── train_configurable_multiscale.py
   ```

3. **运行训练**：
   ```bash
   python train_configurable_multiscale.py --scale_factor 4 --num_epochs 20 --batch_size 4
   ```

### 方法3: 使用改进的训练脚本

新版本的`train_configurable_multiscale.py`已经包含了多种导入方式的容错处理，可以直接运行：

```bash
python train_configurable_multiscale.py [参数]
```

## 常用训练命令

### 快速测试
```bash
python train_configurable_multiscale.py --scale_factor 4 --num_epochs 3 --batch_size 2
```

### 标准训练
```bash
python train_configurable_multiscale.py --scale_factor 4 --num_epochs 20 --batch_size 4
```

### 自定义分辨率训练
```bash
python train_configurable_multiscale.py --scale_factor 4 --input_resolution 9 9 --output_resolution 128 128 --num_epochs 20 --batch_size 4
```

### 长时间训练
```bash
python train_configurable_multiscale.py --scale_factor 4 --num_epochs 100 --batch_size 4
```

## 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--scale_factor` | 4 | 多尺度缩放因子 |
| `--input_resolution` | [32, 32] | 输入分辨率 |
| `--output_resolution` | [128, 128] | 输出分辨率 |
| `--num_epochs` | 50 | 训练轮数 |
| `--batch_size` | 4 | 批次大小 |
| `--learning_rate` | 1e-4 | 学习率 |
| `--dataset_name` | darcy_flow | 数据集名称 |
| `--data_root` | ./data | 数据根目录 |

## 故障排除

### 1. 导入错误
如果仍然遇到导入错误，请检查：
- 确保在正确的目录下运行脚本
- 检查所有必要的文件是否存在
- 运行`fix_server_import.py`进行诊断

### 2. 内存不足
如果遇到内存不足错误，尝试：
- 减小`batch_size`参数
- 使用较小的输入/输出分辨率

### 3. CUDA错误
如果遇到CUDA相关错误：
- 检查GPU可用性：`nvidia-smi`
- 确保PyTorch CUDA版本匹配

### 4. 数据集问题
如果数据集加载失败：
- 检查数据目录是否存在
- 确保有足够的磁盘空间
- 检查网络连接（如果需要下载数据）

## 监控训练进度

1. **查看日志**：
   ```bash
   tail -f logs/training_*.log
   ```

2. **使用TensorBoard**：
   ```bash
   tensorboard --logdir=logs
   ```
   然后在浏览器中访问 `http://localhost:6006`

## 联系支持

如果遇到其他问题，请：
1. 运行`fix_server_import.py`获取详细的诊断信息
2. 检查错误日志
3. 提供完整的错误信息和环境配置

---

**注意**：确保在运行训练之前已经安装了所有必要的依赖包，包括PyTorch、NumPy、h5py等。