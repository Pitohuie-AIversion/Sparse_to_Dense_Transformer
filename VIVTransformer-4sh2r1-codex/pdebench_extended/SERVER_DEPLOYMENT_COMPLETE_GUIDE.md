# VIVTransformer 服务器部署完整指南

## 📋 概述

本指南提供了在Linux服务器上部署和运行VIVTransformer多尺度训练系统的完整解决方案，包括Python脚本工具和详细的操作说明。

## 🎯 部署目标

- ✅ 在Linux服务器上快速部署VIVTransformer
- ✅ 支持多尺度训练 (9x9 → 128x128)
- ✅ 提供可视化训练和监控
- ✅ 支持远程访问和结果下载
- ✅ 自动化环境配置和依赖管理

## 🛠️ 系统要求

### 基础环境
- **操作系统**: Ubuntu 18.04+ / CentOS 7+ / RHEL 7+
- **Python**: 3.7+
- **内存**: 建议 8GB+
- **存储**: 建议 20GB+ 可用空间
- **GPU**: 可选，支持CUDA的NVIDIA GPU

### 网络要求
- 能够访问PyPI (pip安装)
- 如需远程访问TensorBoard，开放6006端口

## 🚀 快速部署

### 方法一：使用Python部署工具（推荐）

```bash
# 1. 进入项目目录
cd pdebench_extended

# 2. 运行部署工具
python3 server_deployment_toolkit.py --deploy

# 3. 启动训练
./run_training.sh
```

### 方法二：使用Shell脚本

```bash
# 1. 进入项目目录
cd pdebench_extended

# 2. 运行部署脚本
./deploy_linux.sh

# 3. 启动训练
./run_training.sh
```

### 方法三：手动部署

```bash
# 1. 创建虚拟环境
python3 -m venv viv_env
source viv_env/bin/activate

# 2. 安装依赖
pip install --upgrade pip
pip install torch torchvision matplotlib numpy seaborn pandas psutil tqdm tensorboard

# 3. 配置环境变量
export MPLBACKEND=Agg
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TORCH_NUM_THREADS=1

# 4. 启动训练
python train_configurable_multiscale.py --scale_factor 4 --num_epochs 20 --batch_size 4
```

## 📊 训练模式选择

### 1. 多尺度训练（主要功能）
```bash
# 标准多尺度训练 (9x9 → 128x128)
python train_configurable_multiscale.py --scale_factor 4 --num_epochs 20 --batch_size 4

# 自定义参数训练
python train_configurable_multiscale.py \
    --scale_factor 4 \
    --input_resolution 9 9 \
    --output_resolution 128 128 \
    --num_epochs 50 \
    --batch_size 8
```

### 2. 增强版可视化训练
```bash
# 带完整可视化的训练
python enhanced_visual_training.py --epochs 20 --output-dir enhanced_training_output
```

### 3. 独立预测可视化
```bash
# 仅生成预测可视化（需要预训练模型）
python create_prediction_visualization.py
```

## 📈 监控和可视化

### TensorBoard监控

```bash
# 启动TensorBoard（自动选择最新结果）
./start_tensorboard.sh

# 手动指定日志目录
tensorboard --logdir configurable_multiscale_results_sf4_20250126_123456/tensorboard --port 6006 --host 0.0.0.0

# 远程访问
# 浏览器打开: http://your_server_ip:6006
```

### 训练进度监控

```bash
# 实时查看训练日志
tail -f configurable_multiscale_results_*/logs/training.log

# 查看系统资源使用
htop
nvidia-smi  # 如果有GPU
```

## 📁 输出文件结构

### 多尺度训练输出
```
configurable_multiscale_results_sf4_YYYYMMDD_HHMMSS/
├── config.json                 # 训练配置
├── model_checkpoints/          # 模型检查点
│   ├── best_model.pth         # 最佳模型
│   ├── latest_model.pth       # 最新模型
│   └── epoch_*.pth            # 各轮次模型
├── tensorboard/               # TensorBoard日志
├── logs/                      # 训练日志
│   └── training.log
└── visualizations/            # 可视化结果
    ├── training_curves.png
    ├── sample_predictions.png
    └── error_analysis.png
```

### 增强版训练输出
```
enhanced_training_output/
├── plots/                     # 传统训练可视化
├── predictions/               # 预测结果可视化
│   ├── single_samples/        # 单样本预测对比
│   ├── batch_comparisons/     # 批次预测对比
│   ├── error_analysis/        # 误差分析
│   └── time_series/           # 训练进度跟踪
├── logs/                      # 训练日志
├── hardware_logs/             # 硬件监控数据
├── comprehensive_training_report.md  # 综合报告
└── training_report.html       # HTML报告
```

## 🔧 高级配置

### GPU配置

```bash
# 检查CUDA可用性
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# 安装GPU版本PyTorch（如需要）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 设置GPU设备
export CUDA_VISIBLE_DEVICES=0  # 使用第一个GPU
```

### 内存优化

```bash
# 限制线程数（减少内存使用）
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TORCH_NUM_THREADS=1

# 减少批次大小
python train_configurable_multiscale.py --batch_size 2
```

### 无显示器环境

```bash
# 设置matplotlib后端
export MPLBACKEND=Agg

# 或在Python脚本中设置
# import matplotlib
# matplotlib.use('Agg')
```

## 🌐 远程访问和结果获取

### 1. SCP下载结果

```bash
# 下载训练结果到本地
scp -r username@server_ip:/path/to/configurable_multiscale_results_* ./local_results/

# 下载特定文件
scp username@server_ip:/path/to/results/config.json ./
scp username@server_ip:/path/to/results/model_checkpoints/best_model.pth ./
```

### 2. HTTP服务器查看

```bash
# 在结果目录启动HTTP服务器
cd configurable_multiscale_results_sf4_20250126_123456
python3 -m http.server 8000

# 浏览器访问: http://server_ip:8000
```

### 3. Jupyter Notebook

```bash
# 安装并启动Jupyter
pip install jupyter
jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root

# 浏览器访问: http://server_ip:8888
```

## 🛠️ 故障排除

### 常见问题及解决方案

#### 1. 权限错误
```bash
# 解决方案
chmod +x *.sh
sudo chown -R $USER:$USER ./
```

#### 2. Python包安装失败
```bash
# 更新pip
pip install --upgrade pip

# 使用国内镜像
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple torch torchvision

# 清理缓存
pip cache purge
```

#### 3. 内存不足
```bash
# 减少批次大小
python train_configurable_multiscale.py --batch_size 2

# 减少训练轮数
python train_configurable_multiscale.py --num_epochs 10

# 监控内存使用
free -h
htop
```

#### 4. 显示相关错误
```bash
# 设置环境变量
export MPLBACKEND=Agg
export DISPLAY=:0.0  # 如果有X11转发

# 安装字体（如果需要）
sudo apt install fonts-dejavu-core
```

#### 5. CUDA相关问题
```bash
# 检查CUDA版本
nvcc --version
nvidia-smi

# 重新安装对应版本的PyTorch
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 日志检查

```bash
# 查看训练日志
tail -f configurable_multiscale_results_*/logs/training.log

# 查看系统日志
sudo journalctl -f

# 查看Python错误
python3 -c "import torch; print(torch.__version__)"
```

## 📋 部署检查清单

### 部署前检查
- [ ] Linux系统版本兼容
- [ ] Python 3.7+ 已安装
- [ ] 足够的磁盘空间 (20GB+)
- [ ] 网络连接正常
- [ ] 必要端口开放 (6006, 8000, 8888)

### 部署后验证
- [ ] 虚拟环境创建成功
- [ ] 所有依赖包安装完成
- [ ] 环境变量配置正确
- [ ] 启动脚本可执行
- [ ] 训练脚本运行正常
- [ ] TensorBoard可访问

### 功能测试
- [ ] 多尺度训练正常启动
- [ ] 模型保存和加载正常
- [ ] 可视化结果生成正常
- [ ] 远程访问功能正常

## 🔄 自动化脚本集

### 1. 一键部署脚本
```bash
# server_deployment_toolkit.py
# 提供完整的Python部署工具
python3 server_deployment_toolkit.py --deploy
```

### 2. 快速启动脚本
```bash
# run_training.sh
# 提供交互式训练模式选择
./run_training.sh
```

### 3. TensorBoard启动脚本
```bash
# start_tensorboard.sh
# 自动选择最新结果启动TensorBoard
./start_tensorboard.sh
```

### 4. 结果打包脚本
```bash
# 创建结果打包脚本
cat > package_results.sh << 'EOF'
#!/bin/bash
latest_dir=$(ls -td configurable_multiscale_results_* 2>/dev/null | head -1)
if [ -n "$latest_dir" ]; then
    tar -czf "${latest_dir}.tar.gz" "$latest_dir"
    echo "结果已打包: ${latest_dir}.tar.gz"
else
    echo "未找到训练结果"
fi
EOF
chmod +x package_results.sh
```

## 📞 技术支持

### 联系方式
- **项目**: VIVTransformer 多尺度训练系统
- **版本**: Server Deployment v1.0
- **更新时间**: 2025-01-26

### 问题报告
如果在部署过程中遇到问题，请提供以下信息：
1. Linux发行版和版本
2. Python版本
3. 错误日志
4. 系统资源情况
5. 网络环境

### 性能优化建议
1. **GPU加速**: 使用CUDA GPU可显著提升训练速度
2. **内存优化**: 根据服务器内存调整批次大小
3. **存储优化**: 使用SSD可提升数据加载速度
4. **网络优化**: 使用本地镜像源加速包安装

---

**🎉 恭喜！您已完成VIVTransformer服务器部署。开始您的多尺度训练之旅吧！**