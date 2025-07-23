# Linux服务器端可视化训练部署指南

## 📋 系统要求

### 基础环境
- **操作系统**: Ubuntu 18.04+ / CentOS 7+ / RHEL 7+
- **Python**: 3.7+
- **内存**: 建议 8GB+
- **存储**: 建议 10GB+ 可用空间
- **GPU**: 可选，支持CUDA的NVIDIA GPU

### 必要软件包
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install python3 python3-pip python3-venv git

# CentOS/RHEL
sudo yum update
sudo yum install python3 python3-pip git
```

## 🚀 快速部署

### 1. 环境准备

```bash
# 创建虚拟环境
python3 -m venv viv_env
source viv_env/bin/activate

# 升级pip
pip install --upgrade pip

# 安装必要依赖
pip install torch torchvision matplotlib numpy seaborn pandas psutil
```

### 2. 项目部署

```bash
# 进入项目目录
cd pdebench_extended

# 给脚本添加执行权限
chmod +x *.sh

# 检查Python环境
python3 -c "import torch, matplotlib, numpy, seaborn; print('Environment OK')"
```

### 3. 运行可视化训练

#### 方式一：增强版可视化训练
```bash
# 直接运行脚本
./启动增强版可视化训练.sh

# 或者手动运行
python3 enhanced_visual_training.py --epochs 20 --output-dir enhanced_training_output
```

#### 方式二：独立预测可视化
```bash
# 直接运行脚本
./启动预测可视化演示.sh

# 或者手动运行
python3 create_prediction_visualization.py
```

## 📊 输出文件说明

### 增强版训练输出结构
```
enhanced_training_output/
├── plots/                    # 传统训练可视化
│   └── visualizations/
├── predictions/              # 预测结果可视化
│   ├── single_samples/       # 单样本预测对比
│   ├── batch_comparisons/    # 批次预测对比
│   ├── error_analysis/       # 误差分析
│   └── time_series/          # 训练进度跟踪
├── logs/                     # 训练日志
├── hardware_logs/            # 硬件监控数据
├── comprehensive_training_report.md  # 综合报告
└── training_report.html      # HTML报告
```

### 预测可视化输出结构
```
demo_prediction_visualizations/
├── single_samples/           # 单样本预测对比
├── batch_comparisons/        # 批次预测对比
├── error_analysis/           # 误差分析
├── time_series/              # 训练进度跟踪
└── epoch_10_visualization_report.md  # 可视化报告
```

## 🔧 服务器端特殊配置

### 1. 无显示器环境配置

如果服务器没有图形界面，需要配置matplotlib使用非交互式后端：

```bash
# 方法1：设置环境变量
export MPLBACKEND=Agg

# 方法2：在Python脚本开头添加
# import matplotlib
# matplotlib.use('Agg')
```

### 2. 内存优化配置

对于内存有限的服务器：

```bash
# 限制OpenMP线程数
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# 设置PyTorch线程数
export TORCH_NUM_THREADS=1
```

### 3. GPU配置（可选）

如果服务器有GPU：

```bash
# 检查CUDA可用性
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# 安装GPU版本的PyTorch（如需要）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## 📈 远程访问可视化结果

### 1. 使用SCP下载结果

```bash
# 从服务器下载可视化结果到本地
scp -r username@server_ip:/path/to/enhanced_training_output ./local_results/
```

### 2. 使用HTTP服务器查看

```bash
# 在输出目录启动简单HTTP服务器
cd enhanced_training_output
python3 -m http.server 8000

# 然后在浏览器访问: http://server_ip:8000
```

### 3. 使用Jupyter Notebook

```bash
# 安装Jupyter
pip install jupyter

# 启动Jupyter服务器
jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root

# 在浏览器访问: http://server_ip:8888
```

## 🛠️ 故障排除

### 常见问题

1. **权限错误**
   ```bash
   chmod +x *.sh
   ```

2. **Python包缺失**
   ```bash
   pip install -r requirements.txt
   ```

3. **显示错误**
   ```bash
   export DISPLAY=:0.0  # 如果有X11转发
   export MPLBACKEND=Agg  # 无显示器环境
   ```

4. **内存不足**
   ```bash
   # 减少批次大小或训练轮数
   python3 enhanced_visual_training.py --epochs 10 --batch-size 4
   ```

### 日志检查

```bash
# 查看训练日志
tail -f enhanced_training_output/logs/training_log.txt

# 查看系统资源使用
top
htop
nvidia-smi  # 如果有GPU
```

## 📝 自定义配置

### 修改训练参数

```bash
# 自定义训练轮数和输出目录
python3 enhanced_visual_training.py --epochs 50 --output-dir my_training_results

# 自定义批次大小
python3 enhanced_visual_training.py --batch-size 16
```

### 修改可视化参数

编辑 `enhanced_visual_training.py` 文件中的参数：

```python
# 修改可视化频率
if epoch % 3 == 0:  # 每3个epoch记录一次预测
    training_logger.log_prediction(...)

# 修改批次可视化频率
if epoch % 10 == 0:  # 每10个epoch记录一次批次预测
    training_logger.log_batch_predictions(...)
```

## 🔄 自动化部署脚本

创建一键部署脚本 `deploy.sh`：

```bash
#!/bin/bash

# 一键部署脚本
echo "开始部署VIVTransformer可视化训练环境..."

# 创建虚拟环境
python3 -m venv viv_env
source viv_env/bin/activate

# 安装依赖
pip install --upgrade pip
pip install torch torchvision matplotlib numpy seaborn pandas psutil

# 设置权限
chmod +x *.sh

# 设置环境变量
echo 'export MPLBACKEND=Agg' >> ~/.bashrc
echo 'export OMP_NUM_THREADS=1' >> ~/.bashrc

echo "部署完成！可以运行 ./启动增强版可视化训练.sh 开始训练"
```

## 📞 技术支持

如果在Linux服务器部署过程中遇到问题，请检查：

1. **系统兼容性**: 确保Linux发行版和Python版本符合要求
2. **网络连接**: 确保可以下载Python包
3. **存储空间**: 确保有足够的磁盘空间
4. **权限设置**: 确保有执行脚本和写入文件的权限

---

**项目**: VIVTransformer 压力场重建  
**版本**: Linux Server Edition v1.0  
**更新时间**: 2025-07-23