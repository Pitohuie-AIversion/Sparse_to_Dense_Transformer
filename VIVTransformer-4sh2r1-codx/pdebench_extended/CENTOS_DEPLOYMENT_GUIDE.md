# CentOS 服务器部署指南

本指南详细介绍如何在 CentOS 服务器上部署和运行压力场训练系统。

## 📋 系统要求

### 最低配置
- **操作系统**: CentOS 7/8 或 RHEL 7/8
- **CPU**: 4核心以上
- **内存**: 8GB 以上
- **存储**: 50GB 可用空间
- **网络**: 稳定的互联网连接

### 推荐配置
- **操作系统**: CentOS 8 或更新版本
- **CPU**: 8核心以上
- **内存**: 16GB 以上
- **GPU**: NVIDIA GPU (可选，用于加速训练)
- **存储**: 100GB+ SSD

## 🚀 快速部署

### 1. 上传文件到服务器

```bash
# 使用 scp 上传项目文件
scp -r pdebench_extended/ user@server_ip:/home/user/

# 或使用 rsync
rsync -avz pdebench_extended/ user@server_ip:/home/user/pdebench_extended/
```

### 2. 运行一键部署脚本

```bash
# 登录服务器
ssh user@server_ip

# 进入项目目录
cd pdebench_extended

# 给脚本执行权限
chmod +x deploy_centos.sh

# 运行部署脚本
bash deploy_centos.sh
```

### 3. 上传数据文件

```bash
# 上传训练数据
scp your_data.pt user@server_ip:/home/user/pressure_field_training/
```

### 4. 开始训练

```bash
# 进入项目目录
cd /home/user/pressure_field_training

# 运行简化训练脚本
bash run_training_simple.sh
```

## 📁 脚本说明

### 1. `deploy_centos.sh` - 完整部署脚本

**功能**:
- 自动检测系统版本
- 安装系统依赖包
- 编译安装 Python 3.9
- 可选安装 CUDA
- 创建虚拟环境
- 安装 Python 依赖
- 创建运行脚本
- 配置防火墙
- 创建系统服务

**使用方法**:
```bash
bash deploy_centos.sh
```

### 2. `quick_start.sh` - 交互式启动脚本

**功能**:
- 交互式菜单
- 自动检测数据文件
- 支持前台/后台运行
- 集成 TensorBoard
- 系统监控

**使用方法**:
```bash
bash quick_start.sh

# 或直接执行特定功能
bash quick_start.sh train
bash quick_start.sh tensorboard
bash quick_start.sh test
```

### 3. `run_training_simple.sh` - 一键训练脚本

**功能**:
- 最简化的训练启动
- 自动环境检测
- 多种运行模式
- 自动数据文件搜索

**使用方法**:
```bash
# 使用默认数据路径
bash run_training_simple.sh

# 指定数据路径
DATA_PATH=/path/to/data.pt bash run_training_simple.sh
```

## 🔧 详细配置

### 环境变量配置

```bash
# 在 ~/.bashrc 中添加
export PYTHONPATH=/home/user/pressure_field_training/pdebench_extended:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0  # 如果有GPU
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

# 重新加载配置
source ~/.bashrc
```

### 数据路径配置

编辑配置文件 `pdebench_extended/configs/pressure_field_training.yaml`:

```yaml
data:
  path: "/home/user/pressure_field_training/your_data.pt"
  train_ratio: 0.8
  val_ratio: 0.1
  test_ratio: 0.1
```

### GPU 配置 (可选)

如果有多个 GPU，可以指定使用的 GPU:

```bash
# 使用第一个GPU
export CUDA_VISIBLE_DEVICES=0

# 使用多个GPU
export CUDA_VISIBLE_DEVICES=0,1

# 检查GPU状态
nvidia-smi
```

## 🖥️ 运行模式

### 1. 前台运行

适合调试和短时间训练:

```bash
bash run_training_simple.sh
# 选择模式 1
```

### 2. 后台运行

适合长时间训练:

```bash
bash run_training_simple.sh
# 选择模式 2

# 查看日志
tail -f outputs/pressure_field_*/training.log
```

### 3. Screen 会话运行 (推荐)

最稳定的运行方式:

```bash
bash run_training_simple.sh
# 选择模式 3

# 管理 Screen 会话
screen -ls                    # 查看所有会话
screen -r session_name        # 连接到会话
# Ctrl+A+D                    # 分离会话
screen -X -S session_name quit # 终止会话
```

## 📊 监控和可视化

### TensorBoard 监控

```bash
# 启动 TensorBoard
tensorboard --logdir=./outputs --host=0.0.0.0 --port=6006

# 访问地址
http://服务器IP:6006
```

### 系统监控

```bash
# CPU 和内存监控
htop

# GPU 监控
watch -n 1 nvidia-smi

# 磁盘使用
df -h

# 网络连接
netstat -tlnp | grep -E ':(6006|8888)'
```

### 训练进程监控

```bash
# 查看训练进程
ps aux | grep python | grep train_pressure_field

# 查看日志
tail -f outputs/pressure_field_*/training.log

# 实时监控脚本
bash monitor.sh
```

## 🔒 安全配置

### 防火墙设置

```bash
# 开放 TensorBoard 端口
sudo firewall-cmd --permanent --add-port=6006/tcp

# 开放 Jupyter 端口
sudo firewall-cmd --permanent --add-port=8888/tcp

# 重载防火墙
sudo firewall-cmd --reload

# 查看开放端口
sudo firewall-cmd --list-ports
```

### SSH 安全

```bash
# 修改 SSH 配置
sudo vim /etc/ssh/sshd_config

# 建议设置:
# Port 22222                    # 修改默认端口
# PermitRootLogin no           # 禁止root登录
# PasswordAuthentication no    # 禁用密码登录(使用密钥)

# 重启 SSH 服务
sudo systemctl restart sshd
```

## 🛠️ 故障排除

### 常见问题

#### 1. Python 版本问题

```bash
# 检查 Python 版本
python3.9 --version

# 如果没有 python3.9，创建软链接
sudo ln -sf /usr/local/bin/python3.9 /usr/local/bin/python3
```

#### 2. 虚拟环境问题

```bash
# 重新创建虚拟环境
rm -rf venv
python3.9 -m venv venv
source venv/bin/activate
pip install --upgrade pip
```

#### 3. CUDA 问题

```bash
# 检查 CUDA 安装
nvcc --version
nvidia-smi

# 检查 PyTorch CUDA 支持
python -c "import torch; print(torch.cuda.is_available())"
```

#### 4. 内存不足

```bash
# 检查内存使用
free -h

# 创建交换文件
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

#### 5. 磁盘空间不足

```bash
# 检查磁盘使用
df -h

# 清理临时文件
sudo yum clean all
rm -rf ~/.cache/pip
```

### 日志分析

```bash
# 查看系统日志
sudo journalctl -f

# 查看训练日志
tail -f outputs/pressure_field_*/training.log

# 查看错误日志
grep -i error outputs/pressure_field_*/training.log
```

## 📈 性能优化

### CPU 优化

```bash
# 设置 CPU 线程数
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

# 检查 CPU 信息
lscpu
```

### GPU 优化

```bash
# 设置 GPU 内存增长
export TF_FORCE_GPU_ALLOW_GROWTH=true

# 设置 CUDA 缓存
export CUDA_CACHE_PATH=/tmp/cuda_cache
```

### 网络优化

```bash
# 设置 pip 镜像源
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# 设置 conda 镜像源
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
```

## 🔄 系统服务

### 创建 systemd 服务

```bash
# 创建服务文件
sudo tee /etc/systemd/system/pressure-field-training.service > /dev/null << EOF
[Unit]
Description=Pressure Field Training Service
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=/home/$USER/pressure_field_training
ExecStart=/home/$USER/pressure_field_training/run_training_simple.sh
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# 启用服务
sudo systemctl daemon-reload
sudo systemctl enable pressure-field-training.service

# 管理服务
sudo systemctl start pressure-field-training    # 启动
sudo systemctl stop pressure-field-training     # 停止
sudo systemctl status pressure-field-training   # 状态
sudo systemctl restart pressure-field-training  # 重启
```

## 📝 最佳实践

### 1. 数据管理

```bash
# 创建数据目录结构
mkdir -p data/{raw,processed,results}
mkdir -p outputs/{models,logs,visualizations}

# 定期备份重要数据
tar -czf backup_$(date +%Y%m%d).tar.gz outputs/ data/
```

### 2. 实验管理

```bash
# 使用有意义的实验名称
EXPERIMENT_NAME="pressure_field_lr0.001_batch32_$(date +%Y%m%d)"

# 记录实验参数
echo "实验配置: $EXPERIMENT_NAME" > outputs/$EXPERIMENT_NAME/experiment_info.txt
```

### 3. 资源监控

```bash
# 定期检查资源使用
watch -n 5 'echo "=== CPU ==="; top -bn1 | head -5; echo "=== Memory ==="; free -h; echo "=== GPU ==="; nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits'
```

### 4. 自动化脚本

```bash
# 创建定时任务
crontab -e

# 每小时检查训练状态
0 * * * * /home/user/pressure_field_training/monitor.sh >> /home/user/monitor.log 2>&1

# 每天备份结果
0 2 * * * tar -czf /home/user/backup_$(date +\%Y\%m\%d).tar.gz /home/user/pressure_field_training/outputs/
```

## 📞 技术支持

如果遇到问题，请检查:

1. **系统日志**: `sudo journalctl -f`
2. **训练日志**: `tail -f outputs/*/training.log`
3. **系统资源**: `htop`, `nvidia-smi`
4. **网络连接**: `ping google.com`
5. **磁盘空间**: `df -h`

---

**祝您训练顺利！** 🚀

如有问题，请查看日志文件或联系技术支持。