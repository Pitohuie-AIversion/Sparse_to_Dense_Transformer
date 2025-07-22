# VIV Transformer 压力场重建训练指南

## 🚀 快速开始

### 1. 一键启动训练管理系统
```bash
# 启动统一管理界面
bash training_manager.sh
```

### 2. 快速训练
```bash
# 快速开始训练（推荐新手）
bash quick_start.sh

# 或使用优化版本（推荐服务器）
bash run_training_optimized.sh
```

### 3. 监控系统
```bash
# GPU监控
bash gpu_monitor.sh

# 系统监控
bash system_monitor.sh
```

## 📋 系统概览

### 服务器配置
- **服务器**: node36
- **GPU**: 2x NVIDIA L40 (48GB显存)
- **驱动**: 535.183.01
- **CUDA**: 12.2

### 脚本功能

| 脚本名称 | 功能描述 | 使用场景 |
|---------|---------|----------|
| `training_manager.sh` | 统一训练管理界面 | 主要入口，集成所有功能 |
| `quick_start.sh` | 快速启动训练 | 新手快速开始 |
| `run_training_optimized.sh` | 优化训练脚本 | 服务器环境，高性能训练 |
| `gpu_monitor.sh` | GPU实时监控 | 监控GPU状态和进程 |
| `system_monitor.sh` | 系统资源监控 | 监控CPU、内存、磁盘 |

## 🎯 训练模式选择

### 1. 快速开始模式
- **适用**: 新手用户，快速验证
- **特点**: 自动配置，一键启动
- **命令**: `bash quick_start.sh`

### 2. 优化训练模式
- **适用**: 服务器环境，长时间训练
- **特点**: 性能优化，资源监控
- **命令**: `bash run_training_optimized.sh`

### 3. 管理界面模式
- **适用**: 需要完整管理功能
- **特点**: 图形化菜单，功能齐全
- **命令**: `bash training_manager.sh`

## 🔧 GPU配置说明

### 自动GPU选择
系统会自动检测GPU状态并推荐最佳选择：

```bash
# GPU状态示例
GPU 0: 18276MB/49140MB (37%), 0%, 45°C - 轻载
GPU 1: 2MB/49140MB (0%), 0%, 42°C - 空闲

推荐使用: GPU 1 (空闲状态)
```

### 手动GPU设置
```bash
# 使用GPU 0
export CUDA_VISIBLE_DEVICES=0

# 使用GPU 1
export CUDA_VISIBLE_DEVICES=1

# 使用双GPU
export CUDA_VISIBLE_DEVICES=0,1
```

## 📊 监控功能

### GPU监控
```bash
# 实时监控
bash gpu_monitor.sh

# 简单状态
bash gpu_monitor.sh -s

# 显示推荐
bash gpu_monitor.sh -r
```

**监控内容**:
- GPU使用率和显存占用
- 温度和功耗
- 运行进程详情
- 使用推荐

### 系统监控
```bash
# 实时监控
bash system_monitor.sh

# 简单状态
bash system_monitor.sh -s

# 性能报告
bash system_monitor.sh -r
```

**监控内容**:
- CPU和内存使用率
- 磁盘空间和IO
- 网络状态
- 训练进程状态

## 🗂️ 文件结构

```
pdebench_extended/
├── configs/
│   └── pressure_field_training.yaml    # 训练配置文件
├── data/
│   └── pressure_field_data.pt          # 训练数据
├── outputs/                            # 训练输出目录
│   ├── checkpoints/                    # 模型检查点
│   ├── logs/                          # 训练日志
│   └── visualizations/                # 可视化结果
├── training_manager.sh                 # 主管理脚本
├── quick_start.sh                      # 快速启动
├── run_training_optimized.sh          # 优化训练
├── gpu_monitor.sh                      # GPU监控
├── system_monitor.sh                   # 系统监控
└── train_pressure_field.py            # 训练主程序
```

## ⚙️ 配置参数

### 主要配置文件: `configs/pressure_field_training.yaml`

```yaml
# 关键参数
data:
  batch_size: 32              # 批处理大小
  input_dim: [20, 20]         # 输入维度
  output_dim: [200, 200]      # 输出维度

model:
  d_model: 512               # 模型维度
  nhead: 8                   # 注意力头数
  num_layers: 6              # 层数

training:
  epochs: 100                # 训练轮数
  learning_rate: 0.001       # 学习率
  device: "auto"             # 设备选择
```

### 性能优化参数

```bash
# 环境变量
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=1

# 训练参数
--batch_size 32
--num_workers 4
--pin_memory
--mixed_precision
```

## 🚨 故障排除

### 常见问题

#### 1. GPU内存不足
```bash
# 错误信息
RuntimeError: CUDA out of memory

# 解决方案
# 1. 减小batch_size
# 2. 使用梯度累积
# 3. 启用混合精度训练
```

#### 2. 数据文件未找到
```bash
# 错误信息
FileNotFoundError: data/pressure_field_data.pt

# 解决方案
# 检查数据文件路径
ls -la data/

# 重新生成数据（如果需要）
python generate_data.py
```

#### 3. 训练进程卡死
```bash
# 查看进程
ps aux | grep python

# 终止进程
kill <PID>

# 或使用管理界面
bash training_manager.sh
# 选择 "管理训练任务" -> "停止训练进程"
```

#### 4. TensorBoard无法访问
```bash
# 检查TensorBoard进程
ps aux | grep tensorboard

# 重新启动TensorBoard
tensorboard --logdir=outputs --port=6006 --host=0.0.0.0

# 访问地址
http://<服务器IP>:6006
```

### 性能优化建议

#### 1. GPU选择策略
- **单GPU训练**: 选择显存使用最少的GPU
- **双GPU训练**: 使用DataParallel或DistributedDataParallel
- **混合精度**: 启用AMP减少显存占用

#### 2. 数据加载优化
```python
# 优化参数
num_workers = 4          # 数据加载进程数
pin_memory = True        # 固定内存
prefetch_factor = 2      # 预取因子
```

#### 3. 训练策略
- **学习率调度**: 使用CosineAnnealingLR
- **早停机制**: 防止过拟合
- **梯度裁剪**: 防止梯度爆炸

## 📈 监控和日志

### TensorBoard可视化
```bash
# 启动TensorBoard
tensorboard --logdir=outputs --port=6006 --host=0.0.0.0

# 访问地址
http://<服务器IP>:6006
```

### 日志文件
- **训练日志**: `outputs/logs/training.log`
- **系统日志**: `system_monitor.log`
- **错误日志**: `nohup.out`

### 检查点管理
```bash
# 检查点目录
ls -la outputs/checkpoints/

# 恢复训练
python train_pressure_field.py --resume outputs/checkpoints/best_model.pth
```

## 🔄 工作流程

### 标准训练流程
1. **环境检查**: 验证GPU、Python、PyTorch
2. **数据准备**: 确认数据文件存在
3. **配置设置**: 调整训练参数
4. **启动训练**: 选择合适的训练模式
5. **监控训练**: 使用监控脚本观察状态
6. **结果分析**: 查看TensorBoard和日志

### 最佳实践
1. **训练前**: 使用`training_manager.sh`检查系统状态
2. **训练中**: 定期查看GPU和系统监控
3. **训练后**: 备份重要的检查点和日志

## 📞 技术支持

### 快速诊断
```bash
# 系统状态检查
bash training_manager.sh
# 选择 "系统配置" 查看详细信息

# 生成系统报告
bash training_manager.sh
# 选择 "工具箱" -> "生成系统报告"
```

### 联系方式
如遇到问题，请提供：
1. 错误信息截图
2. 系统报告文件
3. 训练配置文件
4. 相关日志文件

---

**祝您训练顺利！** 🎉