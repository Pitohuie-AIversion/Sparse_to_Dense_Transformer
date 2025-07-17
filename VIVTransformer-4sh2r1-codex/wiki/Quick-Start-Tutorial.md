# 快速开始教程

本教程将指导您在5分钟内完成VIVTransformer项目的安装、配置和首次运行。

## 📋 目录

- [环境准备](#环境准备)
- [项目安装](#项目安装)
- [基础配置](#基础配置)
- [首次运行](#首次运行)
- [结果查看](#结果查看)
- [常见问题](#常见问题)
- [下一步](#下一步)

## 环境准备

### 🖥️ 系统要求

| 组件 | 最低要求 | 推荐配置 |
|------|----------|----------|
| **操作系统** | Windows 10/Linux/macOS | Windows 11/Ubuntu 20.04+ |
| **Python** | 3.8+ | 3.9+ |
| **内存** | 8GB | 16GB+ |
| **GPU** | 可选 | NVIDIA RTX 3060+ |
| **存储** | 5GB | 10GB+ |

### 🐍 Python环境检查

```bash
# 检查Python版本
python --version
# 应该显示 Python 3.8.x 或更高版本

# 检查pip版本
pip --version
```

### 🎮 GPU环境检查（可选）

```bash
# 检查NVIDIA驱动
nvidia-smi

# 检查CUDA版本
nvcc --version
```

## 项目安装

### 📥 步骤1：克隆项目

```bash
# 克隆仓库
git clone https://github.com/yourusername/VIVTransformer.git

# 进入项目目录
cd VIVTransformer

# 查看项目结构
ls -la
```

### 📦 步骤2：安装依赖

```bash
# 安装Python依赖
pip install -r modify_multi_attention/requirements.txt

# 验证安装
python -c "import torch; print(f'PyTorch版本: {torch.__version__}')"
python -c "import torch; print(f'CUDA可用: {torch.cuda.is_available()}')"
```

**依赖包说明**：
- `torch`: PyTorch深度学习框架
- `numpy`: 数值计算库
- `pyyaml`: YAML配置文件解析
- `matplotlib`: 图表绘制
- `fightingcv_attention`: 注意力机制库
- `tensorboard`: 训练监控
- `black`: 代码格式化
- `flake8`: 代码检查

### ✅ 步骤3：验证安装

```bash
# 运行帮助命令
python -m modify_multi_attention.main --help

# 应该显示帮助信息
```

## 基础配置

### ⚙️ 配置文件说明

主配置文件位于 `modify_multi_attention/configs/config.yaml`：

```yaml
# 全局设置
global:
  seed: 42                    # 随机种子
  deterministic: true         # 确定性训练
  device: cuda:0             # 设备选择
  max_memory_fraction: 0.8   # GPU内存限制

# 数据设置
data:
  path: "path/to/your/data.pt"  # 数据路径
  batch_size: 128               # 批大小
  use_augmentation: true        # 数据增强
  crop_size: [128, 128]        # 裁剪尺寸

# 模型设置
model:
  attention_type: self    # 注意力类型
  d_model: 256           # 模型维度
  num_heads: 4           # 注意力头数
  num_layers: 6          # 层数

# 训练设置
training:
  epochs: 10                  # 训练轮数
  learning_rate: 0.0001      # 学习率
  early_stop_patience: 10    # 早停耐心值
```

### 🔧 快速配置修改

1. **CPU模式**（如果没有GPU）：
   ```yaml
   global:
     device: cpu
   ```

2. **减少内存使用**：
   ```yaml
   data:
     batch_size: 64  # 减小批大小
   model:
     d_model: 128    # 减小模型维度
   ```

3. **快速测试**：
   ```yaml
   training:
     epochs: 2       # 减少训练轮数
   ```

## 首次运行

### 🚀 步骤1：准备数据

如果您没有自己的数据，可以使用项目提供的示例数据：

```bash
# 检查数据路径
ls modify_multi_attention/data/

# 如果没有数据文件，请修改配置文件中的数据路径
```

### 🎯 步骤2：运行单个实验

```bash
# 运行单个损失配置的实验（推荐首次运行）
python -m modify_multi_attention.main --loss_idx 0
```

**输出说明**：
```
加载配置文件: .../config.yaml
Using device: cuda:0
Available GPUs: 1
Found 50 loss configurations
只运行 loss_config_0
Data loaders created successfully.
===== 当前loss设置 [loss_config_0]: base_weight=0.5, svd_weights=[...] =====
Starting experiments...
正在运行实验: self attention...
训练进度: Epoch 1/10, Loss: 0.456
...
实验完成: Test Loss: 0.046695
🎉 所有loss配置和注意力机制均运行成功！
```

### 📊 步骤3：启动监控（可选）

在另一个终端窗口中启动TensorBoard：

```bash
# 启动TensorBoard
python modify_multi_attention/start_tensorboard.py

# 在浏览器中打开 http://localhost:6006
```

### 🔄 步骤4：运行完整实验

```bash
# 运行所有注意力机制和损失配置（需要较长时间）
python -m modify_multi_attention.main

# 或者指定自定义配置
python -m modify_multi_attention.main --config my_config.yaml --results-dir ./my_results
```

## 结果查看

### 📁 结果目录结构

```
attention_results/
├── loss_config_0/              # 损失配置0的结果
│   ├── self/                   # 自注意力结果
│   │   ├── model_best.pth      # 最佳模型
│   │   ├── training_log.txt    # 训练日志
│   │   └── attention_vis.png   # 注意力可视化
│   ├── muse/                   # MUSE注意力结果
│   └── ...
├── train.log                   # 全局训练日志
└── failed_attention_log.txt    # 失败实验记录
```

### 📈 查看训练日志

```bash
# 查看全局日志
tail -f attention_results/train.log

# 查看特定实验日志
cat attention_results/loss_config_0/self/training_log.txt
```

### 🎨 查看可视化结果

1. **TensorBoard**：
   - 打开 http://localhost:6006
   - 查看训练曲线、损失变化等

2. **注意力热图**：
   ```bash
   # 查看注意力可视化图片
   ls attention_results/loss_config_0/*/attention_vis.png
   ```

### 📊 性能分析

```bash
# 查看所有实验的测试损失
grep "Test Loss" attention_results/train.log

# 查看失败的实验
cat attention_results/failed_attention_log.txt
```

## 常见问题

### ❓ Q1: 出现CUDA内存不足错误

**解决方案**：
```yaml
# 在config.yaml中调整以下参数
data:
  batch_size: 32        # 减小批大小
global:
  max_memory_fraction: 0.6  # 减少GPU内存使用
```

### ❓ Q2: 找不到数据文件

**解决方案**：
```bash
# 检查数据路径
ls /path/to/your/data

# 修改配置文件中的数据路径
vim modify_multi_attention/configs/config.yaml
```

### ❓ Q3: 训练速度太慢

**解决方案**：
1. 使用GPU加速
2. 减少模型复杂度
3. 使用更高效的注意力机制

```yaml
model:
  attention_type: muse    # 使用高效注意力
  d_model: 128           # 减小模型维度
training:
  epochs: 5              # 减少训练轮数
```

### ❓ Q4: 导入错误

**解决方案**：
```bash
# 重新安装依赖
pip install -r modify_multi_attention/requirements.txt --force-reinstall

# 检查Python路径
python -c "import sys; print(sys.path)"
```

### ❓ Q5: 配置文件错误

**解决方案**：
```bash
# 验证YAML语法
python -c "import yaml; yaml.safe_load(open('modify_multi_attention/configs/config.yaml'))"

# 使用默认配置
cp modify_multi_attention/configs/config.yaml modify_multi_attention/configs/config_backup.yaml
```

## 下一步

### 🎯 学习路径

1. **深入了解**：
   - 📖 [项目架构概览](Architecture-Overview)
   - 🔍 [注意力机制指南](Attention-Mechanisms-Guide)

2. **高级使用**：
   - ⚙️ [高级配置](Advanced-Configuration)
   - 🧪 [批量实验](Batch-Experiments)

3. **自定义开发**：
   - 🔧 [添加新功能](Adding-Features)
   - 🧠 [自定义注意力机制](Custom-Attention)

4. **结果分析**：
   - 📊 [结果分析](Results-Analysis)
   - 📈 [可视化分析](Visualization-Analysis)

### 🚀 实践建议

1. **从简单开始**：
   - 先运行单个注意力机制
   - 使用小数据集测试
   - 逐步增加复杂度

2. **系统性实验**：
   - 制定实验计划
   - 记录实验结果
   - 对比不同配置

3. **持续学习**：
   - 关注最新研究
   - 尝试新的注意力机制
   - 参与社区讨论

### 📞 获取帮助

如果您在使用过程中遇到问题：

1. 🔍 查看 [常见问题](FAQ)
2. 📖 阅读详细文档
3. 🐛 提交 [Issue](../../issues)
4. 💬 参与讨论

---

**🎉 恭喜！** 您已经成功完成了VIVTransformer项目的快速入门。现在可以开始探索更多高级功能了！

**💡 提示**：建议先运行几个简单的实验来熟悉系统，然后再进行大规模的实验。