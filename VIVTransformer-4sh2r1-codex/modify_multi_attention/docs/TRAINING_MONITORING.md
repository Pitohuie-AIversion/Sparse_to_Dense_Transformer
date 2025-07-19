# 训练监控系统使用指南

本文档介绍了VIVTransformer项目中新增的训练监控系统，包括硬件监控、增强日志记录和训练可视化功能。

## 🎯 功能概述

### 1. 硬件监控 (HardwareMonitor)
- **GPU监控**: GPU使用率、显存占用、温度、功耗
- **CPU监控**: CPU使用率、内存占用
- **训练时间统计**: Epoch时间、批次时间、总训练时间
- **系统信息记录**: 硬件配置、环境信息

### 2. 增强日志记录 (EnhancedTrainingLogger)
- **详细训练日志**: 批次级别的指标记录
- **结构化数据**: JSON和CSV格式输出
- **自动时间戳**: 精确的时间记录
- **多级别日志**: 支持不同详细程度的日志

### 3. 训练可视化 (TrainingVisualizer)
- **损失曲线**: 训练、验证、测试损失可视化
- **硬件使用图表**: GPU/CPU使用率趋势
- **训练仪表板**: 综合性能概览
- **HTML报告**: 交互式训练报告

## 🚀 快速开始

### 1. 配置文件设置

在 `config.yaml` 中添加硬件监控配置：

```yaml
hardware_monitoring:
  enable_gpu_monitoring: true      # 启用GPU监控
  log_batch_metrics: true          # 记录批次级别指标
  batch_log_interval: 10           # 批次日志记录间隔
  save_detailed_metrics: true      # 保存详细指标
  monitor_temperature: true        # 监控温度
  monitor_power_usage: true        # 监控功耗
```

### 2. 基本使用示例

```python
from utils.hardware_monitor import HardwareMonitor
from utils.enhanced_logger import EnhancedTrainingLogger
from utils.training_visualizer import TrainingVisualizer

# 初始化监控器
hardware_monitor = HardwareMonitor(
    log_dir="./logs/hardware",
    enable_gpu_monitoring=True
)

# 开始训练监控
hardware_monitor.start_training()

# 训练循环
for epoch in range(num_epochs):
    hardware_monitor.start_epoch(epoch + 1)
    
    for batch_idx, (data, target) in enumerate(train_loader):
        hardware_monitor.start_batch(batch_idx + 1)
        
        # 训练代码...
        
        hardware_monitor.end_batch()
    
    hardware_monitor.end_epoch()

# 结束训练监控
hardware_monitor.end_training()

# 生成可视化报告
visualizer = TrainingVisualizer("./logs/hardware")
visualizer.generate_complete_report()
```

### 3. 运行演示脚本

```bash
# 运行训练监控演示
python examples/training_with_monitoring.py
```

## 📊 输出文件说明

### 硬件监控输出

```
logs/hardware/
├── hardware_metrics.json      # 详细硬件指标数据
├── epoch_summaries.json       # Epoch级别摘要
├── training_summary.json      # 训练总体摘要
└── system_info.json          # 系统信息
```

### 可视化输出

```
logs/hardware/visualizations/
├── loss_curves.png           # 损失曲线图
├── hardware_usage.png        # 硬件使用率图
├── training_dashboard.png     # 训练仪表板
├── training_report.html       # HTML交互式报告
├── hardware_metrics.csv       # 硬件指标CSV
└── epoch_summaries.csv        # Epoch摘要CSV
```

## 🔧 高级配置

### 1. 自定义监控间隔

```python
hardware_monitor = HardwareMonitor(
    log_dir="./logs",
    monitoring_interval=1.0,  # 监控间隔（秒）
    enable_gpu_monitoring=True,
    enable_detailed_logging=True
)
```

### 2. 选择性监控

```python
# 只监控GPU，不监控CPU
hardware_monitor = HardwareMonitor(
    log_dir="./logs",
    enable_gpu_monitoring=True,
    enable_cpu_monitoring=False
)
```

### 3. 自定义可视化

```python
visualizer = TrainingVisualizer("./logs")

# 只生成损失曲线
visualizer.plot_loss_curves(save_path="custom_loss.png")

# 只生成硬件使用图
visualizer.plot_hardware_usage(save_path="custom_hardware.png")

# 生成自定义仪表板
visualizer.create_training_dashboard(
    save_path="custom_dashboard.png",
    figsize=(20, 12)
)
```

## 📈 性能影响

### 监控开销
- **GPU监控**: ~0.1-0.5% 训练时间开销
- **CPU监控**: ~0.05-0.2% 训练时间开销
- **文件I/O**: ~0.1-0.3% 训练时间开销

### 存储需求
- **硬件指标**: ~1-5MB/小时（取决于监控频率）
- **可视化文件**: ~5-20MB（图片和HTML）
- **日志文件**: ~10-50MB（取决于训练时长）

## 🛠️ 故障排除

### 常见问题

1. **GPU监控失败**
   ```
   错误: NVIDIA-ML-PY not available
   解决: pip install nvidia-ml-py3
   ```

2. **权限错误**
   ```
   错误: Permission denied when creating log directory
   解决: 确保有写入权限或更改日志目录
   ```

3. **内存不足**
   ```
   错误: Out of memory when generating visualizations
   解决: 减少监控频率或增加系统内存
   ```

### 调试模式

```python
# 启用详细日志
import logging
logging.basicConfig(level=logging.DEBUG)

# 使用调试模式
hardware_monitor = HardwareMonitor(
    log_dir="./logs",
    debug=True
)
```

## 🔗 集成到现有训练代码

### 1. 最小集成

```python
# 在训练函数开始处添加
from utils.hardware_monitor import HardwareMonitor

hardware_monitor = HardwareMonitor(log_dir=result_dir / "hardware_logs")
hardware_monitor.start_training()

# 在训练循环中添加
for epoch in range(num_epochs):
    hardware_monitor.start_epoch(epoch + 1)
    # ... 训练代码 ...
    hardware_monitor.end_epoch()

# 在训练结束时添加
hardware_monitor.end_training()
```

### 2. 完整集成

参考 `training/trainer.py` 中的实现，已经完全集成了所有监控功能。

## 📚 API参考

### HardwareMonitor

```python
class HardwareMonitor:
    def __init__(self, log_dir: str, enable_gpu_monitoring: bool = True)
    def start_training(self) -> None
    def end_training(self) -> None
    def start_epoch(self, epoch: int) -> None
    def end_epoch(self) -> None
    def start_batch(self, batch: int) -> None
    def end_batch(self) -> None
    def get_training_summary(self) -> Dict[str, Any]
```

### TrainingVisualizer

```python
class TrainingVisualizer:
    def __init__(self, log_dir: str)
    def plot_loss_curves(self, save_path: str = None) -> None
    def plot_hardware_usage(self, save_path: str = None) -> None
    def create_training_dashboard(self, save_path: str = None) -> None
    def generate_complete_report(self) -> None
```

## 🎨 自定义和扩展

### 添加自定义指标

```python
class CustomHardwareMonitor(HardwareMonitor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.custom_metrics = []
    
    def log_custom_metric(self, name: str, value: float):
        """记录自定义指标"""
        self.custom_metrics.append({
            'timestamp': time.time(),
            'name': name,
            'value': value
        })
```

### 自定义可视化样式

```python
# 修改matplotlib样式
import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8')  # 或其他样式

# 自定义颜色
visualizer = TrainingVisualizer("./logs")
visualizer.plot_loss_curves(
    colors=['#1f77b4', '#ff7f0e', '#2ca02c']  # 自定义颜色
)
```

## 📝 最佳实践

1. **合理设置监控频率**: 避免过于频繁的监控影响性能
2. **定期清理日志**: 避免日志文件过大占用存储空间
3. **使用异步监控**: 对于长时间训练，考虑异步监控
4. **备份重要数据**: 定期备份训练日志和可视化结果
5. **监控系统资源**: 确保监控本身不会成为瓶颈

## 🤝 贡献

欢迎提交问题和改进建议！请参考项目的贡献指南。

## 📄 许可证

本项目采用 MIT 许可证。详见 LICENSE 文件。