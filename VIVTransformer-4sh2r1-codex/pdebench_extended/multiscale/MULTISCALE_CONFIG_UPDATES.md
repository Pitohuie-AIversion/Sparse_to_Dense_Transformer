# 多尺度配置文件更新说明

## 更新概述

基于对 `pdebench_extended` 项目整体架构的深入分析，对多尺度模块的配置文件进行了全面优化和重构，确保与主项目的设计理念和技术架构保持完全一致。

## 主要更新内容

### 1. 全局配置统一

**更新前：**
```yaml
device: "auto"
seed: 42
```

**更新后：**
```yaml
global:
  seed: 42
  deterministic: true
  device: "cuda:0"              # 强烈建议明确指定主卡
  max_memory_fraction: 0.8      # 每进程最大显存占用比例
  use_dataparallel: true        # 双卡/多卡训练控制
```

**改进说明：**
- 采用与主项目一致的全局配置结构
- 明确指定GPU设备，避免自动选择的不确定性
- 添加内存管理和多GPU训练控制

### 2. 模型架构配置优化

**更新前：**
```yaml
model:
  input_dim: 4096
  output_dim: 16384
  d_model: 512
  nhead: 8
  num_encoder_layers: 6
  num_decoder_layers: 6
```

**更新后：**
```yaml
model:
  attention_type: "relative"  # 默认使用相对位置注意力
  input_dim: 4096
  output_dim: 16384
  d_model: 512
  num_heads: 8      # 与主项目保持一致
  num_layers: 6     # 与主项目保持一致
  max_time_steps: 100
  seq_len: 49       # 明确指定7x7空间尺寸，支持CNN注意力
```

**改进说明：**
- 统一参数命名规范（`num_heads` 替代 `nhead`）
- 添加注意力机制类型配置
- 集成时间步长和序列长度配置
- 移除编码器/解码器分离，采用统一的层数配置

### 3. 损失函数集成SVD架构

**更新前：**
```yaml
loss:
  primary:
    type: "MSELoss"
    weight: 1.0
  additional:
    - type: "L1Loss"
      weight: 0.1
```

**更新后：**
```yaml
loss:
  type: "TotalLossWithSVD"
  svd_config:
    base_weight: 0.7
    svd_weights: [0.1, 0.08, 0.06, 0.04, 0.02]
    topk: 5
  additional:
    - type: "L1Loss"
      weight: 0.05
    - type: "SSIMLoss"
      weight: 0.03
```

**改进说明：**
- 集成项目核心的SVD损失函数架构
- 针对多尺度任务优化SVD权重配置
- 为不同缩放因子提供专门的SVD配置

### 4. 注意力机制配置扩展

**更新前：**
```yaml
attention_mechanisms:
  - "standard"
  - "linear"
  - "performer"
  - "linformer"
```

**更新后：**
```yaml
attention_types:
  - "relative"      # 相对位置注意力（默认）
  - "sparse"        # 稀疏注意力
  - "lsh"           # LSH注意力
  - "se"            # SE注意力
  - "cbam"          # CBAM注意力
  - "sge"           # SGE注意力
  - "muse"          # MUSE注意力
  - "mobilevit"     # MobileViT注意力
  - "dat"           # DAT注意力
  - "crossformer"   # CrossFormer注意力
  - "moa"           # MOA注意力

attention_test:
  types: 
    - self
    - simplified_self
    - muse
    - ufo
    - relative
    # ... 更多注意力类型
```

**改进说明：**
- 采用与主项目完全一致的注意力机制列表
- 添加注意力测试配置，支持实验对比
- 扩展支持的注意力类型，包括最新的研究成果

### 5. 硬件监控和可视化集成

**新增配置：**
```yaml
hardware_monitoring:
  enable_gpu_monitoring: true
  log_batch_metrics: true
  batch_log_interval: 50
  save_detailed_metrics: true
  monitor_temperature: true
  monitor_power_usage: true

visualization:
  enabled: true
  interval: 100
  max_samples: 5
  plot_types:
    - "input_output_comparison"
    - "error_maps"
    - "attention_maps"
    - "loss_curves"
    - "metric_trends"
```

**改进说明：**
- 集成主项目的硬件监控功能
- 扩展可视化类型，支持多种分析图表
- 添加性能监控和资源使用跟踪

### 6. PDEBench数据集集成

**新增配置：**
```yaml
pdebench:
  data_root: "./data/pdebench"
  pde_configs:
    darcy_flow:
      data_file: "darcy_flow_beta_0.01.h5"
      sequence_length: 1
      spatial_resolution: [128, 128]
      normalize: true
    ns_incom:
      data_file: "ns_incom_inhom_2d.h5"
      sequence_length: 49
      spatial_resolution: [128, 128]
      normalize: true
    ns_compressible:
      data_file: "ns_compressible_2d.h5"
      sequence_length: 49
      spatial_resolution: [128, 128]
      normalize: true
```

**改进说明：**
- 支持多种PDE类型的数据集
- 统一数据集配置管理
- 与主项目的PDEBench适配器完全兼容

### 7. 多尺度实验配置优化

**更新后：**
```yaml
multiscale_experiments:
  scale_factors: [2, 4, 8]
  scale_configs:
    2:  # 2倍缩放
      input_dim: 4096
      output_dim: 16384
      batch_size: 16
      learning_rate: 0.0001
      svd_config:
        base_weight: 0.75
        svd_weights: [0.08, 0.06, 0.05, 0.04, 0.02]
        topk: 5
    4:  # 4倍缩放
      # ... 针对4倍缩放的优化配置
    8:  # 8倍缩放
      # ... 针对8倍缩放的优化配置
```

**改进说明：**
- 为每个缩放因子提供专门的SVD损失配置
- 根据缩放难度调整损失权重分布
- 优化批次大小和学习率设置

### 8. 高级功能配置

**新增配置：**
```yaml
advanced:
  gradient_clipping:
    enabled: true
    max_norm: 1.0
    norm_type: 2
  
  weight_initialization:
    method: "xavier_uniform"
    gain: 1.0
  
  regularization:
    l1_weight: 0.0
    l2_weight: 0.0001
    dropout_schedule: false
  
  model_saving:
    save_every_n_epochs: 10
    keep_last_n_checkpoints: 5
    save_optimizer_state: true
    compress_checkpoints: false

experiment_management:
  version_control:
    enabled: true
    save_code_snapshot: true
    track_dependencies: true
  
  comparison:
    enabled: true
    metrics_to_compare: ["mse", "mae", "psnr", "ssim"]
    save_comparison_plots: true
```

**改进说明：**
- 添加梯度裁剪和权重初始化配置
- 集成实验管理和版本控制功能
- 支持实验对比和结果分析

## 兼容性保证

### 向后兼容
- 保留所有原有的核心配置项
- 新增配置项都有合理的默认值
- 支持渐进式迁移

### 版本信息
```yaml
compatibility:
  config_version: "2.0"
  min_pytorch_version: "1.12.0"
  legacy_mode: false
```

## 使用建议

### 1. 快速开始
使用默认配置即可开始多尺度实验：
```bash
python multiscale/quick_start.py --scale_factor 2
```

### 2. 自定义实验
根据具体需求修改配置文件中的相应部分：
- 调整SVD损失权重以优化特定缩放因子
- 选择合适的注意力机制类型
- 配置硬件监控和可视化选项

### 3. 性能优化
- 根据GPU内存调整 `max_memory_fraction`
- 使用 `gradient_checkpointing` 节省内存
- 启用 `mixed_precision` 加速训练

## 总结

本次配置文件更新实现了多尺度模块与主项目架构的深度集成，主要改进包括：

1. **架构统一**：采用与主项目一致的配置结构和命名规范
2. **功能增强**：集成SVD损失函数、多种注意力机制、硬件监控等核心功能
3. **实验支持**：提供完整的实验管理和结果分析功能
4. **性能优化**：针对多尺度任务特点优化各项参数配置
5. **扩展性强**：支持多种PDE类型和缩放因子的灵活配置

这些改进确保了多尺度模块能够充分利用主项目的技术优势，同时为超分辨率重构任务提供了专门的优化配置。