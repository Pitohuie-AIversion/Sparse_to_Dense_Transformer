---
layout: doc
title: Configuration System
description: 配置文件系统的使用和管理
permalink: /pages/configuration-system/
---

# 配置系统 {#配置系统}

本文档详细介绍VIVTransformer项目的配置系统设计，包括配置文件结构、参数管理和动态配置功能。

## 📋 目录 {#目录}

- [配置系统概览](#配置系统概览)
- [主配置文件](#主配置文件)
- [损失配置管理](#损失配置管理)
- [动态配置](#动态配置)
- [配置验证](#配置验证)
- [环境配置](#环境配置)
- [最佳实践](#最佳实践)

## 配置系统概览 {#配置系统概览}

### 🏗️ 配置架构 {#配置架构}

```
配置系统
├── 📁 configs/
│   ├── config.yaml              # 主配置文件
│   ├── 📁 loss_configs/         # 损失函数配置
│   │   ├── loss_config_0.yaml
│   │   ├── loss_config_1.yaml
│   │   └── ...
│   ├── 📁 model_configs/        # 模型配置
│   ├── 📁 data_configs/         # 数据配置
│   └── 📁 experiment_configs/   # 实验配置
├── utils/config.py              # 配置管理工具
└── generate_loss_configs.py     # 配置生成脚本
```

### 🎯 设计原则 {#设计原则}

1. **层次化结构**: 配置按功能模块分层组织
2. **可扩展性**: 支持新配置项的动态添加
3. **类型安全**: 严格的类型检查和验证
4. **环境适配**: 支持不同环境的配置覆盖
5. **版本控制**: 配置变更的版本管理

## 主配置文件 {#主配置文件}

### 📝 config.yaml 结构 {#config-yaml-结构}

```yaml
# ==================== 全局配置 ==================== {#全局配置}
global:
  # 基础设置
  seed: 42                        # 随机种子
  deterministic: true             # 确定性训练
  device: "cuda:0"                # 计算设备
  mixed_precision: false          # 混合精度训练
  
  # 内存管理
  max_memory_fraction: 0.8        # GPU内存使用限制
  memory_growth: true             # 动态内存增长
  
  # 调试设置
  debug: false                    # 调试模式
  verbose: true                   # 详细输出
  log_level: "INFO"               # 日志级别

# ==================== 数据配置 ==================== {#数据配置}
data:
  # 数据路径
  path: "data/viv_dataset.pt"     # 数据文件路径
  cache_dir: "cache/"             # 缓存目录
  
  # 数据加载
  batch_size: 128                 # 批大小
  num_workers: 4                  # 数据加载进程数
  pin_memory: true                # 内存锁定
  prefetch_factor: 2              # 预取因子
  
  # 数据预处理
  normalize: true                 # 数据标准化
  crop_size: [400]                # 裁剪尺寸
  use_augmentation: true          # 数据增强
  
  # 数据分割
  train_ratio: 0.7                # 训练集比例
  valid_ratio: 0.15               # 验证集比例
  test_ratio: 0.15                # 测试集比例

# ==================== 模型配置 ==================== {#模型配置}
model:
  # 基础架构
  attention_type: "self"          # 注意力机制类型
  d_model: 256                    # 模型维度
  num_heads: 4                    # 注意力头数
  num_layers: 6                   # 层数
  
  # 输入输出
  input_dim: 400                  # 输入维度
  output_dim: 40000               # 输出维度
  seq_len: 49                     # 序列长度
  
  # 正则化
  dropout: 0.1                    # Dropout率
  layer_norm_eps: 1e-6            # 层归一化epsilon
  
  # 初始化
  init_method: "xavier_uniform"   # 权重初始化方法
  bias_init: 0.0                  # 偏置初始化值

# ==================== 训练配置 ==================== {#训练配置}
training:
  # 基础设置
  epochs: 10                      # 训练轮数
  learning_rate: 0.0001           # 学习率
  weight_decay: 0.01              # 权重衰减
  
  # 优化器
  optimizer: "adamw"              # 优化器类型
  beta1: 0.9                      # Adam beta1
  beta2: 0.999                    # Adam beta2
  eps: 1e-8                       # Adam epsilon
  
  # 学习率调度
  lr_scheduler: "cosine"          # 学习率调度器
  warmup_epochs: 2                # 预热轮数
  min_lr: 1e-6                    # 最小学习率
  
  # 早停
  early_stop: true                # 启用早停
  early_stop_patience: 10         # 早停耐心值
  early_stop_delta: 1e-4          # 早停阈值
  
  # 梯度处理
  gradient_clip_norm: 1.0         # 梯度裁剪
  gradient_accumulation_steps: 1  # 梯度累积步数

# ==================== 损失配置 ==================== {#损失配置}
loss:
  # 基础损失
  base_loss: "mse"                # 基础损失函数
  base_weight: 0.5                # 基础损失权重
  
  # SVD正则化
  use_svd_regularization: true    # 启用SVD正则化
  svd_weights: [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
  
  # 其他损失
  auxiliary_losses: []            # 辅助损失函数

# ==================== 评估配置 ==================== {#评估配置}
evaluation:
  # 评估频率
  eval_frequency: 1               # 评估频率（每N个epoch）
  save_frequency: 5               # 模型保存频率
  
  # 评估指标
  metrics: ["mse", "mae", "r2"]   # 评估指标
  
  # 可视化
  visualize_attention: true       # 可视化注意力
  save_predictions: false         # 保存预测结果

# ==================== 日志配置 ==================== {#日志配置}
logging:
  # 基础设置
  log_dir: "logs/"                # 日志目录
  experiment_name: "vivtransformer" # 实验名称
  
  # TensorBoard
  use_tensorboard: true           # 启用TensorBoard
  tensorboard_port: 6006          # TensorBoard端口
  
  # 日志内容
  log_model_summary: true         # 记录模型摘要
  log_gradients: false            # 记录梯度
  log_weights: false              # 记录权重
  
  # 保存设置
  save_config: true               # 保存配置文件
  save_code: true                 # 保存代码快照

# ==================== 注意力测试配置 ==================== {#注意力测试配置}
attention_test:
  # 测试设置
  enabled: true                   # 启用注意力测试
  types: ["self", "muse", "ufo", "eca", "se"]  # 测试的注意力类型
  
  # 并行设置
  parallel: false                 # 并行测试
  max_workers: 4                  # 最大工作进程数
  
  # 结果保存
  save_results: true              # 保存测试结果
  results_dir: "attention_results/" # 结果保存目录
```

### 🔧 配置加载器 {#配置加载器}

```python
import yaml
import os
from typing import Dict, Any, Optional
from pathlib import Path

class ConfigLoader:
    """配置加载器"""
    
    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        self.config = None
        
    def load(self) -> Dict[str, Any]:
        """加载配置文件"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {self.config_path}")
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # 应用环境变量覆盖
        self._apply_env_overrides()
        
        # 验证配置
        self._validate_config()
        
        return self.config
    
    def _apply_env_overrides(self):
        """应用环境变量覆盖"""
        env_prefix = "VIV_"
        
        for key, value in os.environ.items():
            if key.startswith(env_prefix):
                config_key = key[len(env_prefix):].lower().replace('_', '.')
                self._set_nested_value(self.config, config_key, value)
    
    def _set_nested_value(self, config: dict, key_path: str, value: str):
        """设置嵌套配置值"""
        keys = key_path.split('.')
        current = config
        
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        # 类型转换
        final_key = keys[-1]
        if final_key in current:
            original_type = type(current[final_key])
            if original_type == bool:
                current[final_key] = value.lower() in ('true', '1', 'yes')
            elif original_type == int:
                current[final_key] = int(value)
            elif original_type == float:
                current[final_key] = float(value)
            else:
                current[final_key] = value
```

## 损失配置管理 {#损失配置管理}

### 📊 损失配置生成 {#损失配置生成}

```python
import numpy as np
import yaml
from pathlib import Path

class LossConfigGenerator:
    """损失配置生成器"""
    
    def __init__(self, output_dir: str = "configs/loss_configs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_configs(self, num_configs: int = 50):
        """生成多个损失配置"""
        configs = []
        
        for i in range(num_configs):
            config = self._generate_single_config(i)
            configs.append(config)
            
            # 保存单个配置文件
            config_path = self.output_dir / f"loss_config_{i}.yaml"
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        
        print(f"✅ 生成了 {num_configs} 个损失配置文件")
        return configs
    
    def _generate_single_config(self, config_id: int) -> Dict[str, Any]:
        """生成单个损失配置"""
        
        # 基础权重范围
        base_weight = np.random.uniform(0.1, 1.0)
        
        # SVD权重生成策略
        svd_strategies = [
            self._uniform_weights,
            self._decreasing_weights,
            self._increasing_weights,
            self._random_weights,
            self._sparse_weights
        ]
        
        strategy = svd_strategies[config_id % len(svd_strategies)]
        svd_weights = strategy()
        
        config = {
            'config_id': config_id,
            'base_weight': round(base_weight, 3),
            'svd_weights': [round(w, 4) for w in svd_weights],
            'description': f"Loss configuration {config_id}",
            'strategy': strategy.__name__
        }
        
        return config
    
    def _uniform_weights(self) -> List[float]:
        """均匀权重策略"""
        weight = np.random.uniform(0.01, 0.2)
        return [weight] * 10
    
    def _decreasing_weights(self) -> List[float]:
        """递减权重策略"""
        start_weight = np.random.uniform(0.1, 0.3)
        decay_factor = np.random.uniform(0.7, 0.9)
        weights = [start_weight * (decay_factor ** i) for i in range(10)]
        return weights
    
    def _increasing_weights(self) -> List[float]:
        """递增权重策略"""
        start_weight = np.random.uniform(0.01, 0.05)
        growth_factor = np.random.uniform(1.1, 1.3)
        weights = [start_weight * (growth_factor ** i) for i in range(10)]
        return weights
    
    def _random_weights(self) -> List[float]:
        """随机权重策略"""
        return np.random.uniform(0.01, 0.2, 10).tolist()
    
    def _sparse_weights(self) -> List[float]:
        """稀疏权重策略"""
        weights = np.zeros(10)
        num_nonzero = np.random.randint(2, 6)
        indices = np.random.choice(10, num_nonzero, replace=False)
        weights[indices] = np.random.uniform(0.05, 0.3, num_nonzero)
        return weights.tolist()
```

### 📋 损失配置模板 {#损失配置模板}

```yaml
# loss_config_template.yaml {#loss-config-template-yaml}
config_id: 0
description: "基础损失配置模板"

# 基础损失设置 {#基础损失设置}
base_weight: 0.5
base_loss_type: "mse"

# SVD正则化设置 {#svd正则化设置}
use_svd_regularization: true
svd_weights: [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
svd_components: 10

# 高级设置 {#高级设置}
weight_decay: 0.01
label_smoothing: 0.0
focal_loss_gamma: 2.0

# 动态权重 {#动态权重}
dynamic_weighting: false
weight_schedule: "constant"  # constant, linear, cosine

# 损失组合 {#损失组合}
loss_combination: "weighted_sum"  # weighted_sum, adaptive, uncertainty

# 元数据 {#元数据}
created_by: "auto_generator"
creation_time: "2024-01-01T00:00:00"
version: "1.0"
```

## 动态配置 {#动态配置}

### 🔄 运行时配置修改 {#运行时配置修改}

```python
class DynamicConfig:
    """动态配置管理器"""
    
    def __init__(self, base_config: Dict[str, Any]):
        self.base_config = base_config.copy()
        self.current_config = base_config.copy()
        self.config_history = [base_config.copy()]
    
    def update(self, updates: Dict[str, Any]):
        """更新配置"""
        self._deep_update(self.current_config, updates)
        self.config_history.append(self.current_config.copy())
    
    def rollback(self, steps: int = 1):
        """回滚配置"""
        if len(self.config_history) > steps:
            self.current_config = self.config_history[-(steps + 1)].copy()
            self.config_history = self.config_history[:-(steps)]
    
    def reset(self):
        """重置到基础配置"""
        self.current_config = self.base_config.copy()
        self.config_history = [self.base_config.copy()]
    
    def _deep_update(self, base_dict: dict, update_dict: dict):
        """深度更新字典"""
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value

# 使用示例 {#使用示例}
config_manager = DynamicConfig(base_config)

# 动态调整学习率 {#动态调整学习率}
config_manager.update({
    'training': {
        'learning_rate': 0.0005
    }
})

# 动态调整批大小 {#动态调整批大小}
config_manager.update({
    'data': {
        'batch_size': 64
    }
})
```

### 🎛️ 自适应配置 {#自适应配置}

```python
class AdaptiveConfig:
    """自适应配置管理器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.performance_history = []
        self.adaptation_rules = self._setup_adaptation_rules()
    
    def adapt_based_on_performance(self, metrics: Dict[str, float]):
        """基于性能指标自适应调整配置"""
        self.performance_history.append(metrics)
        
        # 应用适应规则
        for rule in self.adaptation_rules:
            if rule.should_apply(metrics, self.performance_history):
                updates = rule.get_updates(metrics, self.config)
                self._apply_updates(updates)
                print(f"🔄 应用适应规则: {rule.name}")
    
    def _setup_adaptation_rules(self):
        """设置适应规则"""
        return [
            LearningRateAdaptationRule(),
            BatchSizeAdaptationRule(),
            DropoutAdaptationRule(),
            EarlyStopAdaptationRule()
        ]

class AdaptationRule:
    """适应规则基类"""
    
    def __init__(self, name: str):
        self.name = name
    
    def should_apply(self, current_metrics: Dict[str, float], 
                    history: List[Dict[str, float]]) -> bool:
        raise NotImplementedError
    
    def get_updates(self, metrics: Dict[str, float], 
                   config: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError

class LearningRateAdaptationRule(AdaptationRule):
    """学习率自适应规则"""
    
    def __init__(self):
        super().__init__("学习率自适应")
        self.patience = 3
        self.factor = 0.5
    
    def should_apply(self, current_metrics, history):
        if len(history) < self.patience + 1:
            return False
        
        # 检查最近几次的损失是否没有改善
        recent_losses = [m.get('loss', float('inf')) for m in history[-self.patience:]]
        return all(recent_losses[i] >= recent_losses[i-1] for i in range(1, len(recent_losses)))
    
    def get_updates(self, metrics, config):
        current_lr = config['training']['learning_rate']
        new_lr = current_lr * self.factor
        return {
            'training': {
                'learning_rate': max(new_lr, 1e-6)
            }
        }
```

## 配置验证 {#配置验证}

### ✅ 配置验证器 {#配置验证器}

```python
from typing import List, Tuple
import jsonschema

class ConfigValidator:
    """配置验证器"""
    
    def __init__(self):
        self.schema = self._load_schema()
        self.custom_validators = self._setup_custom_validators()
    
    def validate(self, config: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """验证配置"""
        errors = []
        
        # JSON Schema验证
        try:
            jsonschema.validate(config, self.schema)
        except jsonschema.ValidationError as e:
            errors.append(f"Schema验证错误: {e.message}")
        
        # 自定义验证
        for validator in self.custom_validators:
            validator_errors = validator.validate(config)
            errors.extend(validator_errors)
        
        return len(errors) == 0, errors
    
    def _load_schema(self) -> Dict[str, Any]:
        """加载配置Schema"""
        return {
            "type": "object",
            "properties": {
                "global": {
                    "type": "object",
                    "properties": {
                        "seed": {"type": "integer", "minimum": 0},
                        "device": {"type": "string"},
                        "deterministic": {"type": "boolean"}
                    },
                    "required": ["seed", "device"]
                },
                "data": {
                    "type": "object",
                    "properties": {
                        "batch_size": {"type": "integer", "minimum": 1},
                        "num_workers": {"type": "integer", "minimum": 0}
                    },
                    "required": ["batch_size"]
                },
                "model": {
                    "type": "object",
                    "properties": {
                        "d_model": {"type": "integer", "minimum": 1},
                        "num_heads": {"type": "integer", "minimum": 1},
                        "num_layers": {"type": "integer", "minimum": 1}
                    },
                    "required": ["d_model", "num_heads"]
                },
                "training": {
                    "type": "object",
                    "properties": {
                        "epochs": {"type": "integer", "minimum": 1},
                        "learning_rate": {"type": "number", "minimum": 0}
                    },
                    "required": ["epochs", "learning_rate"]
                }
            },
            "required": ["global", "data", "model", "training"]
        }
    
    def _setup_custom_validators(self) -> List['CustomValidator']:
        """设置自定义验证器"""
        return [
            DeviceValidator(),
            AttentionTypeValidator(),
            DimensionValidator(),
            PathValidator()
        ]

class CustomValidator:
    """自定义验证器基类"""
    
    def validate(self, config: Dict[str, Any]) -> List[str]:
        raise NotImplementedError

class DeviceValidator(CustomValidator):
    """设备验证器"""
    
    def validate(self, config: Dict[str, Any]) -> List[str]:
        errors = []
        device = config.get('global', {}).get('device', '')
        
        if device.startswith('cuda:'):
            try:
                gpu_id = int(device.split(':')[1])
                if not torch.cuda.is_available():
                    errors.append("CUDA不可用，但配置了CUDA设备")
                elif gpu_id >= torch.cuda.device_count():
                    errors.append(f"GPU {gpu_id} 不存在")
            except (ValueError, IndexError):
                errors.append(f"无效的设备格式: {device}")
        elif device != 'cpu':
            errors.append(f"不支持的设备类型: {device}")
        
        return errors

class AttentionTypeValidator(CustomValidator):
    """注意力类型验证器"""
    
    def validate(self, config: Dict[str, Any]) -> List[str]:
        errors = []
        attention_type = config.get('model', {}).get('attention_type', '')
        
        valid_types = [
            'self', 'simplified_self', 'muse', 'ufo', 'sparse', 'lsh',
            'relative', 'axial', 'mobilevit', 'mobilevitv2', 'emsa',
            'dat', 'crossformer', 'moa', 'crisscross', 'se', 'sk',
            'cbam', 'bam', 'eca', 'psa', 'danet', 'cot', 'polarized'
        ]
        
        if attention_type not in valid_types:
            errors.append(f"不支持的注意力类型: {attention_type}")
        
        return errors
```

## 环境配置 {#环境配置}

### 🌍 环境变量支持 {#环境变量支持}

```bash
# 环境变量配置示例 {#环境变量配置示例}
export VIV_GLOBAL_DEVICE="cuda:1"
export VIV_DATA_BATCH_SIZE="64"
export VIV_TRAINING_LEARNING_RATE="0.0005"
export VIV_GLOBAL_DEBUG="true"
```

### 🔧 环境特定配置 {#环境特定配置}

```python
class EnvironmentConfig:
    """环境特定配置管理"""
    
    def __init__(self, base_config_path: str):
        self.base_config_path = base_config_path
        self.environment = self._detect_environment()
    
    def load_config(self) -> Dict[str, Any]:
        """加载环境特定配置"""
        # 加载基础配置
        base_config = self._load_base_config()
        
        # 加载环境特定覆盖
        env_overrides = self._load_environment_overrides()
        
        # 合并配置
        final_config = self._merge_configs(base_config, env_overrides)
        
        return final_config
    
    def _detect_environment(self) -> str:
        """检测运行环境"""
        if os.getenv('VIV_ENV'):
            return os.getenv('VIV_ENV')
        elif os.getenv('CUDA_VISIBLE_DEVICES'):
            return 'gpu'
        else:
            return 'cpu'
    
    def _load_environment_overrides(self) -> Dict[str, Any]:
        """加载环境特定覆盖配置"""
        env_configs = {
            'development': {
                'global': {'debug': True, 'verbose': True},
                'training': {'epochs': 2},
                'data': {'batch_size': 32}
            },
            'production': {
                'global': {'debug': False, 'verbose': False},
                'training': {'epochs': 50},
                'logging': {'log_level': 'WARNING'}
            },
            'testing': {
                'global': {'deterministic': True},
                'training': {'epochs': 1},
                'data': {'batch_size': 16}
            }
        }
        
        return env_configs.get(self.environment, {})
```

## 最佳实践 {#最佳实践}

### 📚 配置管理最佳实践 {#配置管理最佳实践}

1. **版本控制**
   ```yaml
   # 在配置文件中包含版本信息
   version: "1.2.0"
   config_format_version: "2.0"
   last_modified: "2024-01-01T12:00:00Z"
   ```

2. **文档化**
   ```yaml
   # 为每个配置项添加注释
   training:
     learning_rate: 0.0001  # 初始学习率，建议范围: [1e-5, 1e-2]
     epochs: 10             # 训练轮数，根据数据集大小调整
   ```

3. **默认值**
   ```python
   def get_config_value(config, key_path, default=None):
       """安全获取配置值，支持默认值"""
       keys = key_path.split('.')
       current = config
       
       for key in keys:
           if isinstance(current, dict) and key in current:
               current = current[key]
           else:
               return default
       
       return current
   ```

4. **配置模板**
   ```python
   class ConfigTemplate:
       """配置模板生成器"""
       
       @staticmethod
       def generate_quick_start():
           """生成快速开始配置"""
           return {
               'global': {'device': 'cpu', 'seed': 42},
               'data': {'batch_size': 32},
               'model': {'attention_type': 'self', 'd_model': 128},
               'training': {'epochs': 5, 'learning_rate': 0.001}
           }
       
       @staticmethod
       def generate_high_performance():
           """生成高性能配置"""
           return {
               'global': {'device': 'cuda:0', 'mixed_precision': True},
               'data': {'batch_size': 256, 'num_workers': 8},
               'model': {'attention_type': 'muse', 'd_model': 512},
               'training': {'epochs': 50, 'learning_rate': 0.0001}
           }
   ```

### 🔍 配置调试 {#配置调试}

```python
class ConfigDebugger:
    """配置调试工具"""
    
    @staticmethod
    def print_config_summary(config: Dict[str, Any]):
        """打印配置摘要"""
        print("📋 配置摘要:")
        print(f"   设备: {config.get('global', {}).get('device', 'unknown')}")
        print(f"   批大小: {config.get('data', {}).get('batch_size', 'unknown')}")
        print(f"   注意力类型: {config.get('model', {}).get('attention_type', 'unknown')}")
        print(f"   训练轮数: {config.get('training', {}).get('epochs', 'unknown')}")
    
    @staticmethod
    def validate_config_compatibility(config: Dict[str, Any]):
        """验证配置兼容性"""
        warnings = []
        
        # 检查设备和批大小兼容性
        device = config.get('global', {}).get('device', 'cpu')
        batch_size = config.get('data', {}).get('batch_size', 32)
        
        if device == 'cpu' and batch_size > 128:
            warnings.append("CPU模式下建议使用较小的批大小")
        
        # 检查注意力类型和模型维度
        attention_type = config.get('model', {}).get('attention_type', 'self')
        d_model = config.get('model', {}).get('d_model', 256)
        
        if attention_type in ['muse', 'ufo'] and d_model < 128:
            warnings.append("高效注意力机制建议使用较大的模型维度")
        
        return warnings
```

---

**💡 提示**: 良好的配置系统是项目成功的关键，它不仅提高了实验的可重现性，还大大简化了参数调优和实验管理的复杂度。

---

*需要帮助？查看 [FAQ](faq) 或 [故障排除](troubleshooting) 页面。*
