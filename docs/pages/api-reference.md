---
layout: default
title: API 参考
nav_order: 15
parent: 核心文档
permalink: /pages/api-reference/
---

# API 参考
{: .no_toc }

本页面提供 VIVTransformer 项目的完整 API 参考文档，包括所有核心类、函数和接口的详细说明。
{: .fs-6 .fw-300 }

## 目录
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## 核心模型 API

### VIVTransformer 类

主要的 Transformer 模型类，用于涡激振动预测。

```python
class VIVTransformer(nn.Module):
    """
    VIV Transformer 模型主类
    
    Args:
        config (VIVConfig): 模型配置对象
        num_layers (int): Transformer 层数，默认为 6
        d_model (int): 模型维度，默认为 512
        num_heads (int): 注意力头数，默认为 8
        d_ff (int): 前馈网络维度，默认为 2048
        dropout (float): Dropout 概率，默认为 0.1
        max_seq_length (int): 最大序列长度，默认为 1000
    """
    
    def __init__(self, config: VIVConfig, **kwargs):
        pass
    
    def forward(self, 
                flow_data: torch.Tensor,
                structure_data: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            flow_data: 流体数据 [batch_size, seq_len, flow_dim]
            structure_data: 结构数据 [batch_size, seq_len, struct_dim]
            mask: 注意力掩码 [batch_size, seq_len]
            
        Returns:
            Dict containing:
                - 'displacement': 位移预测 [batch_size, seq_len, 3]
                - 'velocity': 速度预测 [batch_size, seq_len, 3]
                - 'force': 力预测 [batch_size, seq_len, 3]
                - 'attention_weights': 注意力权重
        """
        pass
```

### 配置类

```python
class VIVConfig:
    """
    VIV Transformer 配置类
    
    Attributes:
        model_type (str): 模型类型
        d_model (int): 模型维度
        num_layers (int): 层数
        num_heads (int): 注意力头数
        d_ff (int): 前馈网络维度
        dropout (float): Dropout 概率
        activation (str): 激活函数类型
        max_position_embeddings (int): 最大位置编码长度
        flow_input_dim (int): 流体输入维度
        structure_input_dim (int): 结构输入维度
        output_dim (int): 输出维度
    """
    
    def __init__(self, **kwargs):
        pass
    
    @classmethod
    def from_json_file(cls, json_file: str) -> 'VIVConfig':
        """从 JSON 文件加载配置"""
        pass
    
    def to_json_file(self, json_file: str):
        """保存配置到 JSON 文件"""
        pass
```

---

## 注意力机制 API

### MultiHeadAttention 类

```python
class MultiHeadAttention(nn.Module):
    """
    多头注意力机制
    
    Args:
        d_model (int): 模型维度
        num_heads (int): 注意力头数
        dropout (float): Dropout 概率
        temperature (float): 注意力温度参数
    """
    
    def forward(self, 
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query: 查询张量 [batch_size, seq_len, d_model]
            key: 键张量 [batch_size, seq_len, d_model]
            value: 值张量 [batch_size, seq_len, d_model]
            mask: 注意力掩码
            
        Returns:
            output: 注意力输出 [batch_size, seq_len, d_model]
            attention_weights: 注意力权重 [batch_size, num_heads, seq_len, seq_len]
        """
        pass
```

### CrossModalAttention 类

```python
class CrossModalAttention(nn.Module):
    """
    跨模态注意力机制
    
    用于融合流体和结构数据的注意力机制
    """
    
    def forward(self, 
                flow_features: torch.Tensor,
                structure_features: torch.Tensor) -> torch.Tensor:
        """跨模态注意力计算"""
        pass
```

---

## 数据处理 API

### VIVDataset 类

```python
class VIVDataset(Dataset):
    """
    VIV 数据集类
    
    Args:
        data_dir (str): 数据目录路径
        split (str): 数据集分割 ('train', 'val', 'test')
        sequence_length (int): 序列长度
        transform: 数据变换函数
    """
    
    def __init__(self, data_dir: str, split: str = 'train', **kwargs):
        pass
    
    def __len__(self) -> int:
        """返回数据集大小"""
        pass
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        获取单个数据样本
        
        Returns:
            Dict containing:
                - 'flow_data': 流体数据
                - 'structure_data': 结构数据
                - 'target': 目标值
                - 'metadata': 元数据
        """
        pass
```

### DataProcessor 类

```python
class DataProcessor:
    """
    数据预处理器
    
    提供数据标准化、归一化、特征提取等功能
    """
    
    def __init__(self, config: Dict):
        pass
    
    def normalize(self, data: np.ndarray) -> np.ndarray:
        """数据归一化"""
        pass
    
    def extract_features(self, raw_data: np.ndarray) -> np.ndarray:
        """特征提取"""
        pass
    
    def create_sequences(self, data: np.ndarray, seq_length: int) -> np.ndarray:
        """创建序列数据"""
        pass
```

---

## 训练 API

### Trainer 类

```python
class Trainer:
    """
    模型训练器
    
    Args:
        model: 要训练的模型
        config: 训练配置
        train_dataloader: 训练数据加载器
        val_dataloader: 验证数据加载器
        optimizer: 优化器
        scheduler: 学习率调度器
    """
    
    def __init__(self, model, config, **kwargs):
        pass
    
    def train(self) -> Dict[str, List[float]]:
        """
        执行训练
        
        Returns:
            训练历史记录，包含损失值和指标
        """
        pass
    
    def evaluate(self, dataloader) -> Dict[str, float]:
        """
        模型评估
        
        Args:
            dataloader: 评估数据加载器
            
        Returns:
            评估指标字典
        """
        pass
    
    def save_checkpoint(self, filepath: str):
        """保存检查点"""
        pass
    
    def load_checkpoint(self, filepath: str):
        """加载检查点"""
        pass
```

### LossFunction 类

```python
class VIVLoss(nn.Module):
    """
    VIV 专用损失函数
    
    结合物理约束的多任务损失函数
    """
    
    def __init__(self, 
                 mse_weight: float = 1.0,
                 physics_weight: float = 0.1,
                 consistency_weight: float = 0.05):
        pass
    
    def forward(self, 
                predictions: Dict[str, torch.Tensor],
                targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        计算损失
        
        Returns:
            Dict containing:
                - 'total_loss': 总损失
                - 'mse_loss': MSE 损失
                - 'physics_loss': 物理约束损失
                - 'consistency_loss': 一致性损失
        """
        pass
```

---

## 评估 API

### Evaluator 类

```python
class Evaluator:
    """
    模型评估器
    
    提供各种评估指标的计算
    """
    
    def __init__(self, config: Dict):
        pass
    
    def compute_metrics(self, 
                       predictions: np.ndarray,
                       targets: np.ndarray) -> Dict[str, float]:
        """
        计算评估指标
        
        Returns:
            包含各种指标的字典：
            - 'mse': 均方误差
            - 'mae': 平均绝对误差
            - 'rmse': 均方根误差
            - 'r2': R² 分数
            - 'physics_consistency': 物理一致性
        """
        pass
    
    def plot_results(self, 
                    predictions: np.ndarray,
                    targets: np.ndarray,
                    save_path: str):
        """绘制结果对比图"""
        pass
```

---

## 工具函数 API

### 模型工具

```python
def load_model(model_path: str, config_path: str) -> VIVTransformer:
    """
    加载预训练模型
    
    Args:
        model_path: 模型权重文件路径
        config_path: 配置文件路径
        
    Returns:
        加载的模型实例
    """
    pass

def save_model(model: VIVTransformer, save_path: str):
    """
    保存模型
    
    Args:
        model: 要保存的模型
        save_path: 保存路径
    """
    pass
```

### 数据工具

```python
def create_dataloader(dataset: VIVDataset, 
                      batch_size: int = 32,
                      shuffle: bool = True,
                      num_workers: int = 4) -> DataLoader:
    """
    创建数据加载器
    """
    pass

def split_dataset(dataset: VIVDataset, 
                 train_ratio: float = 0.8,
                 val_ratio: float = 0.1) -> Tuple[VIVDataset, VIVDataset, VIVDataset]:
    """
    分割数据集
    
    Returns:
        (train_dataset, val_dataset, test_dataset)
    """
    pass
```

### 可视化工具

```python
def plot_training_curves(history: Dict[str, List[float]], save_path: str):
    """
    绘制训练曲线
    
    Args:
        history: 训练历史记录
        save_path: 保存路径
    """
    pass

def plot_attention_weights(attention_weights: torch.Tensor, 
                          save_path: str,
                          layer_idx: int = 0,
                          head_idx: int = 0):
    """
    可视化注意力权重
    
    Args:
        attention_weights: 注意力权重张量
        save_path: 保存路径
        layer_idx: 层索引
        head_idx: 头索引
    """
    pass

def plot_viv_results(predictions: np.ndarray,
                     targets: np.ndarray,
                     time_steps: np.ndarray,
                     save_path: str):
    """
    绘制 VIV 预测结果
    
    Args:
        predictions: 预测结果
        targets: 真实值
        time_steps: 时间步
        save_path: 保存路径
    """
    pass
```

---

## 配置管理 API

### ConfigManager 类

```python
class ConfigManager:
    """
    配置管理器
    
    统一管理所有配置文件
    """
    
    def __init__(self, config_dir: str):
        pass
    
    def load_config(self, config_name: str) -> Dict:
        """加载指定配置"""
        pass
    
    def save_config(self, config: Dict, config_name: str):
        """保存配置"""
        pass
    
    def merge_configs(self, *configs: Dict) -> Dict:
        """合并多个配置"""
        pass
    
    def validate_config(self, config: Dict) -> bool:
        """验证配置有效性"""
        pass
```

---

## 异常处理

### 自定义异常

```python
class VIVTransformerError(Exception):
    """VIV Transformer 基础异常类"""
    pass

class ConfigurationError(VIVTransformerError):
    """配置错误异常"""
    pass

class DataError(VIVTransformerError):
    """数据错误异常"""
    pass

class ModelError(VIVTransformerError):
    """模型错误异常"""
    pass

class TrainingError(VIVTransformerError):
    """训练错误异常"""
    pass
```

---

## 使用示例

### 基本使用流程

```python
# 1. 加载配置
config = VIVConfig.from_json_file('config/model_config.json')

# 2. 创建模型
model = VIVTransformer(config)

# 3. 准备数据
dataset = VIVDataset('data/', split='train')
dataloader = create_dataloader(dataset, batch_size=32)

# 4. 训练模型
trainer = Trainer(model, config, train_dataloader=dataloader)
history = trainer.train()

# 5. 评估模型
evaluator = Evaluator(config)
metrics = evaluator.compute_metrics(predictions, targets)

# 6. 保存模型
save_model(model, 'models/viv_transformer.pth')
```

### 推理示例

```python
# 加载预训练模型
model = load_model('models/viv_transformer.pth', 'config/model_config.json')
model.eval()

# 准备输入数据
flow_data = torch.randn(1, 100, 64)  # [batch, seq_len, flow_dim]
structure_data = torch.randn(1, 100, 32)  # [batch, seq_len, struct_dim]

# 执行推理
with torch.no_grad():
    outputs = model(flow_data, structure_data)
    displacement = outputs['displacement']
    velocity = outputs['velocity']
    force = outputs['force']
```

---

## 版本信息

- **当前版本**: 1.0.0
- **Python 要求**: >= 3.8
- **PyTorch 要求**: >= 1.9.0
- **最后更新**: 2024年12月

---

## 相关链接

- [快速入门教程](quick-start-tutorial.html) - 快速上手指南
- [架构概览](architecture-overview.html) - 了解整体架构
- [训练指南](training-guide.html) - 详细训练说明
- [示例代码](examples.html) - 实际使用示例
- [故障排除](troubleshooting.html) - 常见问题解决

*需要帮助？查看 [FAQ](faq.html) 或 [故障排除](troubleshooting.html) 页面。*