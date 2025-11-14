---
layout: default
title: 快速开始教程
nav_order: 2
description: "VIVTransformer 快速入门指南"
permalink: /zh/pages/quick-start-tutorial/
lang: zh
ref: quick-start-tutorial
---

# 快速开始教程

本教程将帮助您在5分钟内快速上手 VIVTransformer。

## 🚀 前置要求

在开始之前，请确保您的系统满足以下要求：

- Python 3.8 或更高版本
- PyTorch 2.0 或更高版本
- CUDA 11.8 或更高版本（用于GPU加速）

## 📦 安装

### 1. 克隆仓库

```bash
git clone https://github.com/Pitohuie-AIversion/Sparse_to_Dense_Transformer.git
cd Sparse_to_Dense_Transformer
```

### 2. 创建虚拟环境

```bash
python -m venv vivtransformer-env
source vivtransformer-env/bin/activate  # Linux/Mac
# 或者
vivtransformer-env\Scripts\activate  # Windows
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

## 🎯 第一个示例

### 1. 数据准备

创建您的第一个数据集：

```python
import numpy as np
from vivtransformer.data import VIVDataLoader

# 创建示例数据
n_samples = 1000
n_timesteps = 50
n_features = 10

X = np.random.randn(n_samples, n_timesteps, n_features)
y = np.random.randn(n_samples, 1)

# 创建数据加载器
data_loader = VIVDataLoader(X, y, batch_size=32)
```

### 2. 模型定义

定义您的第一个 VIVTransformer 模型：

```python
from vivtransformer.models import VIVTransformer

# 创建模型
model = VIVTransformer(
    input_dim=n_features,
    d_model=128,
    n_heads=8,
    n_layers=6,
    dropout=0.1
)
```

### 3. 训练模型

训练您的模型：

```python
from vivtransformer.trainer import VIVTrainer

# 创建训练器
trainer = VIVTrainer(
    model=model,
    data_loader=data_loader,
    learning_rate=1e-4,
    n_epochs=100
)

# 开始训练
trainer.train()
```

### 4. 模型评估

评估模型性能：

```python
# 评估模型
metrics = trainer.evaluate()
print(f"MSE: {metrics['mse']:.4f}")
print(f"MAE: {metrics['mae']:.4f}")
print(f"R²: {metrics['r2']:.4f}")
```

## 📊 可视化结果

使用内置的可视化工具：

```python
import matplotlib.pyplot as plt
from vivtransformer.visualization import plot_predictions

# 绘制预测结果
predictions = model.predict(X[:100])
plot_predictions(y[:100], predictions)
plt.show()
```

## 🎉 恭喜！

您已经成功运行了第一个 VIVTransformer 模型！接下来，您可以：

- 探索更多[高级配置]({{ site.baseurl }}/zh/pages/advanced-configuration/)
- 了解[模型架构]({{ site.baseurl }}/zh/pages/architecture-overview/)
- 查看[训练优化技巧]({{ site.baseurl }}/zh/pages/training-optimization/)

## 📞 需要帮助？

如果您遇到问题，请查看：

- [故障排除指南]({{ site.baseurl }}/zh/pages/troubleshooting/)
- [常见问题]({{ site.baseurl }}/zh/pages/faq/)
- [社区支持]({{ site.baseurl }}/zh/pages/community-guide/)

---

<div style="text-align: center; margin-top: 2rem;">
  <p><strong>下一步：</strong> <a href="{{ site.baseurl }}/zh/pages/installation-guide/">安装指南</a></p>
</div>