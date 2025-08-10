---
layout: default
title: Examples
nav_order: 5
parent: Getting Started
permalink: /pages/examples/
---

<div class="lang-content" data-lang-zh style="display: none;">
<h1>示例代码</h1>
<p class="fs-6 fw-300">本页面提供了VIVTransformer项目的实用代码示例，帮助您快速上手和理解项目的使用方法。</p>

<h2>📋 目录</h2>
<ul>
<li><a href="#基础示例">基础示例</a></li>
<li><a href="#数据处理示例">数据处理示例</a></li>
<li><a href="#训练示例">训练示例</a></li>
<li><a href="#推理示例">推理示例</a></li>
<li><a href="#高级用法">高级用法</a></li>
</ul>

<hr>

<h2 id="基础示例">🚀 基础示例</h2>

<h3>1. 快速开始示例</h3>
<p>最简单的使用示例，展示如何创建模型并进行前向传播。</p>

<pre><code class="language-python">import torch
from vivtransformer import VIVTransformer, VIVConfig

# 1. 创建配置
config = VIVConfig(
    d_model=512,
    num_layers=6,
    num_heads=8,
    d_ff=2048,
    dropout=0.1,
    flow_input_dim=64,
    structure_input_dim=32,
    output_dim=9  # 3D位移 + 速度 + 力
)

# 2. 创建模型
model = VIVTransformer(config)
print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")

# 3. 准备示例数据
batch_size, seq_len = 4, 100
flow_data = torch.randn(batch_size, seq_len, config.flow_input_dim)
structure_data = torch.randn(batch_size, seq_len, config.structure_input_dim)

# 4. 前向传播
model.eval()
with torch.no_grad():
    outputs = model(flow_data, structure_data)
    
print("输出形状:")
for key, value in outputs.items():
    print(f"  {key}: {value.shape}")
</code></pre>

<h2 id="数据处理示例">📊 数据处理示例</h2>

<h3>2. 数据加载示例</h3>
<p>展示如何加载和预处理VIV数据。</p>

<pre><code class="language-python">import os
import pandas as pd
from torch.utils.data import DataLoader
from vivtransformer.data import VIVDataset, DataProcessor

# 1. 数据预处理
processor = DataProcessor({
    'normalize': True,
    'sequence_length': 100,
    'overlap': 0.5,
    'features': ['velocity', 'pressure', 'displacement']
})

# 2. 创建数据集
train_dataset = VIVDataset(
    data_dir='data/train',
    split='train',
    sequence_length=100,
    processor=processor
)

val_dataset = VIVDataset(
    data_dir='data/val',
    split='val',
    sequence_length=100,
    processor=processor
)

# 3. 创建数据加载器
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

# 4. 检查数据
for batch in train_loader:
    print("批次数据形状:")
    print(f"  流体数据: {batch['flow_data'].shape}")
    print(f"  结构数据: {batch['structure_data'].shape}")
    print(f"  目标数据: {batch['target'].shape}")
    break
</code></pre>

<h2 id="训练示例">🎯 训练示例</h2>

<h3>3. 完整训练流程</h3>
<p>展示完整的模型训练过程。</p>

<pre><code class="language-python">import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from vivtransformer import VIVTransformer, VIVConfig
from vivtransformer.training import Trainer, VIVLoss
from vivtransformer.utils import set_seed, save_checkpoint

# 1. 设置随机种子
set_seed(42)

# 2. 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 3. 模型配置
config = VIVConfig(
    d_model=512,
    num_layers=6,
    num_heads=8,
    d_ff=2048,
    dropout=0.1,
    max_seq_length=1000,
    flow_input_dim=64,
    structure_input_dim=32
)

# 4. 创建模型
model = VIVTransformer(config).to(device)

# 5. 损失函数和优化器
criterion = VIVLoss(
    mse_weight=1.0,
    physics_weight=0.1,
    consistency_weight=0.05
)

optimizer = optim.AdamW(
    model.parameters(),
    lr=1e-4,
    weight_decay=1e-5
)

scheduler = CosineAnnealingLR(
    optimizer,
    T_max=100,
    eta_min=1e-6
)

# 6. 训练配置
training_config = {
    'epochs': 100,
    'save_every': 10,
    'eval_every': 5,
    'early_stopping_patience': 15,
    'gradient_clip_norm': 1.0,
    'log_every': 100
}

# 7. 开始训练
for epoch in range(training_config['epochs']):
    model.train()
    total_loss = 0
    
    for batch_idx, batch in enumerate(train_loader):
        # 数据移到设备
        flow_data = batch['flow_data'].to(device)
        structure_data = batch['structure_data'].to(device)
        target = batch['target'].to(device)
        
        # 前向传播
        optimizer.zero_grad()
        outputs = model(flow_data, structure_data)
        loss = criterion(outputs, target)
        
        # 反向传播
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), training_config['gradient_clip_norm'])
        optimizer.step()
        
        total_loss += loss.item()
        
        if batch_idx % training_config['log_every'] == 0:
            print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.6f}')
    
    # 学习率调度
    scheduler.step()
    
    # 验证
    if epoch % training_config['eval_every'] == 0:
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                flow_data = batch['flow_data'].to(device)
                structure_data = batch['structure_data'].to(device)
                target = batch['target'].to(device)
                
                outputs = model(flow_data, structure_data)
                loss = criterion(outputs, target)
                val_loss += loss.item()
        
        print(f'Epoch {epoch}, Train Loss: {total_loss/len(train_loader):.6f}, Val Loss: {val_loss/len(val_loader):.6f}')
    
    # 保存检查点
    if epoch % training_config['save_every'] == 0:
        save_checkpoint({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': total_loss/len(train_loader)
        }, f'checkpoint_epoch_{epoch}.pth')
</code></pre>

<h2 id="推理示例">🔮 推理示例</h2>

<h3>4. 模型推理</h3>
<p>展示如何使用训练好的模型进行推理。</p>

<pre><code class="language-python">import torch
import numpy as np
import matplotlib.pyplot as plt
from vivtransformer import VIVTransformer, VIVConfig
from vivtransformer.utils import load_checkpoint

# 1. 加载模型
config = VIVConfig.from_json('config.json')
model = VIVTransformer(config)
checkpoint = load_checkpoint('best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# 2. 准备测试数据
test_flow_data = torch.randn(1, 200, config.flow_input_dim)
test_structure_data = torch.randn(1, 200, config.structure_input_dim)

# 3. 进行推理
with torch.no_grad():
    predictions = model(test_flow_data, test_structure_data)

# 4. 处理结果
displacement = predictions['displacement'].squeeze().numpy()
velocity = predictions['velocity'].squeeze().numpy()
force = predictions['force'].squeeze().numpy()

# 5. 可视化结果
fig, axes = plt.subplots(3, 1, figsize=(12, 10))

# 位移
axes[0].plot(displacement)
axes[0].set_title('结构位移预测')
axes[0].set_ylabel('位移 (m)')
axes[0].grid(True)

# 速度
axes[1].plot(velocity)
axes[1].set_title('结构速度预测')
axes[1].set_ylabel('速度 (m/s)')
axes[1].grid(True)

# 力
axes[2].plot(force)
axes[2].set_title('流体力预测')
axes[2].set_xlabel('时间步')
axes[2].set_ylabel('力 (N)')
axes[2].grid(True)

plt.tight_layout()
plt.savefig('predictions.png', dpi=300, bbox_inches='tight')
plt.show()
</code></pre>

<h2 id="高级用法">🔧 高级用法</h2>

<h3>5. 自定义注意力机制</h3>
<p>展示如何实现和使用自定义注意力机制。</p>

<pre><code class="language-python">import torch
import torch.nn as nn
from vivtransformer.attention import BaseAttention

class CustomAttention(BaseAttention):
    """自定义注意力机制示例"""
    
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, query, key, value, mask=None):
        batch_size, seq_len, d_model = query.size()
        
        # 1. 线性变换
        Q = self.w_q(query).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # 2. 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)
        
        # 3. 应用掩码
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # 4. 应用softmax
        attention_weights = torch.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # 5. 应用注意力权重
        context = torch.matmul(attention_weights, V)
        
        # 6. 重塑和输出投影
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len, d_model
        )
        output = self.w_o(context)
        
        # 7. 残差连接和层归一化
        output = self.layer_norm(output + query)
        
        return output, attention_weights

# 使用自定义注意力机制
config = VIVConfig(
    d_model=512,
    num_layers=6,
    num_heads=8,
    attention_class=CustomAttention  # 指定自定义注意力
)

model = VIVTransformer(config)
</code></pre>

<h3>6. 模型分析和可视化</h3>
<p>展示如何分析模型性能和可视化注意力权重。</p>

<pre><code class="language-python">import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from vivtransformer.analysis import AttentionVisualizer, ModelAnalyzer

# 1. 创建分析器
analyzer = ModelAnalyzer(model)
visualizer = AttentionVisualizer()

# 2. 分析模型复杂度
flops, params = analyzer.compute_flops(input_shape=(1, 100, 64))
print(f"模型参数数量: {params:,}")
print(f"浮点运算数: {flops:,}")

# 3. 获取注意力权重
with torch.no_grad():
    outputs, attention_weights = model(test_flow_data, test_structure_data, return_attention=True)

# 4. 可视化注意力权重
for layer_idx, layer_attention in enumerate(attention_weights):
    # 选择第一个头的注意力权重
    attention = layer_attention[0, 0].cpu().numpy()  # [seq_len, seq_len]
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(attention, cmap='Blues', cbar=True)
    plt.title(f'Layer {layer_idx + 1} - Attention Weights')
    plt.xlabel('Key Position')
    plt.ylabel('Query Position')
    plt.savefig(f'attention_layer_{layer_idx + 1}.png', dpi=300, bbox_inches='tight')
    plt.show()

# 5. 分析注意力模式
attention_patterns = analyzer.analyze_attention_patterns(attention_weights)
print("注意力模式分析:")
for pattern_name, score in attention_patterns.items():
    print(f"  {pattern_name}: {score:.4f}")
</code></pre>
</div>

<div class="lang-content" data-lang-en>
<h1>Examples</h1>
<p class="fs-6 fw-300">This page provides practical code examples for the VIVTransformer project to help you get started quickly and understand how to use the project.</p>

<h2>📋 Table of Contents</h2>
<ul>
<li><a href="#basic-examples">Basic Examples</a></li>
<li><a href="#data-processing-examples">Data Processing Examples</a></li>
<li><a href="#training-examples">Training Examples</a></li>
<li><a href="#inference-examples">Inference Examples</a></li>
<li><a href="#advanced-usage">Advanced Usage</a></li>
</ul>

<hr>

<h2 id="basic-examples">🚀 Basic Examples</h2>

<h3>1. Quick Start Example</h3>
<p>The simplest usage example showing how to create a model and perform forward propagation.</p>

<pre><code class="language-python">import torch
from vivtransformer import VIVTransformer, VIVConfig

# 1. Create configuration
config = VIVConfig(
    d_model=512,
    num_layers=6,
    num_heads=8,
    d_ff=2048,
    dropout=0.1,
    flow_input_dim=64,
    structure_input_dim=32,
    output_dim=9  # 3D displacement + velocity + force
)

# 2. Create model
model = VIVTransformer(config)
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# 3. Prepare sample data
batch_size, seq_len = 4, 100
flow_data = torch.randn(batch_size, seq_len, config.flow_input_dim)
structure_data = torch.randn(batch_size, seq_len, config.structure_input_dim)

# 4. Forward propagation
model.eval()
with torch.no_grad():
    outputs = model(flow_data, structure_data)
    
print("Output shapes:")
for key, value in outputs.items():
    print(f"  {key}: {value.shape}")
</code></pre>

<h2 id="data-processing-examples">📊 Data Processing Examples</h2>

<h3>2. Data Loading Example</h3>
<p>Shows how to load and preprocess VIV data.</p>

<pre><code class="language-python">import os
import pandas as pd
from torch.utils.data import DataLoader
from vivtransformer.data import VIVDataset, DataProcessor

# 1. Data preprocessing
processor = DataProcessor({
    'normalize': True,
    'sequence_length': 100,
    'overlap': 0.5,
    'features': ['velocity', 'pressure', 'displacement']
})

# 2. Create datasets
train_dataset = VIVDataset(
    data_dir='data/train',
    split='train',
    sequence_length=100,
    processor=processor
)

val_dataset = VIVDataset(
    data_dir='data/val',
    split='val',
    sequence_length=100,
    processor=processor
)

# 3. Create data loaders
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

# 4. Check data
for batch in train_loader:
    print("Batch data shapes:")
    print(f"  Flow data: {batch['flow_data'].shape}")
    print(f"  Structure data: {batch['structure_data'].shape}")
    print(f"  Target data: {batch['target'].shape}")
    break
</code></pre>

<h2 id="training-examples">🎯 Training Examples</h2>

<h3>3. Complete Training Process</h3>
<p>Shows the complete model training process.</p>

<pre><code class="language-python">import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from vivtransformer import VIVTransformer, VIVConfig
from vivtransformer.training import Trainer, VIVLoss
from vivtransformer.utils import set_seed, save_checkpoint

# 1. Set random seed
set_seed(42)

# 2. Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 3. Model configuration
config = VIVConfig(
    d_model=512,
    num_layers=6,
    num_heads=8,
    d_ff=2048,
    dropout=0.1,
    max_seq_length=1000,
    flow_input_dim=64,
    structure_input_dim=32
)

# 4. Create model
model = VIVTransformer(config).to(device)

# 5. Loss function and optimizer
criterion = VIVLoss(
    mse_weight=1.0,
    physics_weight=0.1,
    consistency_weight=0.05
)

optimizer = optim.AdamW(
    model.parameters(),
    lr=1e-4,
    weight_decay=1e-5
)

scheduler = CosineAnnealingLR(
    optimizer,
    T_max=100,
    eta_min=1e-6
)

# 6. Training configuration
training_config = {
    'epochs': 100,
    'save_every': 10,
    'eval_every': 5,
    'early_stopping_patience': 15,
    'gradient_clip_norm': 1.0,
    'log_every': 100
}

# 7. Start training
for epoch in range(training_config['epochs']):
    model.train()
    total_loss = 0
    
    for batch_idx, batch in enumerate(train_loader):
        # Move data to device
        flow_data = batch['flow_data'].to(device)
        structure_data = batch['structure_data'].to(device)
        target = batch['target'].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(flow_data, structure_data)
        loss = criterion(outputs, target)
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), training_config['gradient_clip_norm'])
        optimizer.step()
        
        total_loss += loss.item()
        
        if batch_idx % training_config['log_every'] == 0:
            print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.6f}')
    
    # Learning rate scheduling
    scheduler.step()
    
    # Validation
    if epoch % training_config['eval_every'] == 0:
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                flow_data = batch['flow_data'].to(device)
                structure_data = batch['structure_data'].to(device)
                target = batch['target'].to(device)
                
                outputs = model(flow_data, structure_data)
                loss = criterion(outputs, target)
                val_loss += loss.item()
        
        print(f'Epoch {epoch}, Train Loss: {total_loss/len(train_loader):.6f}, Val Loss: {val_loss/len(val_loader):.6f}')
    
    # Save checkpoint
    if epoch % training_config['save_every'] == 0:
        save_checkpoint({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': total_loss/len(train_loader)
        }, f'checkpoint_epoch_{epoch}.pth')
</code></pre>

<h2 id="inference-examples">🔮 Inference Examples</h2>

<h3>4. Model Inference</h3>
<p>Shows how to use a trained model for inference.</p>

<pre><code class="language-python">import torch
import numpy as np
import matplotlib.pyplot as plt
from vivtransformer import VIVTransformer, VIVConfig
from vivtransformer.utils import load_checkpoint

# 1. Load model
config = VIVConfig.from_json('config.json')
model = VIVTransformer(config)
checkpoint = load_checkpoint('best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# 2. Prepare test data
test_flow_data = torch.randn(1, 200, config.flow_input_dim)
test_structure_data = torch.randn(1, 200, config.structure_input_dim)

# 3. Perform inference
with torch.no_grad():
    predictions = model(test_flow_data, test_structure_data)

# 4. Process results
displacement = predictions['displacement'].squeeze().numpy()
velocity = predictions['velocity'].squeeze().numpy()
force = predictions['force'].squeeze().numpy()

# 5. Visualize results
fig, axes = plt.subplots(3, 1, figsize=(12, 10))

# Displacement
axes[0].plot(displacement)
axes[0].set_title('Structure Displacement Prediction')
axes[0].set_ylabel('Displacement (m)')
axes[0].grid(True)

# Velocity
axes[1].plot(velocity)
axes[1].set_title('Structure Velocity Prediction')
axes[1].set_ylabel('Velocity (m/s)')
axes[1].grid(True)

# Force
axes[2].plot(force)
axes[2].set_title('Fluid Force Prediction')
axes[2].set_xlabel('Time Step')
axes[2].set_ylabel('Force (N)')
axes[2].grid(True)

plt.tight_layout()
plt.savefig('predictions.png', dpi=300, bbox_inches='tight')
plt.show()
</code></pre>

<h2 id="advanced-usage">🔧 Advanced Usage</h2>

<h3>5. Custom Attention Mechanism</h3>
<p>Shows how to implement and use custom attention mechanisms.</p>

<pre><code class="language-python">import torch
import torch.nn as nn
from vivtransformer.attention import BaseAttention

class CustomAttention(BaseAttention):
    """Custom attention mechanism example"""
    
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, query, key, value, mask=None):
        batch_size, seq_len, d_model = query.size()
        
        # 1. Linear transformations
        Q = self.w_q(query).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # 2. Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)
        
        # 3. Apply mask
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # 4. Apply softmax
        attention_weights = torch.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # 5. Apply attention weights
        context = torch.matmul(attention_weights, V)
        
        # 6. Reshape and output projection
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len, d_model
        )
        output = self.w_o(context)
        
        # 7. Residual connection and layer normalization
        output = self.layer_norm(output + query)
        
        return output, attention_weights

# Use custom attention mechanism
config = VIVConfig(
    d_model=512,
    num_layers=6,
    num_heads=8,
    attention_class=CustomAttention  # Specify custom attention
)

model = VIVTransformer(config)
</code></pre>

<h3>6. Model Analysis and Visualization</h3>
<p>Shows how to analyze model performance and visualize attention weights.</p>

<pre><code class="language-python">import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from vivtransformer.analysis import AttentionVisualizer, ModelAnalyzer

# 1. Create analyzer
analyzer = ModelAnalyzer(model)
visualizer = AttentionVisualizer()

# 2. Analyze model complexity
flops, params = analyzer.compute_flops(input_shape=(1, 100, 64))
print(f"Model parameters: {params:,}")
print(f"FLOPs: {flops:,}")

# 3. Get attention weights
with torch.no_grad():
    outputs, attention_weights = model(test_flow_data, test_structure_data, return_attention=True)

# 4. Visualize attention weights
for layer_idx, layer_attention in enumerate(attention_weights):
    # Select attention weights from the first head
    attention = layer_attention[0, 0].cpu().numpy()  # [seq_len, seq_len]
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(attention, cmap='Blues', cbar=True)
    plt.title(f'Layer {layer_idx + 1} - Attention Weights')
    plt.xlabel('Key Position')
    plt.ylabel('Query Position')
    plt.savefig(f'attention_layer_{layer_idx + 1}.png', dpi=300, bbox_inches='tight')
    plt.show()

# 5. Analyze attention patterns
attention_patterns = analyzer.analyze_attention_patterns(attention_weights)
print("Attention pattern analysis:")
for pattern_name, score in attention_patterns.items():
    print(f"  {pattern_name}: {score:.4f}")
</code></pre>
</div>