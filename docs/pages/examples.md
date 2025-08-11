---
layout: default
title: Examples
nav_order: 5
parent: Getting Started
permalink: /pages/examples/
---

# Examples

This page provides practical examples for using the VIVTransformer project.

## 📋 Table of Contents
- Basic Examples
- Data Processing
- Training
- Inference
- Advanced Usage

---

## 🚀 Basic Examples

### 1. Quick Start
The simplest usage: create a model and run a forward pass.

```python
import torch
from vivtransformer import VIVTransformer, VIVConfig

# 1. Create config
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
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

# 3. Prepare sample data
batch_size, seq_len = 4, 100
flow_data = torch.randn(batch_size, seq_len, config.flow_input_dim)
structure_data = torch.randn(batch_size, seq_len, config.structure_input_dim)

# 4. Forward pass
model.eval()
with torch.no_grad():
    outputs = model(flow_data, structure_data)

print("Output shapes:")
for key, value in outputs.items():
    print(f"  {key}: {value.shape}")
```

## 📊 Data Processing

### 2. Data Loading
How to load and preprocess VIV data.

```python
import os
import pandas as pd
from torch.utils.data import DataLoader
from vivtransformer.data import VIVDataset, DataProcessor

# 1. Preprocessing
processor = DataProcessor({
    'normalize': True,
    'sequence_length': 100,
    'overlap': 0.5,
    'features': ['velocity', 'pressure', 'displacement']
})

# 2. Datasets
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

# 3. Dataloaders
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

# 4. Inspect batch
for batch in train_loader:
    print("Batch shapes:")
    print(f"  flow: {batch['flow_data'].shape}")
    print(f"  structure: {batch['structure_data'].shape}")
    print(f"  target: {batch['target'].shape}")
    break
```

## 🎯 Training

### 3. Full Training Loop
A complete training pipeline.

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from vivtransformer import VIVTransformer, VIVConfig
from vivtransformer.training import Trainer, VIVLoss
from vivtransformer.utils import set_seed, save_checkpoint

# 1. Seed
set_seed(42)

# 2. Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# 3. Config
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

# 4. Model
model = VIVTransformer(config).to(device)

# 5. Loss & Optimizer
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

# 6. Training config
training_config = {
    'epochs': 100,
    'save_every': 10,
    'eval_every': 5,
    'early_stopping_patience': 15,
    'gradient_clip_norm': 1.0,
    'log_every': 100
}

# 7. Train
for epoch in range(training_config['epochs']):
    model.train()
    total_loss = 0
    
    for batch_idx, batch in enumerate(train_loader):
        # Move to device
        flow_data = batch['flow_data'].to(device)
        structure_data = batch['structure_data'].to(device)
        target = batch['target'].to(device)
        
        # Forward
        optimizer.zero_grad()
        outputs = model(flow_data, structure_data)
        loss = criterion(outputs, target)
        
        # Backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), training_config['gradient_clip_norm'])
        optimizer.step()
        
        total_loss += loss.item()
        
        if batch_idx % training_config['log_every'] == 0:
            print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.6f}')
    
    # LR schedule
    scheduler.step()
    
    # Validate
    if epoch % training_config['eval_every'] == 0:
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                flow_data = batch['flow_data'].to(device)
                structure_data = batch['structure_data'].to(device)
                target = batch['target'].to(device)
                outputs = model(flow_data, structure_data)
                val_loss += criterion(outputs, target).item()
        print(f'Validation Loss: {val_loss:.6f}')
        
        # Save checkpoint
        if epoch % training_config['save_every'] == 0:
            save_checkpoint(model, optimizer, epoch, 'checkpoints/model_epoch_{epoch}.pt')
```

## 🔮 Inference

### 4. Model Inference
Use a trained model for inference.

```python
# 1. Load model
checkpoint = torch.load('checkpoints/best_model.pt', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# 2. Prepare test data
# ...

# 3. Run inference
with torch.no_grad():
    outputs = model(flow_data, structure_data)

# 4. Post-process results
# ...

# 5. Visualization
import matplotlib.pyplot as plt
fig, axes = plt.subplots(3, 1, figsize=(10, 12))

# Displacement
axes[0].plot(outputs['displacement'][0].cpu().numpy())
axes[0].set_title('Predicted Displacement')
axes[0].set_ylabel('Displacement (m)')

# Velocity
axes[1].plot(outputs['velocity'][0].cpu().numpy())
axes[1].set_title('Predicted Velocity')
axes[1].set_ylabel('Velocity (m/s)')

# Force
axes[2].plot(outputs['force'][0].cpu().numpy())
axes[2].set_title('Predicted Force')
axes[2].set_xlabel('Time Step')
axes[2].set_ylabel('Force (N)')
```

## 🔧 Advanced Usage

### 5. Custom Attention
Implement and use a custom attention mechanism.

```python
"""Custom attention example"""
class CustomAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        # 1) Projections
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)
        
        # 2) Attention scores
        scores = (Q @ K.transpose(-2, -1)) / (Q.shape[-1] ** 0.5)
        
        # 3) Masking (optional)
        # ...
        
        # 4) Softmax
        attn = scores.softmax(dim=-1)
        attn = self.dropout(attn)
        
        # 5) Weighted sum
        out = attn @ V
        
        # 6) Output projection
        out = self.out_proj(out)
        return out

# Use custom attention
model = VIVTransformer(config, attention_class=CustomAttention)
```

### 6. Analysis and Visualization
Analyze performance and visualize attention weights.

```python
# 1. Analyzer
analyzer = ModelAnalyzer(model)

# 2. Complexity
params, flops = analyzer.compute_complexity(input_shape=(1, 100, 64))
print(f"Params: {params:,}")
print(f"FLOPs: {flops:,}")

# 3. Get attention
outputs, attn_weights = model.forward_with_attention(flow_data, structure_data)

# 4. Visualize attention (head 0)
import matplotlib.pyplot as plt
head0 = attn_weights[0, 0].cpu().numpy()
plt.imshow(head0, cmap='viridis')
plt.colorbar()
plt.title('Attention Head 0')
plt.show()

# 5. Pattern analysis
analysis = analyzer.analyze_attention_patterns(attn_weights)
print("Attention pattern analysis:")
print(analysis)
```