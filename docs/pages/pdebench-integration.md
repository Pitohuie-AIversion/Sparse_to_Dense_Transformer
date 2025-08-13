---
layout: default
title: PDEBench Integration Guide
parent: Training & Optimization
nav_order: 6
description: "Complete guide for using PDEBench dataset with VIVTransformer"
permalink: /pages/pdebench-integration/
---

# PDEBench Integration Guide

This guide introduces how to use the PDEBench dataset for partial differential equation (PDE) solving experiments within the VIVTransformer project.

## 📋 Table of Contents {#table-of-contents}

- [Overview](#overview)
- [Installation and Setup](#installation-and-setup)
- [Data Preparation](#data-preparation)
- [Configuration](#configuration)
- [Usage Examples](#usage-examples)
- [Supported PDE Types](#supported-pde-types)
- [Performance Optimization](#performance-optimization)
- [Troubleshooting](#troubleshooting)

## Overview {#overview}

PDEBench is a comprehensive benchmark dataset for partial differential equations, containing multiple types of PDE problems:

- **Navier-Stokes equations** (incompressible flow)
- **Darcy flow equations**
- **Shallow water equations**
- **Advection equations**
- **Burgers equations**
- **Reaction-diffusion equations**

### Key Features

- 🔬 **Comprehensive Coverage**: Multiple PDE types with varying complexity
- 📊 **Standardized Format**: HDF5 format for efficient data loading
- 🎯 **Benchmark Quality**: Rigorous evaluation metrics and baselines
- 🔧 **Easy Integration**: Seamless integration with VIVTransformer training pipeline

## Installation and Setup {#installation-and-setup}

### 1. Install Dependencies

```bash
# Install required packages
pip install h5py scipy scikit-learn seaborn tqdm

# Verify installation
python -c "import h5py; print('h5py version:', h5py.version.version)"
```

### 2. Download PDEBench Dataset

Visit the [PDEBench GitHub repository](https://github.com/pdebench/PDEBench) to download the required data files.

Alternative download sources:
- [Official Data Repository](https://darus.uni-stuttgart.de/dataset.xhtml?persistentId=doi:10.18419/darus-2986)
- [PDEBench Documentation](https://pdebench.readthedocs.io/)

### 3. Directory Structure

Organize the downloaded data files in the following structure:

```
data/
└── pdebench/
    ├── ns_incom_inhom_2d.h5          # Navier-Stokes incompressible flow
    ├── darcy_flow_2d.h5              # Darcy flow equation
    ├── shallow_water_2d.h5           # Shallow water equations
    ├── advection_2d.h5               # Advection equations
    ├── burgers_2d.h5                 # Burgers equations
    └── reaction_diffusion_2d.h5      # Reaction-diffusion equations
```

## Data Preparation {#data-preparation}

### Data Format Overview

PDEBench datasets use HDF5 format with the following structure:

```python
# Data shape: [N, T, H, W, C]
# N: Number of samples
# T: Time steps
# H, W: Spatial dimensions (height, width)
# C: Channels (e.g., velocity components)
```

### Data Loading Example

```python
import h5py
import numpy as np
from torch.utils.data import Dataset, DataLoader

class PDEBenchDataset(Dataset):
    def __init__(self, data_file, sequence_length=10, spatial_resolution=(64, 64)):
        self.data_file = data_file
        self.sequence_length = sequence_length
        self.spatial_resolution = spatial_resolution
        
        with h5py.File(data_file, 'r') as f:
            self.data = f['data'][:]  # Load all data into memory
            
        self.data = self._preprocess_data()
    
    def _preprocess_data(self):
        """Preprocess the data"""
        # Reshape spatial dimensions if needed
        N, T, H, W, C = self.data.shape
        target_h, target_w = self.spatial_resolution
        
        if (H, W) != (target_h, target_w):
            # Resize spatial dimensions
            from scipy.ndimage import zoom
            scale_factors = (1, 1, target_h/H, target_w/W, 1)
            self.data = zoom(self.data, scale_factors)
        
        # Flatten spatial dimensions: [N, T, H*W*C]
        N, T, H, W, C = self.data.shape
        self.data = self.data.reshape(N, T, H*W*C)
        
        # Normalize data
        self.data = (self.data - self.data.mean()) / self.data.std()
        
        return self.data
    
    def __len__(self):
        return len(self.data) - self.sequence_length
    
    def __getitem__(self, idx):
        # Input: sequence_length steps
        # Target: next step
        input_seq = self.data[idx:idx + self.sequence_length]
        target = self.data[idx + self.sequence_length]
        
        return {
            'input': torch.tensor(input_seq, dtype=torch.float32),
            'target': torch.tensor(target, dtype=torch.float32)
        }
```

## Configuration {#configuration}

### Basic Configuration

Add PDEBench settings to your config file:

```yaml
# configs/pdebench_config.yaml
data:
  use_pdebench: true
  batch_size: 16
  num_workers: 4
  normalize: true
  use_augmentation: false

current_pde: "ns_incom"  # Current PDE type

pdebench:
  data_root: "./data/pdebench"
  sequence_length: 10
  
  pde_configs:
    ns_incom:
      data_file: "ns_incom_inhom_2d.h5"
      spatial_resolution: [64, 64]
      sequence_length: 49
      input_dim: 4096
      output_dim: 4096
      description: "Navier-Stokes incompressible flow"
    
    darcy_flow:
      data_file: "darcy_flow_2d.h5"
      spatial_resolution: [32, 32]
      sequence_length: 1
      input_dim: 1024
      output_dim: 1024
      description: "Darcy flow equation"
```

### Advanced Configuration

```yaml
pdebench:
  # Data preprocessing options
  preprocessing:
    normalize_method: "standard"  # standard, minmax, robust
    augmentation:
      enable: true
      rotation: true
      flip: true
      noise_level: 0.01
  
  # Training specific settings
  training:
    split_ratio: [0.7, 0.15, 0.15]  # train, val, test
    validation_frequency: 5
    early_stopping_patience: 20
  
  # Evaluation metrics
  metrics:
    - "relative_l2_error"
    - "mse"
    - "mae"
    - "ssim"
```

## Usage Examples {#usage-examples}

### Quick Start Example

```python
# examples/pdebench_quick_start.py
import torch
from pathlib import Path
from config import load_config
from data.pdebench_adapter import get_pdebench_loaders
from models.transformer import VIVTransformer

def pdebench_quick_start():
    """Quick start example for PDEBench"""
    
    # Load configuration
    config = load_config('configs/pdebench_config.yaml')
    
    # Get data loaders
    train_loader, val_loader, test_loader = get_pdebench_loaders(config)
    
    # Initialize model
    model_config = config['model']
    model_config['input_dim'] = config['pdebench']['pde_configs'][config['current_pde']]['input_dim']
    model_config['output_dim'] = config['pdebench']['pde_configs'][config['current_pde']]['output_dim']
    
    model = VIVTransformer(model_config)
    
    # Training setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = torch.nn.MSELoss()
    
    # Training loop
    model.train()
    for epoch in range(10):
        for batch_idx, batch in enumerate(train_loader):
            input_data = batch['input'].to(device)
            target = batch['target'].to(device)
            
            # Forward pass
            output = model(input_data)
            loss = criterion(output, target)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if batch_idx % 10 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.6f}')

if __name__ == "__main__":
    pdebench_quick_start()
```

### Full Training Script

```python
# examples/pdebench_full_training.py
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import time

def main():
    """Complete PDEBench training example"""
    
    # Configuration
    config_path = Path('configs/pdebench_config.yaml')
    config = load_config(config_path)
    
    # Setup logging
    log_dir = Path('logs') / f"pdebench_{config['current_pde']}_{int(time.time())}"
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir)
    
    # Data loaders
    train_loader, val_loader, test_loader = get_pdebench_loaders(config)
    print(f"Dataset: {config['current_pde']}")
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    
    # Model
    model = create_model_from_config(config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Training components
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=config['training']['epochs']
    )
    
    criterion = create_loss_function(config)
    
    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(config['training']['epochs']):
        # Training phase
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        
        # Validation phase
        if epoch % config['pdebench']['training']['validation_frequency'] == 0:
            val_loss = validate_epoch(model, val_loader, criterion, device)
            
            # Logging
            writer.add_scalar('Loss/Train', train_loss, epoch)
            writer.add_scalar('Loss/Validation', val_loss, epoch)
            writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
            
            print(f'Epoch {epoch:3d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}')
            
            # Early stopping and model saving
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                
                # Save best model
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_loss': best_val_loss,
                    'config': config
                }, log_dir / 'best_model.pt')
                
            else:
                patience_counter += 1
                if patience_counter >= config['pdebench']['training']['early_stopping_patience']:
                    print(f'Early stopping at epoch {epoch}')
                    break
        
        scheduler.step()
    
    # Final evaluation
    test_metrics = evaluate_model(model, test_loader, device)
    print(f'Test Results: {test_metrics}')
    
    writer.close()

if __name__ == "__main__":
    main()
```

## Supported PDE Types {#supported-pde-types}

### 1. Navier-Stokes (Incompressible)

```yaml
ns_incom:
  description: "2D incompressible Navier-Stokes equations"
  spatial_resolution: [64, 64]
  time_steps: 49
  variables: ["velocity_x", "velocity_y", "pressure"]
  physics: "Fluid dynamics"
```

### 2. Darcy Flow

```yaml
darcy_flow:
  description: "Darcy flow in porous media"
  spatial_resolution: [32, 32] 
  time_steps: 1
  variables: ["permeability", "pressure"]
  physics: "Porous media flow"
```

### 3. Shallow Water

```yaml
shallow_water:
  description: "2D shallow water equations"
  spatial_resolution: [128, 128]
  time_steps: 40
  variables: ["height", "velocity_x", "velocity_y"]
  physics: "Surface water dynamics"
```

## Performance Optimization {#performance-optimization}

### Memory Optimization

```python
# Optimize data loading for large datasets
def optimize_dataloader(config):
    return DataLoader(
        dataset,
        batch_size=config['data']['batch_size'],
        num_workers=min(8, config['data']['num_workers']),
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )
```

### Training Acceleration

```python
# Use mixed precision training
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for batch in train_loader:
    with autocast():
        output = model(input_data)
        loss = criterion(output, target)
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

## Troubleshooting {#troubleshooting}

### Common Issues

#### 1. Data File Not Found
```
FileNotFoundError: PDEBench data file does not exist
```
**Solution**: Check data file path and filename in configuration

#### 2. Memory Issues
```
CUDA out of memory
```
**Solution**: Reduce batch size or use gradient accumulation

#### 3. Dimension Mismatch
```
RuntimeError: size mismatch
```
**Solution**: Verify model input/output dimensions in configuration

### Debug Tools

```python
# Data verification script
def verify_pdebench_data(data_file):
    """Verify PDEBench data file integrity"""
    import h5py
    
    try:
        with h5py.File(data_file, 'r') as f:
            print(f"Keys: {list(f.keys())}")
            if 'data' in f:
                data_shape = f['data'].shape
                print(f"Data shape: {data_shape}")
                print(f"Data type: {f['data'].dtype}")
                print(f"File size: {Path(data_file).stat().st_size / 1024**2:.1f} MB")
            return True
    except Exception as e:
        print(f"Error reading file: {e}")
        return False

# Usage
verify_pdebench_data('data/pdebench/ns_incom_inhom_2d.h5')
```

## Related Resources {#related-resources}

### Documentation
- [PDEBench Paper](https://arxiv.org/abs/2210.07182)
- [PDEBench GitHub](https://github.com/pdebench/PDEBench)
- [Official Documentation](https://pdebench.readthedocs.io/)

### Related Pages
- [Training Guide]({{ site.baseurl }}/pages/training-guide/) - Complete training workflows
- [Configuration System]({{ site.baseurl }}/pages/configuration-system/) - Configuration management
- [Performance Optimization]({{ site.baseurl }}/pages/performance-optimization/) - Training optimization techniques

---

**Note**: PDEBench dataset follows its original license terms. Please review the license before use.