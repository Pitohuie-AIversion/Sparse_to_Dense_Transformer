---
layout: default
title: Architecture Overview
parent: Core Documentation
nav_order: 1
permalink: /pages/architecture-overview/
description: "VIVTransformer overall architecture design and core components"
---

# Project Architecture Overview {#project-architecture-overview}

This document provides a detailed introduction to the overall architecture design, module organization, and data flow processing of the VIVTransformer project.

## 📋 Table of Contents {#table-of-contents}

- [Overall Architecture](#overall-architecture)
- [Core Modules](#core-modules)
- [Data Flow Processing](#data-flow-processing)
- [Configuration System](#configuration-system)
- [Experiment Management](#experiment-management)
- [Results Analysis](#results-analysis)
- [Extensibility Design](#extensibility-design)

## Overall Architecture {#overall-architecture}

### 🏗️ System Architecture Diagram {#system-architecture-diagram}

```
├── 📁 modify_multi_attention/          # Core code directory
│   ├── 📁 configs/                     # Configuration files
│   │   ├── config.yaml                 # Main configuration file
│   │   └── 📁 loss_configs/            # Loss function configurations
│   ├── 📁 data/                        # Data processing module
│   │   ├── dataloader.py               # Data loader
│   │   └── preprocessing.py            # Data preprocessing
│   ├── 📁 mymodels/                    # Model definitions
│   │   ├── 📁 attention/               # Attention mechanism implementations
│   │   ├── model_factory.py            # Model factory
│   │   └── vivtransformer.py           # Main model
│   ├── 📁 training/                    # Training module
│   │   ├── experiment.py               # Experiment management
│   │   ├── trainer.py                  # Trainer
│   │   └── evaluator.py                # Evaluator
│   ├── 📁 utils/                       # Utility module
│   │   ├── config.py                   # Configuration management
│   │   ├── logger.py                   # Logging system
│   │   ├── svd10_loss.py               # SVD loss function
│   │   └── system.py                   # System utilities
│   ├── main.py                         # Main program entry
│   └── attention_test.py               # Attention testing
├── 📁 attention_results/               # Experiment results
└── 📁 wiki/                           # Project documentation
```

### 🔄 Architecture Design Principles {#architecture-design-principles}

1. **Modular Design**: Each functional module is independent, facilitating maintenance and extension
2. **Configuration-Driven**: All experimental parameters are controlled through configuration files
3. **Factory Pattern**: Uses factory pattern to create different attention mechanisms
4. **Plugin Architecture**: New attention mechanisms can be easily integrated
5. **Layered Design**: Clear separation of data layer, model layer, training layer, and application layer

## Core Modules {#core-modules}

### 🎯 1. Main Program Module (main.py) {#1-main-program-module-main-py}

**Responsibilities**:
- Parse command line arguments
- Load configuration files
- Coordinate module execution
- Manage experiment workflow

**Key Features**:
```python
def main(config_path=None):
    # 1. Parameter parsing
    parser = argparse.ArgumentParser()
    
    # 2. Configuration loading
    cfg = load_config(config_path)
    
    # 3. Environment setup
    set_seed(cfg["global"]["seed"])
    
    # 4. Data preparation
    train_loader, valid_loader, test_loader = get_loaders(...)
    
    # 5. Experiment execution
    for loss_cfg in loss_configs:
        for attn_type in ATTENTION_TYPES:
            run_experiment(...)
```

### 🏭 2. Model Factory (mymodels/model_factory.py) {#2-model-factory-mymodels-model-factory-py}

**Responsibilities**:
- Create different types of attention mechanisms
- Manage model configurations
- Provide unified model interface

**Design Pattern**:
```python
class ModelFactory:
    @staticmethod
    def create_attention(attention_type, **kwargs):
        attention_map = {
            'self': SelfAttention,
            'muse': MuseAttention,
            'ufo': UFOAttention,
            # ... Other attention mechanisms
        }
        return attention_map[attention_type](**kwargs)
```

### 📊 3. Data Processing Module (data/) {#3-data-processing-module-data}

**Components**:
- **dataloader.py**: Data loading and batching
- **preprocessing.py**: Data preprocessing and augmentation

**Features**:
- Support for multiple data formats
- Automatic data augmentation
- Memory-optimized data loading
- Distributed data processing

```python
def get_loaders(data_path, batch_size, use_augmentation, crop_size):
    # Dataset creation
    dataset = VIVDataset(data_path, crop_size, use_augmentation)
    
    # Data splitting
    train_set, valid_set, test_set = split_dataset(dataset)
    
    # Data loaders
    train_loader = DataLoader(train_set, batch_size, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size, shuffle=False)
    
    return train_loader, valid_loader, test_loader
```

### 🧠 4. Attention Mechanism Module (mymodels/attention/) {#4-attention-mechanism-module-mymodels-attention}

**Architecture**:
```
attention/
├── base_attention.py           # Base attention interface
├── self_attention.py           # Self-attention
├── efficient_attention.py      # Efficient attention
├── mobile_attention.py         # Mobile attention
├── spatial_attention.py        # Spatial attention
├── channel_attention.py        # Channel attention
└── hybrid_attention.py         # Hybrid attention
```

**Base Interface**:
```python
class BaseAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
    
    def forward(self, x):
        raise NotImplementedError
    
    def get_attention_weights(self, x):
        """Return attention weights for visualization"""
        raise NotImplementedError
```

### 🎓 5. Training Module (training/) {#5-training-module-training}

**Components**:
- **experiment.py**: Experiment management and coordination
- **trainer.py**: Model training logic
- **evaluator.py**: Model evaluation logic

**Experiment Management Flow**:
```python
def run_experiment(cfg, loss_cfg, loss_config_id, attn_type, 
                  parent_dir, train_loader, valid_loader, 
                  test_loader, device, logger, debug=False):
    # 1. Create model
    model = create_model(cfg, attn_type)
    
    # 2. Setup loss function
    criterion = TotalLossWithSVD(loss_cfg)
    
    # 3. Create trainer
    trainer = Trainer(model, criterion, optimizer)
    
    # 4. Execute training
    trainer.train(train_loader, valid_loader)
    
    # 5. Model evaluation
    evaluator = Evaluator(model)
    test_loss = evaluator.evaluate(test_loader)
    
    return test_loss
```

### 🛠️ 6. Utility Module (utils/) {#6-utility-module-utils}

**Component Functions**:

| Module | Function | Key Features |
|------|------|----------|
| config.py | Configuration management | YAML parsing, configuration validation, default value handling |
| logger.py | Logging system | Multi-level logging, file output, real-time monitoring |
| svd10_loss.py | SVD loss | Singular value decomposition, regularization, TopK selection |
| system.py | System utilities | GPU management, memory control, random seed |

## Data Flow Processing {#data-flow-processing}

### 📈 Data Flow Diagram {#data-flow-diagram}

```
Raw Data → Preprocessing → Data Augmentation → Batching → Model Input
    ↓
Config Files → Parameter Parsing → Model Creation → Training Loop
    ↓
Training Data → Forward Pass → Loss Computation → Backward Pass → Parameter Update
    ↓
Validation Data → Model Evaluation → Performance Monitoring → Early Stopping Decision
    ↓
Test Data → Final Evaluation → Result Saving → Visualization Output
```

### 🔄 Process Flow Details {#process-flow-details}

1. **Data Preprocessing Phase**:
   ```python
   # Data loading
   raw_data = torch.load(data_path)
   
   # Normalization
   normalized_data = normalize(raw_data)
   
   # Data augmentation
   if use_augmentation:
       augmented_data = apply_augmentation(normalized_data)
   ```

2. **Model Forward Pass**:
   ```python
   # Input processing
   x = input_data.to(device)
   
   # Attention computation
   attention_output = attention_layer(x)
   
   # Feature extraction
   features = feature_extractor(attention_output)
   
   # Output prediction
   predictions = output_layer(features)
   ```

3. **Loss Computation**:
   ```python
   # Base loss
   base_loss = criterion(predictions, targets)
   
   # SVD regularization
   svd_loss = compute_svd_regularization(model_weights)
   
   # Total loss
   total_loss = base_weight * base_loss + svd_weight * svd_loss
   ```

## Configuration System {#configuration-system}

### ⚙️ Configuration Hierarchy {#configuration-hierarchy}

```yaml
# Global configuration
global:
  seed: 42
  device: cuda:0
  deterministic: true

# Data configuration
data:
  path: "/path/to/data"
  batch_size: 128
  use_augmentation: true

# Model configuration
model:
  attention_type: self
  d_model: 256
  num_heads: 4

# Training configuration
training:
  epochs: 10
  learning_rate: 0.0001
  early_stop_patience: 10

# Attention test configuration
attention_test:
  types: [self, muse, ufo, ...]
```

### 🔧 Configuration Management Features {#configuration-management-features}

1. **Hierarchical Configuration**: Support for nested configuration structures
2. **Type Validation**: Automatic validation of configuration parameter types
3. **Default Value Handling**: Provide reasonable default configurations
4. **Environment Variable Support**: Support for environment variable overrides
5. **Configuration Inheritance**: Support for configuration file inheritance

## Experiment Management {#experiment-management}

### 🧪 Experiment Organization Structure {#experiment-organization-structure}

```
attention_results/
├── loss_config_0/                 # Loss configuration 0
│   ├── self/                      # Self-attention experiment
│   │   ├── model_best.pth         # Best model
│   │   ├── training_log.txt       # Training log
│   │   ├── config.yaml            # Experiment configuration
│   │   └── attention_vis.png      # Attention visualization
│   ├── muse/                      # MUSE attention experiment
│   └── ...
├── loss_config_1/                 # Loss configuration 1
├── train.log                      # Global training log
└── failed_attention_log.txt       # Failed experiment log
```

### 📊 Experiment Tracking {#experiment-tracking}

1. **Automated Experiments**: Batch execution of all configuration combinations
2. **Result Recording**: Detailed recording of each experiment's results
3. **Failure Handling**: Record and analyze failed experiments
4. **Progress Monitoring**: Real-time display of experiment progress
5. **Resource Management**: Intelligent management of GPU memory and computational resources

## Results Analysis {#results-analysis}

### 📈 Analysis Tools {#analysis-tools}

1. **TensorBoard Integration**:
   ```python
   # Start TensorBoard
   python start_tensorboard.py
   ```

2. **Attention Visualization**:
   ```python
   def visualize_attention(attention_weights, save_path):
       plt.figure(figsize=(10, 8))
       sns.heatmap(attention_weights, cmap='Blues')
       plt.savefig(save_path)
   ```

3. **Performance Comparison**:
   ```python
   def compare_attention_mechanisms(results_dir):
       # Load all experiment results
       results = load_all_results(results_dir)
       
       # Generate comparison charts
       plot_performance_comparison(results)
   ```

### 📊 Output Formats {#output-formats}

- **Training Curves**: Loss and accuracy over time
- **Attention Heatmaps**: Attention weight visualization
- **Performance Comparison Tables**: Performance comparison of different mechanisms
- **Statistical Reports**: Detailed experimental statistics

## Extensibility Design {#extensibility-design}

### 🔌 Plugin Architecture {#plugin-architecture}

1. **New Attention Mechanisms**:
   ```python
   # 1. Implement attention class
   class NewAttention(BaseAttention):
       def forward(self, x):
           # Implement attention logic
           pass
   
   # 2. Register to factory
   ModelFactory.register('new_attention', NewAttention)
   
   # 3. Add to configuration
   attention_test:
     types: [..., 'new_attention']
   ```

2. **New Loss Functions**:
   ```python
   class NewLoss(nn.Module):
       def forward(self, pred, target):
           # Implement loss logic
           pass
   ```

3. **New Data Formats**:
   ```python
   class NewDataset(Dataset):
       def __getitem__(self, idx):
           # Implement data loading logic
           pass
   ```

### 🚀 Performance Optimization {#performance-optimization}

1. **Memory Optimization**:
   - Gradient accumulation
   - Mixed precision training
   - Dynamic memory management

2. **Computation Optimization**:
   - Multi-GPU parallelism
   - Model parallelism
   - Data parallelism

3. **I/O Optimization**:
   - Asynchronous data loading
   - Memory mapping
   - Data prefetching

### 🔧 Maintainability Design {#maintainability-design}

1. **Code Quality**:
   - Type annotations
   - Documentation strings
   - Unit testing

2. **Error Handling**:
   - Exception catching
   - Error recovery
   - Log recording

3. **Version Control**:
   - Configuration versioning
   - Model versioning
   - Result versioning

---

**💡 Tip**: This architecture design supports rapid addition of new attention mechanisms and experiment configurations. For specific implementation details, please refer to the corresponding code modules.

## 📚 Related Documentation

- [Model Design](model-design)
- [Implementation Details](implementation-details)
- [Attention Mechanisms](attention-mechanisms-guide)

---

*Need help? Check the [FAQ](faq) or [Troubleshooting](troubleshooting) pages.*
