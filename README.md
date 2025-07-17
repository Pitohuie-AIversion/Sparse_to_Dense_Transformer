# VIVTransformer: Multi-Attention Mechanism Research Platform

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9%2B-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A comprehensive research platform for training and evaluating VIVTransformer models with **38+ different attention mechanisms**. This repository enables systematic comparison of attention mechanisms across multiple loss configurations for deep learning research.

## 🌟 Key Features

- **38+ Attention Mechanisms**: Including Self-Attention, MUSE, UFO, Relative, Sparse, LSH, EMSA, MobileViT, DAT, CrossFormer, MOA, and many more
- **Multiple Loss Configurations**: Support for 50 different loss configurations with SVD-based regularization
- **Automated Experimentation**: Batch processing of all attention-loss combinations
- **Comprehensive Logging**: Detailed training logs, visualization, and result tracking
- **GPU Acceleration**: CUDA support with memory management and DataParallel
- **Reproducible Results**: Deterministic training with seed control
- **TensorBoard Integration**: Real-time training monitoring and visualization

## 📋 Supported Attention Mechanisms

| Category | Mechanisms |
|----------|------------|
| **Basic** | Self, Simplified Self |
| **Efficient** | MUSE, UFO, Sparse, LSH |
| **Positional** | Relative, Axial |
| **Mobile** | MobileViT, MobileViTv2 |
| **Advanced** | EMSA, DAT, CrossFormer, MOA, CrissCross |
| **Channel** | SE, SK, CBAM, BAM, ECA |
| **Spatial** | PSA, DANet, CoT, Polarized |
| **Hybrid** | CoAtNet, Halo, A2, ParNet, External, AFT |
| **Others** | GFNet, Shuffle, Residual, S2, Triplet, Coord, Outlook, VIP |

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended)
- 8GB+ RAM

### Installation

1. **Clone the repository**:
```bash
git clone https://github.com/yourusername/VIVTransformer.git
cd VIVTransformer
```

2. **Install dependencies**:
```bash
pip install -r modify_multi_attention/requirements.txt
```

3. **Verify installation**:
```bash
python -m modify_multi_attention.main --help
```

### Basic Usage

1. **Run with default configuration**:
```bash
python -m modify_multi_attention.main
```

2. **Run specific loss configuration**:
```bash
python -m modify_multi_attention.main --loss_idx 0
```

3. **Custom configuration and output directory**:
```bash
python -m modify_multi_attention.main --config custom_config.yaml --results-dir ./my_results
```

4. **Start TensorBoard monitoring**:
```bash
python modify_multi_attention/start_tensorboard.py
```

## ⚙️ Configuration

### Main Configuration (`configs/config.yaml`)

```yaml
# Global settings
global:
  seed: 42
  deterministic: true
  device: cuda:0
  max_memory_fraction: 0.8

# Model settings
model:
  d_model: 256
  num_heads: 4
  num_layers: 6
  input_dim: 400
  output_dim: 40000
  seq_len: 49

# Training settings
training:
  epochs: 10
  learning_rate: 0.0001
  early_stop_patience: 10

# Attention mechanisms to test
attention_test:
  types: [self, muse, ufo, relative, sparse, ...]
```

### Loss Configurations

The system supports 50 different loss configurations located in `configs/loss_configs/`. Each configuration defines:

- **Base Weight**: Primary loss component weight
- **SVD Weights**: Singular Value Decomposition regularization weights
- **TopK**: Number of top singular values to consider

## 📊 Results and Analysis

### Output Structure

```
attention_results/
├── loss_config_0/
│   ├── self/
│   │   ├── model_best.pth
│   │   ├── training_log.txt
│   │   └── attention_visualization.png
│   └── muse/
├── loss_config_1/
├── train.log
└── failed_attention_log.txt
```

### Key Metrics

- **Training Loss**: Primary objective function value
- **Validation Loss**: Model generalization performance
- **Test Loss**: Final evaluation metric
- **SVD Regularization**: Structural constraint satisfaction

## 🔧 Advanced Usage

### Custom Attention Mechanism

1. Add your attention class to `mymodels/attention/`
2. Register it in `mymodels/model_factory.py`
3. Add the name to `attention_test.types` in config

### Batch Processing

```bash
# Run all configurations
bash modify_multi_attention/run_all_loss.sh

# Generate loss configurations
python generate_loss_configs_topk10.py
```

### Memory Optimization

```yaml
global:
  max_memory_fraction: 0.6  # Reduce GPU memory usage
  use_dataparallel: true    # Enable multi-GPU training
```

## 📈 Monitoring and Visualization

### TensorBoard

```bash
python modify_multi_attention/start_tensorboard.py
# Open http://localhost:6006 in browser
```

### Attention Visualization

Attention maps are automatically generated and saved during training when `visualization.enabled: true`.

## 🛠️ Development

### Code Quality

```bash
# Format code
black modify_multi_attention/

# Lint code
flake8 modify_multi_attention/
```

### Testing

```bash
python modify_multi_attention/test_attention_test.py
```

### Repository Duplication

```bash
python duplicate_repo.py /path/to/destination
```

## 📚 Documentation

For detailed documentation, please visit our [Wiki](../../wiki) which includes:

- **Architecture Overview**: Detailed model architecture explanation
- **Attention Mechanisms Guide**: In-depth analysis of each attention type
- **Loss Functions**: Mathematical formulation and implementation details
- **Performance Benchmarks**: Comparative analysis results
- **Troubleshooting**: Common issues and solutions

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [fightingcv_attention](https://github.com/xmu-xiaoma666/External-Attention-pytorch) for attention mechanism implementations
- PyTorch team for the deep learning framework
- Research community for attention mechanism innovations

## 📞 Contact

For questions and support:
- Create an [Issue](../../issues)
- Check our [Wiki](../../wiki)
- Email: [your-email@domain.com]

---

**Star ⭐ this repository if you find it helpful for your research!**

