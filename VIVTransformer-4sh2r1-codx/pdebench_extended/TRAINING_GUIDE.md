# VIV Transformer Pressure Field Reconstruction Training Guide

## 🚀 Quick Start

### 1. One-Click Training Management System Launch

```bash
# Launch unified management interface
bash training_manager.sh
```

### 2. Quick Training

```bash
# Quick start training (recommended for beginners)
bash quick_start.sh

# Or use optimized version (recommended for servers)
bash run_training_optimized.sh
```

### 3. Monitoring System

```bash
# GPU monitoring
bash gpu_monitor.sh

# System monitoring
bash system_monitor.sh
```

## 📋 System Overview

### Server Configuration
- **Server**: node36
- **GPU**: 2x NVIDIA L40 (48GB VRAM)
- **Driver**: 535.183.01
- **CUDA**: 12.1

### Script Functions

| Script Name | Function Description | Use Case |
|-------------|---------------------|-----------|
| `training_manager.sh` | Unified training management interface | Main entry, integrates all functions |
| `quick_start.sh` | Quick training start | Beginners quick start |
| `run_training_optimized.sh` | Optimized training script | Server environment, high-performance training |
| `gpu_monitor.sh` | Real-time GPU monitoring | Monitor GPU status and processes |
| `system_monitor.sh` | System resource monitoring | Monitor CPU, memory, disk |

## 🎯 Training Mode Selection

### 1. Quick Start Mode
- **Suitable for**: Beginners, quick validation
- **Features**: Auto configuration, one-click start
- **Command**: `bash quick_start.sh`

### 2. Optimized Training Mode
- **Suitable for**: Server environment, long-term training
- **Features**: Performance optimization, resource monitoring
- **Command**: `bash run_training_optimized.sh`

### 3. Management Interface Mode
- **Suitable for**: Need full management functions
- **Features**: Graphical menu, full functionality
- **Command**: `bash training_manager.sh`

## 🔧 GPU Configuration Guide

### Automatic GPU Selection
The system will automatically detect GPU status and recommend the best choice:

```bash
# GPU status example
GPU 0: 18276MB/49140MB (37%), 0%, 45°C - Light load
GPU 1: 2MB/49140MB (0%), 0%, 42°C - Idle

Recommended: GPU 1 (Idle state)
```

### Manual GPU Setting

```bash
# Use GPU 0
export CUDA_VISIBLE_DEVICES=0

# Use GPU 1
export CUDA_VISIBLE_DEVICES=1

# Use dual GPU
export CUDA_VISIBLE_DEVICES=0,1
```

## 📊 Monitoring Functions

### GPU Monitoring

```bash
# Real-time monitoring
bash gpu_monitor.sh

# Simple status
nvidia-smi

# Show recommendations
bash gpu_monitor.sh --recommend
```

**Monitoring Content**:
- GPU usage and VRAM occupation
- Temperature and power consumption
- Running process details
- Usage recommendations

### System Monitoring

```bash
# Real-time monitoring
bash system_monitor.sh

# Simple status
htop

# Performance report
bash system_monitor.sh --report
```

**Monitoring Content**:
- CPU and memory usage
- Disk space and IO
- Network status
- Training process status

## 🗂️ File Structure

```
pdebench_extended/
├── configs/
│   └── pressure_field_training.yaml    # Training configuration file
├── data/
│   └── pressure_field_data.pt          # Training data
├── outputs/                            # Training output directory
│   ├── checkpoints/                    # Model checkpoints
│   ├── logs/                          # Training logs
│   └── visualizations/                # Visualization results
├── training_manager.sh                 # Main management script
├── quick_start.sh                      # Quick start
├── run_training_optimized.sh          # Optimized training
├── gpu_monitor.sh                      # GPU monitoring
├── system_monitor.sh                   # System monitoring
└── train_pressure_field.py            # Training main program
```

## ⚙️ Configuration Parameters

### Main Configuration File: `configs/pressure_field_training.yaml`

```yaml
# Key parameters
data:
  batch_size: 32              # Batch size
  input_dim: [20, 20]         # Input dimension
  output_dim: [200, 200]      # Output dimension

model:
  d_model: 512               # Model dimension
  nhead: 8                   # Attention heads
  num_layers: 6              # Number of layers

training:
  epochs: 100                # Training epochs
  learning_rate: 0.001       # Learning rate
  device: "auto"             # Device selection
```

### Performance Optimization Parameters

```bash
# Environment variables
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export CUDA_LAUNCH_BLOCKING=0

# Training parameters
mixed_precision: true
gradient_accumulation_steps: 4
dataloader_num_workers: 4
pin_memory: true
```

## 🚨 Troubleshooting

### Common Issues

#### 1. GPU Memory Insufficient

```bash
# Error message
RuntimeError: CUDA out of memory

# Solutions
# 1. Reduce batch_size
# 2. Use gradient accumulation
# 3. Enable mixed precision training
```

#### 2. Data File Not Found

```bash
# Error message
FileNotFoundError: data file not found

# Solution
# Check data file path
ls -la data/

# Regenerate data (if needed)
python generate_data.py
```

#### 3. Training Process Stuck

```bash
# Check processes
ps aux | grep python

# Terminate process
kill <PID>

# Or use management interface
bash training_manager.sh
# Select "Manage Training Tasks" -> "Stop Training Process"
```

#### 4. TensorBoard Inaccessible

```bash
# Check TensorBoard process
ps aux | grep tensorboard

# Restart TensorBoard
tensorboard --logdir=outputs/logs --port=6006 &

# Access address
http://<server_IP>:6006
```

### Performance Optimization Recommendations

#### 1. GPU Selection Strategy
- **Single GPU training**: Choose GPU with least VRAM usage
- **Dual GPU training**: Use DataParallel or DistributedDataParallel
- **Mixed precision**: Enable AMP to reduce VRAM usage

#### 2. Data Loading Optimization

```python
# Optimization parameters
num_workers = 4          # Data loading processes
pin_memory = True        # Pin memory
prefetch_factor = 2      # Prefetch factor
```

#### 3. Training Strategy
- **Learning rate scheduling**: Use CosineAnnealingLR
- **Early stopping**: Prevent overfitting
- **Gradient clipping**: Prevent gradient explosion

## 📈 Monitoring and Logs

### TensorBoard Visualization

```bash
# Launch TensorBoard
tensorboard --logdir=outputs/logs --port=6006 &

# Access address
http://<server_IP>:6006
```

### Log Files
- **Training logs**: `outputs/logs/training.log`
- **System logs**: `system_monitor.log`
- **Error logs**: `nohup.out`

### Checkpoint Management

```bash
# Checkpoint directory
ls outputs/checkpoints/

# Resume training
python train_pressure_field.py --resume outputs/checkpoints/latest.pth
```

## 🔄 Workflow

### Standard Training Process
1. **Environment check**: Verify GPU, Python, PyTorch
2. **Data preparation**: Confirm data files exist
3. **Configuration setup**: Adjust training parameters
4. **Start training**: Choose appropriate training mode
5. **Monitor training**: Use monitoring scripts to observe status
6. **Result analysis**: Check TensorBoard and logs

### Best Practices
1. **Before training**: Use `training_manager.sh` to check system status
2. **During training**: Regularly check GPU and system monitoring
3. **After training**: Backup important checkpoints and logs

## 📞 Technical Support

### Quick Diagnosis

```bash
# System status check
bash training_manager.sh
# Select "System Configuration" to view details

# Generate system report
bash training_manager.sh
# Select "Toolbox" -> "Generate System Report"
```

### Contact Information
If you encounter issues, please provide:
1. Error message screenshots
2. System report file
3. Training configuration file
4. Related log files

---

**Happy training!** 🎉