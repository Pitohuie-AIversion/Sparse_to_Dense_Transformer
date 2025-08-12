#!/usr/bin/env python3
"""
Example: VIVTransformer Basic Usage

This example demonstrates how to use VIVTransformer for vortex-induced vibration prediction.
Includes a complete workflow from data preparation, model training to result visualization.

Author: [Your Name]
Date: 2024-01-01
Version: 1.0.0
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple

# VIVTransformer related imports
from vivtransformer import VIVTransformer
from vivtransformer.data import VIVDataset, VIVDataLoader
from vivtransformer.losses import SVDEnhancedLoss
from vivtransformer.utils import set_seed, get_device
from vivtransformer.visualization import plot_attention_weights, plot_predictions


def main():
    """Main function"""
    # Set random seed
    set_seed(42)
    
    # Get device
    device = get_device()
    print(f"Using device: {device}")
    
    # 1. Data preparation
    print("\n=== Data Preparation ===")
    train_dataset, val_dataset = prepare_data()
    
    train_loader = VIVDataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4
    )
    
    val_loader = VIVDataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4
    )
    
    print(f"Training set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")
    
    # 2. Model creation
    print("\n=== Model Creation ===")
    model = create_model(device)
    print(f"Model parameters: {count_parameters(model):,}")
    
    # 3. Training configuration
    print("\n=== Training Configuration ===")
    criterion = SVDEnhancedLoss(
        reconstruction_weight=1.0,
        singular_value_weight=0.5,
        orthogonality_weight=0.3
    )
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=1e-5
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=100,
        eta_min=1e-6
    )
    
    # 4. Model training
    print("\n=== Starting Training ===")
    train_losses, val_losses = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        epochs=100,
        device=device
    )
    
    # 5. Result evaluation
    print("\n=== Result Evaluation ===")
    evaluate_model(model, val_loader, device)
    
    # 6. Visualization
    print("\n=== Result Visualization ===")
    visualize_results(model, val_loader, device)
    
    # 7. Model saving
    print("\n=== Saving Model ===")
    save_model(model, 'vivtransformer_trained.pth')
    
    print("\nTraining completed!")


def prepare_data() -> Tuple[VIVDataset, VIVDataset]:
    """Prepare training and validation data"""
    # Generate sample data (replace with real data in actual use)
    np.random.seed(42)
    
    # Simulate cylinder flow data
    n_samples = 1000
    seq_len = 128
    n_features = 64
    
    # Input features: velocity field, pressure field, etc.
    X = np.random.randn(n_samples, seq_len, n_features).astype(np.float32)
    
    # Targets: displacement, velocity, etc.
    y = np.random.randn(n_samples, seq_len, 2).astype(np.float32)
    
    # Add some physical correlation
    for i in range(n_samples):
        # Simulate periodic vibration
        t = np.linspace(0, 10, seq_len)
        frequency = np.random.uniform(0.1, 2.0)
        amplitude = np.random.uniform(0.5, 2.0)
        
        y[i, :, 0] = amplitude * np.sin(2 * np.pi * frequency * t)
        y[i, :, 1] = amplitude * np.cos(2 * np.pi * frequency * t)
    
    # Split dataset
    split_idx = int(0.8 * n_samples)
    
    train_dataset = VIVDataset(
        X[:split_idx],
        y[:split_idx],
        transform=None
    )
    
    val_dataset = VIVDataset(
        X[split_idx:],
        y[split_idx:],
        transform=None
    )
    
    return train_dataset, val_dataset


def create_model(device: torch.device) -> VIVTransformer:
    """Create VIVTransformer model"""
    model = VIVTransformer(
        input_dim=64,
        output_dim=2,
        d_model=512,
        n_heads=8,
        n_layers=6,
        attention_type='external',
        dropout=0.1,
        max_seq_len=128
    )
    
    return model.to(device)


def count_parameters(model: nn.Module) -> int:
    """Count model parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_model(
    model: nn.Module,
    train_loader: VIVDataLoader,
    val_loader: VIVDataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    epochs: int,
    device: torch.device
) -> Tuple[List[float], List[float]]:
    """Train model"""
    train_losses = []
    val_losses = []
    
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for batch_idx, batch in enumerate(train_loader):
            inputs = batch['input'].to(device)
            targets = batch['target'].to(device)
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.6f}")
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                inputs = batch['input'].to(device)
                targets = batch['target'].to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item()
        
        # Calculate average losses
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        # Learning rate scheduling
        scheduler.step()
        
        print(f"Epoch {epoch+1}/{epochs}:")
        print(f"  Training loss: {avg_train_loss:.6f}")
        print(f"  Validation loss: {avg_val_loss:.6f}")
        print(f"  Learning rate: {scheduler.get_last_lr()[0]:.8f}")
        
        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break
    
    return train_losses, val_losses


def evaluate_model(
    model: nn.Module,
    val_loader: VIVDataLoader,
    device: torch.device
) -> Dict[str, float]:
    """Evaluate model performance"""
    model.eval()
    
    total_mse = 0.0
    total_mae = 0.0
    total_samples = 0
    
    predictions = []
    targets = []
    
    with torch.no_grad():
        for batch in val_loader:
            inputs = batch['input'].to(device)
            batch_targets = batch['target'].to(device)
            
            outputs = model(inputs)
            
            # Calculate metrics
            mse = torch.mean((outputs - batch_targets) ** 2)
            mae = torch.mean(torch.abs(outputs - batch_targets))
            
            total_mse += mse.item() * inputs.size(0)
            total_mae += mae.item() * inputs.size(0)
            total_samples += inputs.size(0)
            
            # Collect predictions
            predictions.append(outputs.cpu().numpy())
            targets.append(batch_targets.cpu().numpy())
    
    # Calculate average metrics
    avg_mse = total_mse / total_samples
    avg_mae = total_mae / total_samples
    avg_rmse = np.sqrt(avg_mse)
    
    # Calculate R²
    predictions = np.concatenate(predictions, axis=0)
    targets = np.concatenate(targets, axis=0)
    
    ss_res = np.sum((targets - predictions) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    r2_score = 1 - (ss_res / ss_tot)
    
    metrics = {
        'MSE': avg_mse,
        'MAE': avg_mae,
        'RMSE': avg_rmse,
        'R²': r2_score
    }
    
    print("Evaluation results:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.6f}")
    
    return metrics


def visualize_results(
    model: nn.Module,
    val_loader: VIVDataLoader,
    device: torch.device
) -> None:
    """Visualize results"""
    model.eval()
    
    # Get a batch of data
    batch = next(iter(val_loader))
    inputs = batch['input'].to(device)
    targets = batch['target'].to(device)
    
    with torch.no_grad():
        outputs = model(inputs)
        
        # If model returns attention weights
        if hasattr(model, 'get_attention_weights'):
            attention_weights = model.get_attention_weights()
    
    # Convert to numpy
    inputs_np = inputs.cpu().numpy()
    outputs_np = outputs.cpu().numpy()
    targets_np = targets.cpu().numpy()
    
    # Create figures
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot predictions vs true values
    sample_idx = 0
    time_steps = np.arange(outputs_np.shape[1])
    
    # X direction displacement
    axes[0, 0].plot(time_steps, targets_np[sample_idx, :, 0], 'b-', label='True values', linewidth=2)
    axes[0, 0].plot(time_steps, outputs_np[sample_idx, :, 0], 'r--', label='Predictions', linewidth=2)
    axes[0, 0].set_title('X Direction Displacement Prediction')
    axes[0, 0].set_xlabel('Time Step')
    axes[0, 0].set_ylabel('Displacement')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Y direction displacement
    axes[0, 1].plot(time_steps, targets_np[sample_idx, :, 1], 'b-', label='True values', linewidth=2)
    axes[0, 1].plot(time_steps, outputs_np[sample_idx, :, 1], 'r--', label='Predictions', linewidth=2)
    axes[0, 1].set_title('Y Direction Displacement Prediction')
    axes[0, 1].set_xlabel('Time Step')
    axes[0, 1].set_ylabel('Displacement')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Error analysis
    error_x = np.abs(targets_np[sample_idx, :, 0] - outputs_np[sample_idx, :, 0])
    error_y = np.abs(targets_np[sample_idx, :, 1] - outputs_np[sample_idx, :, 1])
    
    axes[1, 0].plot(time_steps, error_x, 'g-', label='X Direction Error', linewidth=2)
    axes[1, 0].plot(time_steps, error_y, 'm-', label='Y Direction Error', linewidth=2)
    axes[1, 0].set_title('Prediction Error')
    axes[1, 0].set_xlabel('Time Step')
    axes[1, 0].set_ylabel('Absolute Error')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Phase diagram
    axes[1, 1].plot(targets_np[sample_idx, :, 0], targets_np[sample_idx, :, 1], 'b-', label='True Trajectory', linewidth=2)
    axes[1, 1].plot(outputs_np[sample_idx, :, 0], outputs_np[sample_idx, :, 1], 'r--', label='Predicted Trajectory', linewidth=2)
    axes[1, 1].set_title('Vibration Trajectory')
    axes[1, 1].set_xlabel('X Displacement')
    axes[1, 1].set_ylabel('Y Displacement')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    axes[1, 1].axis('equal')
    
    plt.tight_layout()
    plt.savefig('prediction_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # If attention weights are available, plot attention map
    if 'attention_weights' in locals():
        plot_attention_weights(attention_weights, save_path='attention_weights.png')


def save_model(model: nn.Module, path: str) -> None:
    """Save model"""
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': model.get_config(),
        'timestamp': torch.tensor(time.time())
    }, path)
    
    print(f"Model saved to: {path}")


if __name__ == '__main__':
    main()



    float* output,
    int batch_size,
    int seq_len,
    int d_model
) {
    // Your optimized CUDA kernel function
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < batch_size * seq_len * d_model) {
        // Computation logic
    }
}

torch::Tensor optimized_attention_cuda(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value
) {
    // CUDA function wrapper
    auto output = torch::zeros_like(query);
    
    const int threads = 256;
    const int blocks = (query.numel() + threads - 1) / threads;
    
    optimized_attention_kernel<<<blocks, threads>>>(
        query.data_ptr<float>(),
        key.data_ptr<float>(),
        value.data_ptr<float>(),
        output.data_ptr<float>(),
        query.size(0),
        query.size(1),
        query.size(2)
    );
    
    return output;
}
```

#### 2. Memory Optimization

```python
# vivtransformer/utils/memory_optimization.py

import torch
from typing import Iterator, Tuple

class GradientCheckpointing:
    """Gradient checkpointing optimization"""
    
    @staticmethod
    def checkpoint_sequential(
        functions: Iterator[torch.nn.Module],
        segments: int,
        input: torch.Tensor
    ) -> torch.Tensor:
        """Sequential gradient checkpointing"""
        def run_function(start, end, functions):
            def forward(input):
                for j in range(start, end + 1):
                    input = functions[j](input)
                return input
            return forward
        
        if segments == 1:
            return torch.utils.checkpoint.checkpoint(
                run_function(0, len(functions) - 1, functions),
                input
            )
        
        # Segmented checkpointing
        segment_size = len(functions) // segments
        for i in range(segments):
            start_idx = i * segment_size
            end_idx = (i + 1) * segment_size - 1
            if i == segments - 1:
                end_idx = len(functions) - 1
            
            input = torch.utils.checkpoint.checkpoint(
                run_function(start_idx, end_idx, functions),
                input
            )
        
        return input
```

---

## 📚 Documentation Contribution

### Documentation Types

#### 1. API Documentation

**Function Documentation Template**:
```python
def complex_function(
    param1: torch.Tensor,
    param2: Optional[str] = None,
    param3: Dict[str, Any] = None
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Detailed documentation for complex functions
    
    This function performs complex computations including multiple steps and conditional logic.
    Suitable for scenarios requiring precise control over the computation process.
    
    Args:
        param1: Input tensor with shape [batch_size, seq_len, d_model]
            - batch_size: Batch size, typically 32-512
            - seq_len: Sequence length, supports variable length
            - d_model: Model dimension, must be multiple of 8
        param2: Optional string parameter, supports the following values:
            - 'mode1': Use standard computation mode
            - 'mode2': Use optimized computation mode
            - None: Automatically select mode
        param3: Configuration dictionary containing the following keys:
            - 'threshold': float, threshold parameter (default: 0.5)
            - 'iterations': int, number of iterations (default: 10)
            - 'verbose': bool, whether to output detailed information (default: False)
    
    Returns:
        result: Result tensor with the same shape as param1
        metrics: Computation metrics dictionary containing:
            - 'computation_time': Computation time (seconds)
            - 'memory_usage': Memory usage (MB)
            - 'convergence_score': Convergence score (0-1)
    
    Raises:
        ValueError: When param1 has incorrect dimensions
        RuntimeError: When numerical instability occurs during computation
        MemoryError: When memory is insufficient
    
    Example:
        Basic usage:
        
        >>> import torch
        >>> input_tensor = torch.randn(32, 128, 512)
        >>> result, metrics = complex_function(input_tensor)
        >>> print(f"Computation time: {metrics['computation_time']:.3f}s")
        
        Advanced configuration:
        
        >>> config = {
        ...     'threshold': 0.8,
        ...     'iterations': 20,
        ...     'verbose': True
        ... }
        >>> result, metrics = complex_function(
        ...     input_tensor,
        ...     param2='mode2',
        ...     param3=config
        ... )
    
    Note:
        - This function performs best when running on GPU
        - For large inputs, gradient checkpointing is recommended
        - Supports mixed precision training
    
    See Also:
        - :func:`related_function`: Related function
        - :class:`RelatedClass`: Related class
        - :doc:`../tutorials/advanced_usage`: Advanced usage tutorial
    
    References:
        [1] Author et al. "Paper Title". Journal Name, 2024.
        [2] https://example.com/documentation
    """
```

#### 2. Tutorial Documentation

**Tutorial Structure Template**:
```markdown
# Tutorial Title

> Brief description of tutorial content and target audience

## Learning Objectives

After completing this tutorial, you will be able to:
- [ ] Objective 1
- [ ] Objective 2
- [ ] Objective 3

## Prerequisites

- Python 3.8+
- PyTorch 1.9+
- Basic knowledge of deep learning

## Step 1: Environment Setup

### Install Dependencies

```bash
pip install vivtransformer
```

### Verify Installation

```python
import vivtransformer
print(f"VIVTransformer version: {vivtransformer.__version__}")
```

## Step 2: Data Preparation

### Data Format

```python
# Data should be in the following format
data = {
    'input': torch.tensor(...),  # [batch_size, seq_len, features]
    'target': torch.tensor(...), # [batch_size, seq_len, targets]
    'metadata': {...}            # Metadata dictionary
}
```

### Data Loading

```python
from vivtransformer.data import VIVDataLoader

# Create data loader
loader = VIVDataLoader(
    data_path='path/to/data',
    batch_size=32,
    shuffle=True
)

# Iterate over data
for batch in loader:
    input_data = batch['input']
    target_data = batch['target']
    # Process data...
```

## Step 3: Model Configuration

### Basic Configuration

```python
from vivtransformer import VIVTransformer

# Create model
model = VIVTransformer(
    d_model=512,
    n_heads=8,
    n_layers=6,
    attention_type='external'
)
```

### Advanced Configuration

```yaml
# config.yaml
model:
  d_model: 512
  n_heads: 8
  n_layers: 6
  attention_type: 'external'
  
loss:
  type: 'svd_enhanced'
  weights:
    reconstruction: 1.0
    singular_value: 0.5
    orthogonality: 0.3

training:
  learning_rate: 1e-4
  batch_size: 32
  epochs: 100
```

```python
from vivtransformer.config import load_config

# Load from configuration file
config = load_config('config.yaml')
model = VIVTransformer.from_config(config.model)
```

## Common Issues

### Q: How to handle memory issues?

A: You can try the following methods:
1. Reduce batch_size
2. Use gradient accumulation
3. Enable gradient checkpointing

```python
# Gradient accumulation example
accumulation_steps = 4
for i, batch in enumerate(dataloader):
    loss = model(batch)
    loss = loss / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### Q: How to choose attention mechanism?

A: Characteristics of different attention mechanisms:
- `scaled_dot_product`: Standard attention, computationally efficient
- `external`: External attention, suitable for long sequences
- `se_attention`: Channel attention, suitable for feature selection

## Summary

This tutorial introduced...

## Next Steps

- [ ] Read advanced configuration tutorial
- [ ] Try custom attention mechanisms
- [ ] Participate in community discussions
```

#### 3. Example Code

**Complete Example Template**:
```python
#!/usr/bin/env python3
"""
Example: VIVTransformer Basic Usage

This example demonstrates how to use VIVTransformer for vortex-induced vibration prediction.
Includes a complete workflow from data preparation, model training to result visualization.

Author: [Your Name]
Date: 2024-01-01
Version: 1.0.0
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple

# VIVTransformer related imports
from vivtransformer import VIVTransformer
from vivtransformer.data import VIVDataset, VIVDataLoader
from vivtransformer.losses import SVDEnhancedLoss
from vivtransformer.utils import set_seed, get_device
from vivtransformer.visualization import plot_attention_weights, plot_predictions


def main():
    """Main function"""
    # Set random seed
    set_seed(42)
    
    # Get device
    device = get_device()
    print(f"Using device: {device}")
    
    # 1. Data preparation
    print("\n=== Data Preparation ===")
    train_dataset, val_dataset = prepare_data()
    
    train_loader = VIVDataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4
    )
    
    val_loader = VIVDataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4
    )
    
    print(f"Training set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")
    
    # 2. Model creation
    print("\n=== Model Creation ===")
    model = create_model(device)
    print(f"Model parameters: {count_parameters(model):,}")
    
    # 3. Training configuration
    print("\n=== Training Configuration ===")
    criterion = SVDEnhancedLoss(
        reconstruction_weight=1.0,
        singular_value_weight=0.5,
        orthogonality_weight=0.3
    )
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=1e-5
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=100,
        eta_min=1e-6
    )
    
    # 4. Model training
    print("\n=== Starting Training ===")
    train_losses, val_losses = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        epochs=100,
        device=device
    )
    
    # 5. Result evaluation
    print("\n=== Result Evaluation ===")
    evaluate_model(model, val_loader, device)
    
    # 6. Visualization
    print("\n=== Result Visualization ===")
    visualize_results(model, val_loader, device)
    
    # 7. Model saving
    print("\n=== Saving Model ===")
    save_model(model, 'vivtransformer_trained.pth')
    
    print("\nTraining completed!")


def prepare_data() -> Tuple[VIVDataset, VIVDataset]:
    """Prepare training and validation data"""
    # Generate sample data (replace with real data in actual use)
    np.random.seed(42)
    
    # Simulate cylinder flow data
    n_samples = 1000
    seq_len = 128
    n_features = 64
    
    # Input features: velocity field, pressure field, etc.
    X = np.random.randn(n_samples, seq_len, n_features).astype(np.float32)
    
    # Targets: displacement, velocity, etc.
    y = np.random.randn(n_samples, seq_len, 2).astype(np.float32)
    
    # Add some physical correlation
    for i in range(n_samples):
        # Simulate periodic vibration
        t = np.linspace(0, 10, seq_len)
        frequency = np.random.uniform(0.1, 2.0)
        amplitude = np.random.uniform(0.5, 2.0)
        
        y[i, :, 0] = amplitude * np.sin(2 * np.pi * frequency * t)
        y[i, :, 1] = amplitude * np.cos(2 * np.pi * frequency * t)
    
    # Split dataset
    split_idx = int(0.8 * n_samples)
    
    train_dataset = VIVDataset(
        X[:split_idx],
        y[:split_idx],
        transform=None
    )
    
    val_dataset = VIVDataset(
        X[split_idx:],
        y[split_idx:],
        transform=None
    )
    
    return train_dataset, val_dataset


def create_model(device: torch.device) -> VIVTransformer:
    """Create VIVTransformer model"""
    model = VIVTransformer(
        input_dim=64,
        output_dim=2,
        d_model=512,
        n_heads=8,
        n_layers=6,
        attention_type='external',
        dropout=0.1,
        max_seq_len=128
    )
    
    return model.to(device)


def count_parameters(model: nn.Module) -> int:
    """Count model parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_model(
    model: nn.Module,
    train_loader: VIVDataLoader,
    val_loader: VIVDataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    epochs: int,
    device: torch.device
) -> Tuple[List[float], List[float]]:
    """Train model"""
    train_losses = []
    val_losses = []
    
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for batch_idx, batch in enumerate(train_loader):
            inputs = batch['input'].to(device)
            targets = batch['target'].to(device)
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.6f}")
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                inputs = batch['input'].to(device)
                targets = batch['target'].to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item()
        
        # Calculate average losses
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        # Learning rate scheduling
        scheduler.step()
        
        print(f"Epoch {epoch+1}/{epochs}:")
        print(f"  Training loss: {avg_train_loss:.6f}")
        print(f"  Validation loss: {avg_val_loss:.6f}")
        print(f"  Learning rate: {scheduler.get_last_lr()[0]:.8f}")
        
        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break
    
    return train_losses, val_losses


def evaluate_model(
    model: nn.Module,
    val_loader: VIVDataLoader,
    device: torch.device
) -> Dict[str, float]:
    """Evaluate model performance"""
    model.eval()
    
    total_mse = 0.0
    total_mae = 0.0
    total_samples = 0
    
    predictions = []
    targets = []
    
    with torch.no_grad():
        for batch in val_loader:
            inputs = batch['input'].to(device)
            batch_targets = batch['target'].to(device)
            
            outputs = model(inputs)
            
            # Calculate metrics
            mse = torch.mean((outputs - batch_targets) ** 2)
            mae = torch.mean(torch.abs(outputs - batch_targets))
            
            total_mse += mse.item() * inputs.size(0)
            total_mae += mae.item() * inputs.size(0)
            total_samples += inputs.size(0)
            
            # Collect predictions
            predictions.append(outputs.cpu().numpy())
            targets.append(batch_targets.cpu().numpy())
    
    # Calculate average metrics
    avg_mse = total_mse / total_samples
    avg_mae = total_mae / total_samples
    avg_rmse = np.sqrt(avg_mse)
    
    # Calculate R²
    predictions = np.concatenate(predictions, axis=0)
    targets = np.concatenate(targets, axis=0)
    
    ss_res = np.sum((targets - predictions) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    r2_score = 1 - (ss_res / ss_tot)
    
    metrics = {
        'MSE': avg_mse,
        'MAE': avg_mae,
        'RMSE': avg_rmse,
        'R²': r2_score
    }
    
    print("Evaluation results:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.6f}")
    
    return metrics


def visualize_results(
    model: nn.Module,
    val_loader: VIVDataLoader,
    device: torch.device
) -> None:
    """Visualize results"""
    model.eval()
    
    # Get a batch of data
    batch = next(iter(val_loader))
    inputs = batch['input'].to(device)
    targets = batch['target'].to(device)
    
    with torch.no_grad():
        outputs = model(inputs)
        
        # If model returns attention weights
        if hasattr(model, 'get_attention_weights'):
            attention_weights = model.get_attention_weights()
    
    # Convert to numpy
    inputs_np = inputs.cpu().numpy()
    outputs_np = outputs.cpu().numpy()
    targets_np = targets.cpu().numpy()
    
    # Create figures
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot predictions vs true values
    sample_idx = 0
    time_steps = np.arange(outputs_np.shape[1])
    
    # X direction displacement
    axes[0, 0].plot(time_steps, targets_np[sample_idx, :, 0], 'b-', label='True values', linewidth=2)
    axes[0, 0].plot(time_steps, outputs_np[sample_idx, :, 0], 'r--', label='Predictions', linewidth=2)
    axes[0, 0].set_title('X Direction Displacement Prediction')
    axes[0, 0].set_xlabel('Time Step')
    axes[0, 0].set_ylabel('Displacement')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Y direction displacement
    axes[0, 1].plot(time_steps, targets_np[sample_idx, :, 1], 'b-', label='True values', linewidth=2)
    axes[0, 1].plot(time_steps, outputs_np[sample_idx, :, 1], 'r--', label='Predictions', linewidth=2)
    axes[0, 1].set_title('Y Direction Displacement Prediction')
    axes[0, 1].set_xlabel('Time Step')
    axes[0, 1].set_ylabel('Displacement')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Error analysis
    error_x = np.abs(targets_np[sample_idx, :, 0] - outputs_np[sample_idx, :, 0])
    error_y = np.abs(targets_np[sample_idx, :, 1] - outputs_np[sample_idx, :, 1])
    
    axes[1, 0].plot(time_steps, error_x, 'g-', label='X Direction Error', linewidth=2)
    axes[1, 0].plot(time_steps, error_y, 'm-', label='Y Direction Error', linewidth=2)
    axes[1, 0].set_title('Prediction Error')
    axes[1, 0].set_xlabel('Time Step')
    axes[1, 0].set_ylabel('Absolute Error')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Phase diagram
    axes[1, 1].plot(targets_np[sample_idx, :, 0], targets_np[sample_idx, :, 1], 'b-', label='True Trajectory', linewidth=2)
    axes[1, 1].plot(outputs_np[sample_idx, :, 0], outputs_np[sample_idx, :, 1], 'r--', label='Predicted Trajectory', linewidth=2)
    axes[1, 1].set_title('Vibration Trajectory')
    axes[1, 1].set_xlabel('X Displacement')
    axes[1, 1].set_ylabel('Y Displacement')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    axes[1, 1].axis('equal')
    
    plt.tight_layout()
    plt.savefig('prediction_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # If attention weights are available, plot attention map
    if 'attention_weights' in locals():
        plot_attention_weights(attention_weights, save_path='attention_weights.png')


def save_model(model: nn.Module, path: str) -> None:
    """Save model"""
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': model.get_config(),
        'timestamp': torch.tensor(time.time())
    }, path)
    
    print(f"Model saved to: {path}")


if __name__ == '__main__':
    main()


# 🤝 Community Contribution Guide

> Welcome to the VIVTransformer open source community! This guide will help you understand how to participate in project contributions

---

## 📋 Table of Contents

- [🌟 Welcome to Join](#-welcome-to-join)
- [🎯 Contribution Types](#-contribution-types)
- [📝 Code Contribution](#-code-contribution)
- [📚 Documentation Contribution](#-documentation-contribution)
- [🐛 Issue Reporting](#-issue-reporting)
- [💡 Feature Suggestions](#-feature-suggestions)
- [🧪 Testing Contribution](#-testing-contribution)
- [🎨 Design Contribution](#-design-contribution)
- [📊 Data Contribution](#-data-contribution)
- [🏆 Contributor Recognition](#-contributor-recognition)
- [📞 Contact Us](#-contact-us)

---

## 🌟 Welcome to Join

### Project Vision

VIVTransformer is committed to becoming **the open source benchmark in the field of vortex-induced vibration analysis**, providing powerful tools for scientific research and engineering applications by combining deep learning with physical modeling.

### Community Values

- **🔬 Scientific Rigor**: Based on solid theoretical foundations and experimental validation
- **🤝 Open Collaboration**: Welcome contributors from different backgrounds
- **📈 Continuous Improvement**: Continuously optimize performance and user experience
- **🎓 Knowledge Sharing**: Promote academic exchange and technology dissemination
- **🌍 Inclusive Diversity**: Respect different viewpoints and cultural backgrounds

### Contributor Types

| Contributor Type | Skill Requirements | Contribution Content | Time Investment |
|------------------|-------------------|---------------------|------------------|
| **Core Developer** | Deep learning, fluid mechanics | Core algorithms, architecture design | 20+ hours/week |
| **Feature Developer** | Python, machine learning | New features, tool development | 10-20 hours/week |
| **Documentation Maintainer** | Technical writing, teaching | Documentation, tutorial writing | 5-15 hours/week |
| **Test Engineer** | Software testing, quality assurance | Test cases, performance testing | 5-10 hours/week |
| **Community Manager** | Communication coordination, project management | Community operations, event organization | 5-10 hours/week |
| **Occasional Contributor** | Any skills | Issue reporting, small fixes | 1-5 hours/week |

---

## 🎯 Contribution Types

### Quick Start

#### 1. Environment Setup

```bash
# 1. Fork the project to your GitHub account
# 2. Clone your fork
git clone https://github.com/your-username/VIVTransformer.git
cd VIVTransformer

# 3. Add upstream repository
git remote add upstream https://github.com/original-repo/VIVTransformer.git

# 4. Create development environment
conda create -n vivtransformer-dev python=3.9
conda activate vivtransformer-dev

# 5. Install development dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

#### 2. Development Workflow

```bash
# 1. Sync latest code
git fetch upstream
git checkout main
git merge upstream/main

# 2. Create feature branch
git checkout -b feature/your-feature-name

# 3. Develop
# ... write code ...

# 4. Run tests
python -m pytest tests/
python -m pytest tests/ --cov=vivtransformer

# 5. Code formatting
black vivtransformer/
isort vivtransformer/
flake8 vivtransformer/

# 6. Commit changes
git add .
git commit -m "feat: add your feature description"

# 7. Push to your fork
git push origin feature/your-feature-name

# 8. Create Pull Request
# Go to GitHub and create a Pull Request
```

### Contribution Guidelines

#### Code Style

**Python Code Standards**:

```python
# Use type annotations
from typing import Optional, Tuple, Dict, Any
import torch
from torch import nn, Tensor

def compute_attention(
    query: Tensor, 
    key: Tensor, 
    mask: Optional[Tensor] = None
) -> Tensor:
    """Compute attention weights
    
    Args:
        query: Query tensor [batch_size, seq_len, d_model]
        key: Key tensor [batch_size, seq_len, d_model]
        mask: Optional mask tensor [batch_size, seq_len, seq_len]
    
    Returns:
        Attention weight tensor [batch_size, seq_len, seq_len]
    """
    # Compute attention scores
    scores = torch.matmul(query, key.transpose(-2, -1))
    
    # Apply mask
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    
    # Apply softmax
    attention_weights = torch.softmax(scores, dim=-1)
    
    return attention_weights
```

**Docstring Standards**:

```python
class VIVTransformer(nn.Module):
    """VIV Transformer model
    
    This is a Transformer model specifically designed for vortex-induced 
    vibration analysis, integrating multiple attention mechanisms and 
    physics-constrained loss functions.
    
    Args:
        d_model: Model dimension
        n_heads: Number of attention heads
        n_layers: Number of layers
        attention_type: Type of attention mechanism
        loss_config: Loss function configuration
    
    Example:
        >>> model = VIVTransformer(
        ...     d_model=512,
        ...     n_heads=8,
        ...     n_layers=6,
        ...     attention_type='scaled_dot_product'
        ... )
        >>> output = model(input_tensor)
    
    Note:
        The model supports multiple attention mechanisms, see attention_mechanisms module.
    """
    pass
```

#### Commit Message Standards

**Commit Message Format**:
```
<type>(<scope>): <subject>

<body>

<footer>
```

**Type Description**:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation update
- `style`: Code formatting
- `refactor`: Code refactoring
- `test`: Test related
- `chore`: Build process or auxiliary tool changes

**Examples**:
```
feat(attention): add external attention mechanism

- Implement external attention based on paper XYZ
- Add configuration options for external attention
- Update factory registration

Closes #123
```

---

## 📝 Code Contribution

### Core Module Development

#### 1. Attention Mechanism Development

**New Attention Mechanism Template**:
```python
# vivtransformer/attention_mechanisms/your_attention.py

import torch
import torch.nn as nn
from typing import Optional, Tuple
from .base_attention import BaseAttention

class YourAttention(BaseAttention):
    """Your attention mechanism implementation
    
    Based on paper: [Paper Title] (Author et al., Year)
    Paper link: https://arxiv.org/abs/XXXX.XXXXX
    
    Args:
        d_model: Model dimension
        n_heads: Number of attention heads
        dropout: Dropout probability
        **kwargs: Other parameters
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        dropout: float = 0.1,
        **kwargs
    ):
        super().__init__(d_model, n_heads, dropout)
        
        # Your specific parameters
        self.your_param = kwargs.get('your_param', default_value)
        
        # Your network layers
        self.your_layer = nn.Linear(d_model, d_model)
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass
        
        Args:
            query: Query tensor [batch_size, seq_len, d_model]
            key: Key tensor [batch_size, seq_len, d_model]
            value: Value tensor [batch_size, seq_len, d_model]
            mask: Mask tensor [batch_size, seq_len, seq_len]
        
        Returns:
            output: Output tensor [batch_size, seq_len, d_model]
            attention_weights: Attention weights [batch_size, n_heads, seq_len, seq_len]
        """
        batch_size, seq_len, d_model = query.shape
        
        # Your attention computation logic
        # ...
        
        return output, attention_weights
    
    @staticmethod
    def get_config_template() -> dict:
        """Return configuration template"""
        return {
            'type': 'your_attention',
            'n_heads': 8,
            'dropout': 0.1,
            'your_param': 'default_value'
        }
```

**Register New Attention Mechanism**:
```python
# vivtransformer/attention_mechanisms/__init__.py

from .your_attention import YourAttention

# Register to factory
ATTENTION_REGISTRY['your_attention'] = YourAttention
```

#### 2. Loss Function Development

**New Loss Function Template**:
```python
# vivtransformer/losses/your_loss.py

import torch
import torch.nn as nn
from typing import Dict, Any
from .base_loss import BaseLoss

class YourLoss(BaseLoss):
    """Your loss function implementation
    
    Args:
        weight: Loss weight
        **kwargs: Other parameters
    """
    
    def __init__(self, weight: float = 1.0, **kwargs):
        super().__init__(weight)
        self.your_param = kwargs.get('your_param', default_value)
    
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """Compute loss
        
        Args:
            predictions: Predicted values
            targets: Target values
            **kwargs: Other inputs
        
        Returns:
            Loss value
        """
        # Your loss computation logic
        loss = your_loss_calculation(predictions, targets)
        
        return self.weight * loss
    
    def get_metrics(self) -> Dict[str, float]:
        """Return related metrics"""
        return {
            'your_loss': self.last_loss_value,
            'your_metric': self.calculate_your_metric()
        }
```

#### 3. Data Processing Module

**New Data Processor Template**:
```python
# vivtransformer/data/processors/your_processor.py

import numpy as np
from typing import Dict, Any, Tuple
from .base_processor import BaseProcessor

class YourDataProcessor(BaseProcessor):
    """Your data processor
    
    Args:
        config: Processor configuration
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.your_param = config.get('your_param', default_value)
    
    def process(
        self,
        data: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Process data
        
        Args:
            data: Input data
        
        Returns:
            processed_data: Processed data
            metadata: Processing metadata
        """
        # Your data processing logic
        processed_data = your_processing_function(data)
        
        metadata = {
            'original_shape': data.shape,
            'processed_shape': processed_data.shape,
            'processing_time': self.processing_time
        }
        
        return processed_data, metadata
```

### Performance Optimization Contribution

#### 1. CUDA Kernel Optimization

```cpp
// vivtransformer/csrc/attention_cuda.cu

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void optimized_attention_kernel(
    const float* query,
    const float* key,
    const float* value,
    float* output,
    int batch_size,
    int seq_len,
    int d_model
) {
    // Your optimized CUDA kernel function
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < batch_size * seq_len * d_model) {
        // Computation logic
    }
}

torch::Tensor optimized_attention_cuda(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value
) {
    // CUDA function wrapper
    auto output = torch::zeros_like(query);
    
    const int threads = 256;
    const int blocks = (query.numel() + threads - 1) / threads;
    
    optimized_attention_kernel<<<blocks, threads>>>(
        query.data_ptr<float>(),
        key.data_ptr<float>(),
        value.data_ptr<float>(),
        output.data_ptr<float>(),
        query.size(0),
        query.size(1),
        query.size(2)
    );
    
    return output;
}
```

#### 2. Memory Optimization

```python
# vivtransformer/utils/memory_optimization.py

import torch
from typing import Iterator, Tuple

class GradientCheckpointing:
    """Gradient checkpointing optimization"""
    
    @staticmethod
    def checkpoint_sequential(
        functions: Iterator[torch.nn.Module],
        segments: int,
        input: torch.Tensor
    ) -> torch.Tensor:
        """Sequential gradient checkpointing"""
        def run_function(start, end, functions):
            def forward(input):
                for j in range(start, end + 1):
                    input = functions[j](input)
                return input
            return forward
        
        if segments == 1:
            return torch.utils.checkpoint.checkpoint(
                run_function(0, len(functions) - 1, functions),
                input
            )
        
        # Segmented checkpointing
        segment_size = len(functions) // segments
        for i in range(segments):
            start_idx = i * segment_size
            end_idx = (i + 1) * segment_size - 1
            if i == segments - 1:
                end_idx = len(functions) - 1
            
            input = torch.utils.checkpoint.checkpoint(
                run_function(start_idx, end_idx, functions),
                input
            )
        
        return input
```

---

## 📚 Documentation Contribution

### Documentation Types

#### 1. API Documentation

**Function Documentation Template**:
```python
def complex_function(
    param1: torch.Tensor,
    param2: Optional[str] = None,
    param3: Dict[str, Any] = None
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Detailed documentation for complex functions
    
    This function performs complex computations including multiple steps and conditional logic.
    Suitable for scenarios requiring precise control over the computation process.
    
    Args:
        param1: Input tensor with shape [batch_size, seq_len, d_model]
            - batch_size: Batch size, typically 32-512
            - seq_len: Sequence length, supports variable length
            - d_model: Model dimension, must be multiple of 8
        param2: Optional string parameter, supports the following values:
            - 'mode1': Use standard computation mode
            - 'mode2': Use optimized computation mode
            - None: Automatically select mode
        param3: Configuration dictionary containing the following keys:
            - 'threshold': float, threshold parameter (default: 0.5)
            - 'iterations': int, number of iterations (default: 10)
            - 'verbose': bool, whether to output detailed information (default: False)
    
    Returns:
        result: Result tensor with the same shape as param1
        metrics: Computation metrics dictionary containing:
            - 'computation_time': Computation time (seconds)
            - 'memory_usage': Memory usage (MB)
            - 'convergence_score': Convergence score (0-1)
    
    Raises:
        ValueError: When param1 has incorrect dimensions
        RuntimeError: When numerical instability occurs during computation
        MemoryError: When memory is insufficient
    
    Example:
        Basic usage:
        
        >>> import torch
        >>> input_tensor = torch.randn(32, 128, 512)
        >>> result, metrics = complex_function(input_tensor)
        >>> print(f"Computation time: {metrics['computation_time']:.3f}s")
        
        Advanced configuration:
        
        >>> config = {
        ...     'threshold': 0.8,
        ...     'iterations': 20,
        ...     'verbose': True
        ... }
        >>> result, metrics = complex_function(
        ...     input_tensor,
        ...     param2='mode2',
        ...     param3=config
        ... )
    
    Note:
        - This function performs best when running on GPU
        - For large inputs, gradient checkpointing is recommended
        - Supports mixed precision training
    
    See Also:
        - :func:`related_function`: Related function
        - :class:`RelatedClass`: Related class
        - :doc:`../tutorials/advanced_usage`: Advanced usage tutorial
    
    References:
        [1] Author et al. "Paper Title". Journal Name, 2024.
        [2] https://example.com/documentation
    """
```

#### 2. Tutorial Documentation

**Tutorial Structure Template**:
```markdown
# Tutorial Title

> Brief description of tutorial content and target audience

## Learning Objectives

After completing this tutorial, you will be able to:
- [ ] Objective 1
- [ ] Objective 2
- [ ] Objective 3

## Prerequisites

- Python 3.8+
- PyTorch 1.9+
- Basic knowledge of deep learning

## Step 1: Environment Setup

### Install Dependencies

```bash
pip install vivtransformer
```

### Verify Installation

```python
import vivtransformer
print(f"VIVTransformer version: {vivtransformer.__version__}")
```

## Step 2: Data Preparation

### Data Format

```python
# Data should be in the following format
data = {
    'input': torch.tensor(...),  # [batch_size, seq_len, features]
    'target': torch.tensor(...), # [batch_size, seq_len, targets]
    'metadata': {...}            # Metadata dictionary
}
```

### Data Loading

```python
from vivtransformer.data import VIVDataLoader

# Create data loader
loader = VIVDataLoader(
    data_path='path/to/data',
    batch_size=32,
    shuffle=True
)

# Iterate over data
for batch in loader:
    input_data = batch['input']
    target_data = batch['target']
    # Process data...
```

## Step 3: Model Configuration

### Basic Configuration

```python
from vivtransformer import VIVTransformer

# Create model
model = VIVTransformer(
    d_model=512,
    n_heads=8,
    n_layers=6,
    attention_type='external'
)
```

### Advanced Configuration

```yaml
# config.yaml
model:
  d_model: 512
  n_heads: 8
  n_layers: 6
  attention_type: 'external'
  
loss:
  type: 'svd_enhanced'
  weights:
    reconstruction: 1.0
    singular_value: 0.5
    orthogonality: 0.3

training:
  learning_rate: 1e-4
  batch_size: 32
  epochs: 100
```

```python
from vivtransformer.config import load_config

# Load from configuration file
config = load_config('config.yaml')
model = VIVTransformer.from_config(config.model)
```

## Common Issues

### Q: How to handle memory issues?

A: You can try the following methods:
1. Reduce batch_size
2. Use gradient accumulation
3. Enable gradient checkpointing

```python
# Gradient accumulation example
accumulation_steps = 4
for i, batch in enumerate(dataloader):
    loss = model(batch)
    loss = loss / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### Q: How to choose attention mechanism?

A: Characteristics of different attention mechanisms:
- `scaled_dot_product`: Standard attention, computationally efficient
- `external`: External attention, suitable for long sequences
- `se_attention`: Channel attention, suitable for feature selection

## Summary

This tutorial introduced...

## Next Steps

- [ ] Read advanced configuration tutorial
- [ ] Try custom attention mechanisms
- [ ] Participate in community discussions
```

#### 3. Example Code

**Complete Example Template**:
```python
#!/usr/bin/env python3
"""
Example: VIVTransformer Basic Usage

This example demonstrates how to use VIVTransformer for vortex-induced vibration prediction.
Includes a complete workflow from data preparation, model training to result visualization.

Author: [Your Name]
Date: 2024-01-01
Version: 1.0.0
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple

# VIVTransformer related imports
from vivtransformer import VIVTransformer
from vivtransformer.data import VIVDataset, VIVDataLoader
from vivtransformer.losses import SVDEnhancedLoss
from vivtransformer.utils import set_seed, get_device
from vivtransformer.visualization import plot_attention_weights, plot_predictions


def main():
    """Main function"""
    # Set random seed
    set_seed(42)
    
    # Get device
    device = get_device()
    print(f"Using device: {device}")
    
    # 1. Data preparation
    print("\n=== Data Preparation ===")
    train_dataset, val_dataset = prepare_data()
    
    train_loader = VIVDataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4
    )
    
    val_loader = VIVDataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4
    )
    
    print(f"Training set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")
    
    # 2. Model creation
    print("\n=== Model Creation ===")
    model = create_model(device)
    print(f"Model parameters: {count_parameters(model):,}")
    
    # 3. Training configuration
    print("\n=== Training Configuration ===")
    criterion = SVDEnhancedLoss(
        reconstruction_weight=1.0,
        singular_value_weight=0.5,
        orthogonality_weight=0.3
    )
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4