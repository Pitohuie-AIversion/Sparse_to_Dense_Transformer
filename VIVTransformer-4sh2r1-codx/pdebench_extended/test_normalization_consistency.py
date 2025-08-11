import torch
import numpy as np
from dataset import PressureFieldDataset, MultiScaleDataset

"""Test normalization consistency"""
print("=== Normalization Consistency Test ===")

# Create base dataset
try:
    base_dataset = PressureFieldDataset(
        data_path="/storage/home/lzl5041/work/Sparse_to_Dense_Transformer/Sparse_to_Dense_Transformer_public/datasets/data/3D_recon_data_sample.pt",
        sequence_length=50,
        downsample_factor=10,
        normalize=False  # No normalization in base dataset
    )
    
    # Create multi-scale dataset
    multiscale_dataset = MultiScaleDataset(
        data_path="/storage/home/lzl5041/work/Sparse_to_Dense_Transformer/Sparse_to_Dense_Transformer_public/datasets/data/3D_recon_data_sample.pt",
        sequence_length=50,
        downsample_factors=[5, 10, 20],
        normalize=True,  # Apply normalization in multi-scale dataset
        target_sparsity=0.95,
        train=True
    )
    
    print(f"\nDataset Information:")
    print(f"- Base dataset size: {len(base_dataset)}")
    print(f"- Multi-scale dataset size: {len(multiscale_dataset)}")
    print(f"- Normalization parameters: mean={multiscale_dataset.data_mean:.6f}, std={multiscale_dataset.data_std:.6f}")
    
    # Get several samples for testing
    num_samples = min(5, len(multiscale_dataset))
    
    print(f"\n=== Analyzing statistical properties of first {num_samples} samples ===")
    
    input_means = []
    target_means = []
    input_stds = []
    target_stds = []
    
    for i in range(num_samples):
        try:
            # Calculate statistics
            inputs, targets = multiscale_dataset[i]
            
            # Convert to CPU for computation
            inputs_np = inputs.cpu().numpy()
            targets_np = targets.cpu().numpy()
            
            input_mean = np.mean(inputs_np)
            target_mean = np.mean(targets_np)
            input_std = np.std(inputs_np)
            target_std = np.std(targets_np)
            
            input_means.append(input_mean)
            target_means.append(target_mean)
            input_stds.append(input_std)
            target_stds.append(target_std)
            
            print(f"\nSample {i+1}:")
            print(f"  Input shape: {inputs.shape}")
            print(f"  Target shape: {targets.shape}")
            print(f"  Input statistics: mean={input_mean:.6f}, std={input_std:.6f}")
            print(f"  Target statistics: mean={target_mean:.6f}, std={target_std:.6f}")
            print(f"  Mean difference: {abs(input_mean - target_mean):.6f}")
            print(f"  Std difference: {abs(input_std - target_std):.6f}")
            
        except Exception as e:
            print(f"Sample {i+1} processing failed: {e}")
            continue
    
    # Calculate overall statistics
    if input_means:
        avg_input_mean = np.mean(input_means)
        avg_target_mean = np.mean(target_means)
        avg_input_std = np.mean(input_stds)
        avg_target_std = np.mean(target_stds)
        
        print(f"\n=== Overall Statistical Analysis ===")
        print(f"Input data average statistics: mean={avg_input_mean:.6f}, std={avg_input_std:.6f}")
        print(f"Target data average statistics: mean={avg_target_mean:.6f}, std={avg_target_std:.6f}")
        print(f"Mean difference: {abs(avg_input_mean - avg_target_mean):.6f}")
        print(f"Std difference: {abs(avg_input_std - avg_target_std):.6f}")
        
        # Assess normalization consistency
        mean_diff = abs(avg_input_mean - avg_target_mean)
        std_diff = abs(avg_input_std - avg_target_std)
        
        print(f"\n=== Normalization Consistency Assessment ===")
        if mean_diff < 0.1 and std_diff < 0.1:
            print("✅ Good normalization consistency: input and target data have similar statistical properties")
        elif mean_diff < 0.3 and std_diff < 0.3:
            print("⚠️  Fair normalization consistency: input and target data have some differences")
        else:
            print("❌ Poor normalization consistency: input and target data have significant statistical differences")
            print("   This may lead to unstable model training and inaccurate predictions")
        
        print(f"\nRecommendations:")
        print(f"- Ensure input and target use the same normalization parameters")
        print(f"- Consider recalculating normalization statistics after downsampling")
        print(f"- Or use original data statistics but ensure processing doesn't change distribution")
    
except Exception as e:
    print(f"Test failed: {e}")
    print("Please check if the data path is correct and the data file exists")