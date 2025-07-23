from typing import Callable, Optional, Tuple
import os

import numpy as np
import torch
from torch.utils.data import Dataset
import h5py


class PressureDataset(Dataset):
    """Dataset for loading pressure field data.

    Attributes:
        data: The loaded data from the torch file.
        in_pressures: Input pressure tensors.
        pressures: Ground truth pressure tensors.
        time_steps: Time step values corresponding to the data.
        transform: Optional transform to be applied on a sample.
        num_samples: The total number of samples in the dataset.
    """

    def __init__(self, merged_file_path: str, transform: Optional[Callable] = None):
        """Initializes the PressureDataset.

        Args:
            merged_file_path: Path to the .pt or .hdf5 file containing the data.
            transform: Optional transform to be applied on a sample.
        """
        self.file_path = merged_file_path
        self.transform = transform
        
        # 根据文件扩展名选择加载方法
        if merged_file_path.endswith('.hdf5') or merged_file_path.endswith('.h5'):
            self._load_hdf5_data()
        elif merged_file_path.endswith('.pt'):
            self._load_pt_data()
        else:
            raise ValueError(f"不支持的文件格式: {merged_file_path}")
        
        self.num_samples = len(self.in_pressures)
    
    def _load_pt_data(self):
        """加载 .pt 格式数据"""
        self.data = torch.load(self.file_path, weights_only=False)
        self.in_pressures = self.data["in_pressure"]  # Shape: [N, 400]
        self.pressures = self.data["pressure"]        # Shape: [N, 40000]
        self.time_steps = self.data["time_steps"]     # Shape: [N]
    
    def _load_hdf5_data(self):
        """加载 .hdf5 格式数据（DarcyFlow 数据集）"""
        with h5py.File(self.file_path, 'r') as f:
            # 检查数据集结构
            print(f"HDF5 文件键: {list(f.keys())}")
            
            # DarcyFlow 数据集通常包含 'tensor' 键
            if 'tensor' in f:
                data = f['tensor'][:]  # Shape: [N, 2, H, W] 或类似
                print(f"原始数据形状: {data.shape}")
                
                # 处理不同的数据格式
                if len(data.shape) == 4:
                    if data.shape[1] == 2:
                        # 格式: [N, 2, H, W] - 两个通道
                        input_data = data[:, 0, :, :]  # [N, H, W]
                        output_data = data[:, 1, :, :] # [N, H, W]
                    elif data.shape[1] == 1:
                        # 格式: [N, 1, H, W] - 单通道，使用同一数据进行超分辨率任务
                        original_data = data[:, 0, :, :]  # [N, H, W]
                        # 输入是下采样版本，输出是原始高分辨率版本
                        input_data = original_data  # 将作为高分辨率输出
                        output_data = original_data  # 原始数据作为输出
                    else:
                        raise ValueError(f"不支持的通道数: {data.shape[1]}")
                    
                    # 调整大小以匹配期望的输入输出维度
                    # 输入: 20x20 -> 400, 输出: 200x200 -> 40000
                    from scipy.ndimage import zoom
                    
                    # 重采样到目标尺寸
                    input_resized = []
                    output_resized = []
                    
                    # 限制样本数量以避免内存问题
                    max_samples = min(1000, data.shape[0])
                    
                    for i in range(max_samples):
                        # 输入重采样到 20x20（下采样）
                        input_20x20 = zoom(input_data[i], (20/input_data.shape[1], 20/input_data.shape[2]))
                        input_resized.append(input_20x20.flatten())
                        
                        # 输出重采样到 200x200（上采样或保持原分辨率）
                        if output_data.shape[1] != 200 or output_data.shape[2] != 200:
                            output_200x200 = zoom(output_data[i], (200/output_data.shape[1], 200/output_data.shape[2]))
                        else:
                            output_200x200 = output_data[i]
                        output_resized.append(output_200x200.flatten())
                    
                    self.in_pressures = torch.tensor(np.array(input_resized), dtype=torch.float32)
                    self.pressures = torch.tensor(np.array(output_resized), dtype=torch.float32)
                    self.time_steps = torch.arange(len(input_resized), dtype=torch.float32)
                    
                    print(f"成功加载 {len(input_resized)} 个样本")
                    print(f"输入形状: {self.in_pressures.shape}")
                    print(f"输出形状: {self.pressures.shape}")
                    
                else:
                    raise ValueError(f"不支持的 HDF5 数据格式: {data.shape}")
            else:
                raise ValueError(f"HDF5 文件中未找到 'tensor' 键，可用键: {list(f.keys())}")

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """Gets a sample from the dataset at the specified index.

        Args:
            idx: The index of the sample to retrieve.

        Returns:
            A tuple containing:
            - The input pressure tensor.
            - The ground truth pressure tensor.
            - The corresponding time step.
        """
        # 直接从数据中获取样本
        in_press = self.in_pressures[idx]  # Shape: [400]
        pressure = self.pressures[idx]     # Shape: [40000]
        time_step = self.time_steps[idx]   # Scalar

        # 应用变换（如果有）
        if self.transform:
            # 重塑为2D进行变换
            in_press_2d = in_press.view(20, 20)
            pressure_2d = pressure.view(200, 200)
            
            in_press_2d = self.transform(in_press_2d)
            pressure_2d = self.transform(pressure_2d)
            
            # 重新展平
            in_press = in_press_2d.reshape(-1)
            pressure = pressure_2d.reshape(-1)

        return in_press, pressure, float(time_step)
