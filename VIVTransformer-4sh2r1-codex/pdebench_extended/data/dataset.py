from typing import Callable, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


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
            merged_file_path: Path to the .pt file containing the data.
            transform: Optional transform to be applied on a sample.
        """
        self.data = torch.load(merged_file_path)
        self.in_pressures = self.data["in_pressure"]  # Shape: [N, 400]
        self.pressures = self.data["pressure"]        # Shape: [N, 40000]
        self.time_steps = self.data["time_steps"]     # Shape: [N]
        self.transform = transform

        self.num_samples = len(self.in_pressures)

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
