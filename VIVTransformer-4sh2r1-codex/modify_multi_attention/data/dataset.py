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
        self.in_pressures = self.data["in_pressure"]
        self.pressures = self.data["pressure"]
        self.time_steps = np.array(self.data["time_steps"])
        self.transform = transform

        if len(self.time_steps.shape) == 1:
            self.time_steps = np.array([self.time_steps] * len(self.in_pressures))

        self.num_samples = len(self.in_pressures) * len(self.in_pressures[0])

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
        reynolds_idx = idx // len(self.in_pressures[0])
        time_step_idx = idx % len(self.in_pressures[0])

        in_press = self.in_pressures[reynolds_idx, time_step_idx]
        pressure = self.pressures[reynolds_idx, time_step_idx]
        time_step = self.time_steps[reynolds_idx][time_step_idx]

        # Reshape to 2D before transform
        in_press_2d = in_press.view(20, 20)
        pressure_2d = pressure.view(200, 200)



        # Flatten after transform
        in_press_flat = in_press_2d.reshape(-1)
        pressure_flat = pressure_2d.reshape(-1)

        return in_press_flat, pressure_flat, time_step
