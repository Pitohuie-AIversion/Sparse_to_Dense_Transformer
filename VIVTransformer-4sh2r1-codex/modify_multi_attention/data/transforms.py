from typing import Tuple

import torch
import random
import numpy as np


class RandomFlip:
    """Randomly flips the input tensors horizontally or vertically.

    Attributes:
        p: The probability of applying a flip.
    """

    def __init__(self, p: float = 0.5):
        """Initializes the RandomFlip transform.

        Args:
            p: The probability of applying a flip.
        """
        self.p = p

    def __call__(
        self, sample: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Applies the transform to a sample.

        Args:
            sample: A tuple containing the input and target pressure tensors.

        Returns:
            The transformed sample.
        """
        in_press, pressure = sample
        # 假设输入是 (C, H, W) 或 (H, W)
        if random.random() < self.p:
            # 水平翻转
            in_press = torch.flip(in_press, dims=[-1])
            pressure = torch.flip(pressure, dims=[-1])
        if random.random() < self.p:
            # 垂直翻转
            in_press = torch.flip(in_press, dims=[-2])
            pressure = torch.flip(pressure, dims=[-2])
        return in_press, pressure


class RandomRotate:
    """Randomly rotates the input tensors by 90, 180, or 270 degrees."""

    def __call__(
        self, sample: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Applies the transform to a sample.

        Args:
            sample: A tuple containing the input and target pressure tensors.

        Returns:
            The transformed sample.
        """
        in_press, pressure = sample
        k = random.randint(0, 3)
        if k > 0:
            in_press = torch.rot90(in_press, k, dims=[-2, -1])
            pressure = torch.rot90(pressure, k, dims=[-2, -1])
        return in_press, pressure


class AddGaussianNoise:
    """Adds Gaussian noise to the input tensor.

    Attributes:
        mean: The mean of the Gaussian noise.
        std: The standard deviation of the Gaussian noise.
    """

    def __init__(self, mean: float = 0.0, std: float = 0.01):
        """Initializes the AddGaussianNoise transform.

        Args:
            mean: The mean of the Gaussian noise.
            std: The standard deviation of the Gaussian noise.
        """
        self.std = std
        self.mean = mean

    def __call__(
        self, sample: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Applies the transform to a sample.

        Args:
            sample: A tuple containing the input and target pressure tensors.

        Returns:
            The transformed sample with noise added to the input tensor.
        """
        in_press, pressure = sample
        noise = torch.randn(in_press.size()) * self.std + self.mean
        in_press = in_press + noise
        # 通常我们只对输入应用噪声
        return in_press, pressure


class RandomCrop:
    """Randomly crops the input tensors.

    Attributes:
        output_size: The desired output size (height, width).
    """

    def __init__(self, output_size: Tuple[int, int]):
        """Initializes the RandomCrop transform.

        Args:
            output_size: The desired output size (height, width).
        """
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(
        self, sample: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Applies the transform to a sample.

        Args:
            sample: A tuple containing the input and target pressure tensors.

        Returns:
            The cropped sample.
        """
        in_press, pressure = sample
        h_p, w_p = pressure.shape[-2:]  # 使用 pressure 作为基准
        h_i, w_i = in_press.shape[-2:]
        new_h, new_w = self.output_size

        if h_p < new_h or w_p < new_w:
            # 如果输出尺寸大于 pressure 尺寸，可以进行填充或直接返回
            # 这里选择直接返回，或者可以实现一个 Pad transform
            return in_press, pressure

        # 计算 pressure 的裁切位置
        top = torch.randint(0, h_p - new_h + 1, (1,)).item()
        left = torch.randint(0, w_p - new_w + 1, (1,)).item()

        # 裁切 pressure
        pressure_cropped = pressure[..., top: top + new_h, left: left + new_w]

        # 根据 pressure 的裁切比例，计算 in_press 的裁切参数
        h_ratio = h_i / h_p
        w_ratio = w_i / w_p

        top_i = int(top * h_ratio)
        left_i = int(left * w_ratio)
        new_h_i = int(new_h * h_ratio)
        new_w_i = int(new_w * w_ratio)

        # 确保 in_press 的裁切尺寸至少为1x1
        new_h_i = max(1, new_h_i)
        new_w_i = max(1, new_w_i)

        # 裁切 in_press
        in_press_cropped = in_press[..., top_i: top_i + new_h_i, left_i: left_i + new_w_i]

        return in_press_cropped, pressure_cropped


class ToTensor:
    """Converts NumPy arrays in a sample to PyTorch tensors."""

    def __call__(
        self, sample: Tuple[np.ndarray, np.ndarray]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Applies the transform to a sample.

        Args:
            sample: A tuple containing the input and target pressure NumPy arrays.

        Returns:
            The transformed sample with tensors.
        """
        in_press, pressure = sample
        return torch.from_numpy(in_press).float(), torch.from_numpy(pressure).float()
