from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch


def _setup_plot_directory(parent_dir: str, attention_type: str, sub_dir: str) -> Path:
    """Creates and returns the directory for saving plots."""
    result_dir = Path(parent_dir) / attention_type / sub_dir
    result_dir.mkdir(exist_ok=True, parents=True)
    return result_dir


def _save_and_close_plot(save_path: Path) -> None:
    """Saves the current plot to a file and closes it."""
    plt.savefig(save_path)
    plt.close()


def plot_comparison_figure(
    input_pressure: np.ndarray,
    true_pressure: np.ndarray,
    predicted_pressure: np.ndarray,
    time_step: float,
    epoch: int,
    attention_type: str,
    idx: int,
    parent_dir: str = "attention_results",
    mode: str = "test",
) -> None:
    """Plots and saves a comparison of input, true, and predicted pressure fields.

    Args:
        input_pressure: The input pressure data.
        true_pressure: The ground truth pressure data.
        predicted_pressure: The model's predicted pressure data.
        time_step: The time step of the data sample.
        epoch: The current epoch number.
        attention_type: The type of attention mechanism used.
        idx: The index of the sample.
        parent_dir: The root directory for saving results.
        mode: The mode of operation ('test' or 'validation').
    """
    result_dir = _setup_plot_directory(
        parent_dir, attention_type, "visualization_results"
    )

    plt.figure(figsize=(18, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(input_pressure, cmap="coolwarm", interpolation="nearest")
    plt.colorbar()
    plt.title(f"Input Pressure Matrix at t={time_step:.2f}")

    plt.subplot(1, 3, 2)
    plt.imshow(true_pressure, cmap="coolwarm", interpolation="nearest")
    plt.colorbar()
    plt.title(f"True Pressure Matrix at t={time_step:.2f}")

    plt.subplot(1, 3, 3)
    plt.imshow(predicted_pressure, cmap="coolwarm", interpolation="nearest")
    plt.colorbar()
    plt.title(f"Predicted Pressure Matrix at t={time_step:.2f}")

    plt.tight_layout()

    # 保存图片到对应文件夹
    save_path = result_dir / f"{mode}_epoch_{epoch}_sample_{idx}.png"
    _save_and_close_plot(save_path)


def plot_losses(
    train_loss: List[float],
    valid_loss: List[float],
    test_loss: List[float],
    save_path: Optional[str] = None,
) -> None:
    """Plots and saves the training, validation, and test loss curves.

    Args:
        train_loss: A list of training losses per epoch.
        valid_loss: A list of validation losses per epoch.
        test_loss: A list of test losses per epoch.
        save_path: The file path to save the plot. If None, displays the plot.
    """
    if len(train_loss) == 0 or len(valid_loss) == 0 or len(test_loss) == 0:
        print("⚠️ 损失列表为空，无法绘制Loss曲线！")
        return

    plt.figure(figsize=(10, 6))
    plt.plot(train_loss, label="Train Loss")
    plt.plot(valid_loss, label="Valid Loss")
    plt.plot(test_loss, label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.legend()

    if save_path:
        _save_and_close_plot(Path(save_path))
        print(f"Loss曲线已保存到: {save_path}")
    else:
        plt.show()
        plt.close()


def plot_attention_maps(
    attention_maps: List[torch.Tensor],
    epoch: int,
    attention_type: str,
    parent_dir: str = "attention_results",
    max_heads: int = 4,  # 最多可视化前几个头
) -> None:
    """Plots and saves attention maps from the model.

    Args:
        attention_maps: The attention weights tensor (batch_size, num_heads, ...).
        epoch: The current epoch number.
        attention_type: The type of attention mechanism used.
        parent_dir: The root directory for saving results.
        max_heads: The maximum number of attention heads to visualize.
    """
    result_dir = _setup_plot_directory(parent_dir, attention_type, "attention_maps")

    for layer_idx, layer_attention in enumerate(attention_maps):
        # 如果 layer_attention 是一个元组，我们假设第一个元素是注意力权重
        if isinstance(layer_attention, tuple):
            layer_attention = layer_attention[0]

        # 在处理之前检查 layer_attention 是否为 None
        if layer_attention is None:
            print(f"Layer {layer_idx + 1} attention map is None after unpacking, skipping visualization.")
            continue

        # 将 Tensor 移动到 CPU 并转换为 numpy
        layer_attention = layer_attention.detach().cpu().numpy()

        # 只可视化第一个样本的注意力图
        sample_attention = layer_attention[0]

        num_heads = min(sample_attention.shape[0], max_heads)

        fig, axes = plt.subplots(1, num_heads, figsize=(5 * num_heads, 5))
        if num_heads == 1:
            axes = [axes]  # 保证 axes 是可迭代的

        for i in range(num_heads):
            im = axes[i].imshow(sample_attention[i], cmap="viridis")
            axes[i].set_title(f"Head {i + 1}")
            axes[i].set_xlabel("Keys")
            axes[i].set_ylabel("Queries")

        fig.colorbar(im, ax=axes, orientation="horizontal", fraction=0.05, pad=0.1)
        plt.suptitle(f"Layer {layer_idx + 1} Attention Maps at Epoch {epoch}")

        save_path = result_dir / f"epoch_{epoch}_layer_{layer_idx + 1}_attention_maps.png"
        _save_and_close_plot(save_path)


def plot_difference_figure(
    true_pressure: np.ndarray,
    predicted_pressure: np.ndarray,
    time_step: float,
    epoch: int,
    attention_type: str,
    idx: int,
    parent_dir: str = "attention_results",
    mode: str = "test",
) -> None:
    """Plots and saves the absolute difference between true and predicted pressure.

    Args:
        true_pressure: The ground truth pressure data.
        predicted_pressure: The model's predicted pressure data.
        time_step: The time step of the data sample.
        epoch: The current epoch number.
        attention_type: The type of attention mechanism used.
        idx: The index of the sample.
        parent_dir: The root directory for saving results.
        mode: The mode of operation ('test' or 'validation').
    """
    result_dir = _setup_plot_directory(
        parent_dir, attention_type, "difference_results"
    )

    # 计算差异（绝对误差）
    difference = np.abs(true_pressure - predicted_pressure)

    plt.figure(figsize=(6, 5))
    plt.imshow(difference, cmap="hot", interpolation="nearest")
    plt.colorbar()
    plt.title(f"Difference (|True - Predicted|) at t={time_step:.2f}")

    plt.tight_layout()

    # 保存图片到对应文件夹
    save_path = result_dir / f"{mode}_epoch_{epoch}_sample_{idx}_difference.png"
    _save_and_close_plot(save_path)
