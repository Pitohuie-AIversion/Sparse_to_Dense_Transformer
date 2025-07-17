import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import torch
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from modify_multi_attention.utils.visualization import (
    plot_attention_maps,  # 导入新的可视化函数
    plot_comparison_figure,
    plot_difference_figure,
    plot_losses,
)


def _train_epoch(
    model, loader, criterion, optimizer, device, debug, epoch, attention_type, result_dir
):
    model.train()
    total_loss = 0
    logger = logging.getLogger(__name__)
    for i, (in_press, out_pressure, time_steps) in enumerate(loader):
        in_press, out_pressure, time_steps = (
            in_press.to(device),
            out_pressure.to(device),
            time_steps.to(device),
        )
        optimizer.zero_grad()
        if debug:
            model_out, attention_weights = model(
                in_press, time_steps, return_attention=True
            )
            if i == 0:
                plot_attention_maps(
                    attention_weights,
                    epoch=epoch,
                    attention_type=attention_type,
                    parent_dir=result_dir,
                )
        else:
            model_out = model(in_press, time_steps)
        loss = criterion(model_out, out_pressure)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if (i + 1) % 50 == 0 or i == 0:
            logger.info(
                f"    🔄 Epoch [{epoch + 1}], Batch [{i + 1}/{len(loader)}], Loss: {loss.item():.6f}"
            )
    return total_loss / len(loader)

def _evaluate_model(model, loader, criterion, device, debug, epoch, attention_type, save_dir):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for i, (in_press, out_pressure, time_steps) in enumerate(loader):
            in_press, out_pressure, time_steps = (
                in_press.to(device),
                out_pressure.to(device),
                time_steps.to(device),
            )
            if debug:
                model_out, attention_weights = model(
                    in_press, time_steps, return_attention=True
                )
                if i == 0 and (epoch + 1) % 100 == 0:
                    plot_attention_maps(
                        attention_weights,
                        epoch=epoch + 1,
                        attention_type=attention_type,
                        parent_dir=save_dir,
                    )
            else:
                model_out = model(in_press, time_steps)
            loss = criterion(model_out, out_pressure)
            total_loss += loss.item()
    return total_loss / len(loader)

def _save_checkpoint(epoch, model, optimizer, losses, best_loss, patience, path):
    train_loss, valid_loss, test_loss = losses
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_loss_history": train_loss,
        "valid_loss_history": valid_loss,
        "test_loss_history": test_loss,
        "best_valid_loss": best_loss,
        "patience_counter": patience,
    }
    torch.save(checkpoint, path)

def _run_visualization(
    model, loader, device, max_samples, epoch, attention_type, parent_dir, mode
):
    model.eval()
    logger = logging.getLogger(__name__)
    with torch.no_grad():
        try:
            sample_loader = iter(loader)
            for idx in range(min(max_samples, len(loader))):
                sample_input, sample_output, sample_time_steps = next(sample_loader)
                sample_input, sample_output, sample_time_steps = (
                    sample_input.to(device),
                    sample_output.to(device),
                    sample_time_steps.to(device),
                )
                predictions = model(sample_input, sample_time_steps)
                input_pressure = sample_input[0].view(20, 20).cpu().numpy()
                true_pressure = sample_output[0].view(200, 200).cpu().numpy()
                predicted_pressure = predictions[0].view(200, 200).cpu().numpy()
                plot_comparison_figure(
                    input_pressure=input_pressure,
                    true_pressure=true_pressure,
                    predicted_pressure=predicted_pressure,
                    time_step=sample_time_steps[0].item(),
                    epoch=epoch + 1,
                    idx=idx,
                    attention_type=attention_type,
                    parent_dir=parent_dir,
                    mode=mode,
                )
                plot_difference_figure(
                    true_pressure=true_pressure,
                    predicted_pressure=predicted_pressure,
                    time_step=sample_time_steps[0].item(),
                    epoch=epoch + 1,
                    idx=idx,
                    attention_type=attention_type,
                    parent_dir=parent_dir,
                    mode=mode,
                )
        except StopIteration:
            logger.warning(f"⚠️ {mode} set data is not enough for visualization.")

def train_model(
    model: torch.nn.Module,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    test_loader: DataLoader,
    criterion: torch.nn.Module,
    optimizer: Optimizer,
    num_epochs: int,
    device: torch.device,
    early_stop_patience: int,
    result_dir: str,
    cfg: Dict[str, Any],
    attention_type: str,
    debug: bool = False,
) -> Tuple[torch.nn.Module, List[float], List[float], List[float]]:
    """Trains a model, evaluates it, and saves checkpoints, with support for TensorBoard logging and debug visualization.

    This function handles the complete training loop, including:
    - Training the model on the training dataset using a custom loss function.
    - Evaluating the model on the validation and test datasets using standard MSE loss for comparability.
    - Logging training, validation, and test losses to a file and TensorBoard.
    - Saving model checkpoints, including model state, optimizer state, and loss history.
    - Implementing early stopping to prevent overfitting.
    - Visualizing attention maps and parameter gradients in debug mode.

    Args:
        model (torch.nn.Module): The neural network model to train.
        train_loader (DataLoader): DataLoader for the training set.
        valid_loader (DataLoader): DataLoader for the validation set.
        test_loader (DataLoader): DataLoader for the test set.
        criterion (torch.nn.Module): The loss function for training.
        optimizer (Optimizer): The optimization algorithm.
        num_epochs (int): The total number of epochs to train.
        device (torch.device): The device to run the model on (e.g., 'cuda' or 'cpu').
        early_stop_patience (int): Number of epochs to wait for improvement before stopping.
        result_dir (str): Directory to save results, checkpoints, and logs.
        cfg (Dict[str, Any]): Configuration dictionary containing parameters for visualization and other settings.
        debug (bool, optional): If True, enables debug mode for visualizing attention maps and gradients. Defaults to False.

    Returns:
        Tuple[torch.nn.Module, List[float], List[float], List[float]]: A tuple containing:
        - The trained model.
        - A list of training loss history.
        - A list of validation loss history.
        - A list of test loss history.
    """
    logger = logging.getLogger(__name__)
    result_dir = Path(result_dir)
    writer = SummaryWriter(log_dir=result_dir / 'runs')

    # 标准 MSELoss（保证横向可比）
    import torch.nn as nn

    mse_loss = nn.MSELoss()

    # 配置参数直接来自cfg
    vis_config = cfg.get("visualization", {})
    vis_enabled = vis_config.get("enabled", False)
    vis_interval = vis_config.get("interval", 100)
    max_samples = vis_config.get("max_samples", 5)

    model.to(device)

    train_loss_history = []
    valid_loss_history = []
    test_loss_history = []

    best_valid_loss = float("inf")
    patience_counter = 0

    result_dir = Path(result_dir)
    save_dir = result_dir
    loss_log_dir = result_dir / "loss_logs"
    loss_log_dir.mkdir(exist_ok=True)
    loss_log_path = loss_log_dir / "loss_log.txt"
    checkpoint_path = result_dir / f"checkpoint_{attention_type}.pth"

    logger.info(f"写入loss_log.txt到：{loss_log_path}")

    # ========== 恢复断点 ==========
    start_epoch = 0
    if os.path.exists(checkpoint_path):
        logger.info(f"检测到断点文件，自动恢复：{checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        train_loss_history = checkpoint.get("train_loss_history", [])
        valid_loss_history = checkpoint.get("valid_loss_history", [])
        test_loss_history = checkpoint.get("test_loss_history", [])
        best_valid_loss = checkpoint.get("best_valid_loss", float("inf"))
        patience_counter = checkpoint.get("patience_counter", 0)
        start_epoch = checkpoint.get("epoch", 0) + 1
        logger.info(f"已恢复到 epoch {start_epoch}，best_valid_loss={best_valid_loss}")
    else:
        with open(loss_log_path, "w") as log_file:
            log_file.write("Epoch, Train Loss, Valid Loss, Test Loss\n")

    # ========== 主训练循环 ==========
    for epoch in range(start_epoch, num_epochs):
        avg_train_loss = _train_epoch(
            model, train_loader, criterion, optimizer, device, debug, epoch, attention_type, result_dir
        )
        train_loss_history.append(avg_train_loss)
        writer.add_scalar('Loss/train', avg_train_loss, epoch)

        avg_valid_loss = _evaluate_model(
            model, valid_loader, mse_loss, device, debug, epoch, attention_type, save_dir
        )
        valid_loss_history.append(avg_valid_loss)
        writer.add_scalar('Loss/valid', avg_valid_loss, epoch)

        avg_test_loss = _evaluate_model(model, test_loader, mse_loss, device, False, 0, '', '')
        test_loss_history.append(avg_test_loss)
        writer.add_scalar('Loss/test', avg_test_loss, epoch)

        logger.info(
            f"🎯 Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_train_loss:.6f}, Valid Loss: {avg_valid_loss:.6f}, Test Loss: {avg_test_loss:.6f}"
        )

        with open(loss_log_path, "a") as log_file:
            log_file.write(
                f"{epoch + 1}, {avg_train_loss:.6f}, {avg_valid_loss:.6f}, {avg_test_loss:.6f}\n"
            )

        _save_checkpoint(
            epoch, model, optimizer, 
            (train_loss_history, valid_loss_history, test_loss_history), 
            best_valid_loss, patience_counter, checkpoint_path
        )

        if avg_valid_loss < best_valid_loss:
            best_valid_loss = avg_valid_loss
            patience_counter = 0
            torch.save(
                model.state_dict(),
                save_dir / f"best_model_{attention_type}.pt",
            )
            logger.info("✅ 模型已保存 (Best Model Updated)")
        else:
            patience_counter += 1
            logger.warning(f"⚠️ 早停计数: {patience_counter}/{early_stop_patience}")

        if patience_counter >= early_stop_patience:
            logger.info("⏹️ 触发 Early Stopping!")
            break

        if vis_enabled and (epoch + 1) % vis_interval == 0:
            _run_visualization(
                model, valid_loader, device, max_samples, epoch, attention_type, save_dir, "validation"
            )

        if (epoch + 1) % vis_interval == 0:
            plot_dir = save_dir / "loss_plots"
            plot_dir.mkdir(exist_ok=True)
            loss_fig_path = plot_dir / f"loss_curve_epoch_{epoch + 1}.png"
            plot_losses(
                train_loss_history,
                valid_loss_history,
                test_loss_history,
                save_path=loss_fig_path,
            )

    plt.ioff()
    writer.close()
    return model, train_loss_history, valid_loss_history, test_loss_history


def _test_loop(model, loader, criterion, device, log_path):
    model.eval()
    total_loss = 0
    with open(log_path, "w") as log_file:
        log_file.write("Batch, Test Loss\n")
    with torch.no_grad():
        for idx, (in_press, out_pressure, time_steps) in enumerate(loader):
            in_press, out_pressure, time_steps = (
                in_press.to(device),
                out_pressure.to(device),
                time_steps.to(device),
            )
            model_out = model(in_press, time_steps)
            loss = criterion(model_out, out_pressure)
            total_loss += loss.item()
            with open(log_path, "a") as log_file:
                log_file.write(f"{idx + 1}, {loss.item():.6f}\n")
    return total_loss / len(loader)

def test_model(
    model: torch.nn.Module,
    test_loader: DataLoader,
    criterion: torch.nn.Module,
    device: str = "cuda",
    attention_type: str = "default",
    parent_dir: Optional[str] = None,
    cfg: Optional[Dict[str, Any]] = None,
    debug: bool = False,
) -> float:
    """Evaluates the model on the test set.

    Args:
        model: The trained neural network model.
        test_loader: DataLoader for the test set.
        criterion: The loss function (though MSE is used for evaluation).
        device: The device to run the model on ('cuda' or 'cpu').
        attention_type: String identifier for the attention mechanism used.
        parent_dir: Directory where results are stored.
        cfg: Configuration dictionary.

    Returns:
        The average test loss.
    """
    logger = logging.getLogger(__name__)

    # 保证测试阶段只用MSELoss
    import torch.nn as nn

    mse_loss = nn.MSELoss()

    # 确保 parent_dir 是 Path 对象
    if parent_dir is None:
        # 如果未提供，则使用默认路径
        base_dir = Path("attention_results")
    else:
        base_dir = Path(parent_dir)

    vis_enabled = cfg["visualization"]["enabled"]
    max_samples = cfg["visualization"]["max_samples"]

    model.eval()
    model.to(device)
    total_test_loss = 0

    save_dir = base_dir / attention_type / "test_results"
    save_dir.mkdir(exist_ok=True, parents=True)

    loss_log_path = save_dir / "test_loss_log.txt"
    with open(loss_log_path, "w") as log_file:
        log_file.write("Batch, Test Loss\n")

    avg_test_loss = _test_loop(model, test_loader, mse_loss, device, loss_log_path)

    if vis_enabled:
        _run_visualization(
            model, test_loader, device, max_samples, 0, attention_type, base_dir, "test"
        )
    logger.info(f"🧪 测试完成，{attention_type} Test Loss: {avg_test_loss:.6f}")

    with open(loss_log_path, "a") as log_file:
        log_file.write(f"Average Test Loss: {avg_test_loss:.6f}\n")

    return avg_test_loss
    """Evaluates the model on the test set.

    Args:
        model: The trained neural network model.
        test_loader: DataLoader for the test set.
        criterion: The loss function (though MSE is used for evaluation).
        device: The device to run the model on ('cuda' or 'cpu').
        attention_type: String identifier for the attention mechanism used.
        parent_dir: Directory where results are stored.
        cfg: Configuration dictionary.

    Returns:
        The average test loss.
    """
    logger = logging.getLogger(__name__)

    # 保证测试阶段只用MSELoss
    import torch.nn as nn

    mse_loss = nn.MSELoss()

    vis_enabled = cfg["visualization"]["enabled"]
    max_samples = cfg["visualization"]["max_samples"]

    model.eval()
    model.to(device)
    total_test_loss = 0

    save_dir = Path(parent_dir) / attention_type / "test_results"
    save_dir.mkdir(exist_ok=True, parents=True)

    loss_log_path = save_dir / "test_loss_log.txt"
    with open(loss_log_path, "w") as log_file:
        log_file.write("Batch, Test Loss\n")

    with torch.no_grad():
        for idx, (in_press, out_pressure, time_steps) in enumerate(test_loader):
            in_press, out_pressure, time_steps = (
                in_press.to(device),
                out_pressure.to(device),
                time_steps.to(device),
            )

            model_out = model(in_press, time_steps)
            loss_value = mse_loss(model_out, out_pressure)  # 只用MSE
            total_test_loss += loss_value.item()

            with open(loss_log_path, "a") as log_file:
                log_file.write(f"{idx + 1}, {loss_value.item():.6f}\n")

            if vis_enabled and idx < max_samples:
                input_pressure = in_press[0].view(20, 20).cpu().numpy()
                true_pressure = out_pressure[0].view(200, 200).cpu().numpy()
                predicted_pressure = model_out[0].view(200, 200).cpu().numpy()

                plot_comparison_figure(
                    input_pressure=input_pressure,
                    true_pressure=true_pressure,
                    predicted_pressure=predicted_pressure,
                    time_step=time_steps[0].item(),
                    epoch=0,
                    idx=idx,
                    attention_type=attention_type,
                    parent_dir=(
                        parent_dir
                        if parent_dir is not None
                        else f"attention_results/{attention_type}"
                    ),
                    mode="test",
                )

                plot_difference_figure(
                    true_pressure=true_pressure,
                    predicted_pressure=predicted_pressure,
                    time_step=time_steps[0].item(),
                    epoch=0,
                    idx=idx,
                    attention_type=attention_type,
                    parent_dir=(
                        parent_dir
                        if parent_dir is not None
                        else f"attention_results/{attention_type}"
                    ),
                    mode="test",
                )

    avg_test_loss = total_test_loss / len(test_loader)
    logger.info(f"🧪 测试完成，{attention_type} Test Loss: {avg_test_loss:.6f}")

    with open(loss_log_path, "a") as log_file:
        log_file.write(f"Average Test Loss: {avg_test_loss:.6f}\n")

    return avg_test_loss
