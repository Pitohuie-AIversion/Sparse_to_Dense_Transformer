import logging
import os
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import torch
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from utils.visualization import (
    plot_losses,
    plot_attention_maps,
    plot_comparison_figure,
    plot_difference_figure,
)
from utils.hardware_monitor import HardwareMonitor
from utils.training_visualizer import TrainingVisualizer


def _train_epoch(
    model, loader, criterion, optimizer, device, debug, epoch, attention_type, result_dir, hardware_monitor=None, max_batches=None, use_amp: bool = False, scaler: Optional[torch.cuda.amp.GradScaler] = None
):
    model.train()
    total_loss = 0
    logger = logging.getLogger(__name__)
    autocast_ctx = (lambda: torch.amp.autocast('cuda')) if use_amp else nullcontext
    processed_batches = 0
    
    data_start = time.time()
    for i, batch in enumerate(loader):
        iter_start = time.time()
        data_time = iter_start - data_start
        if batch is None:
            data_start = time.time()
            continue
        in_press, out_pressure, time_steps = batch
        current_bs = in_press.size(0) if hasattr(in_press, 'size') else None
        # Early exit if max_batches limit reached
        if max_batches is not None and i >= max_batches:
            break
            
        # 开始batch监控
        if hardware_monitor:
            hardware_monitor.start_batch(epoch, i)
        
        copy_start = time.time()
        in_press, out_pressure, time_steps = (
            in_press.to(device, non_blocking=True),
            out_pressure.to(device, non_blocking=True),
            time_steps.to(device, non_blocking=True),
        )
        copy_time = time.time() - copy_start
        optimizer.zero_grad(set_to_none=True)
        comp_start = time.time()
        with autocast_ctx():
            if debug:
                model_out, attention_weights = model(
                    in_press, time_steps, return_attention=True
                )
                # 仅在处理到的首个有效batch进行一次可视化
                if processed_batches == 0:
                    plot_attention_maps(
                        attention_weights,
                        epoch=epoch,
                        attention_type=attention_type,
                        parent_dir=result_dir,
                    )
            else:
                model_out = model(in_press, time_steps)
            loss = criterion(model_out, out_pressure)
        
        if use_amp and scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        compute_time = time.time() - comp_start
        total_iter_time = time.time() - iter_start
        total_loss += loss.item()
        processed_batches += 1
        
        # 结束batch监控
        if hardware_monitor:
            hardware_monitor.end_batch(epoch, i, loss.item())
        
        # 日志：每个batch显示BS、data_time、h2d(copy_time)、compute_time、total_time、samples/s
        if (i + 1) % 50 == 0 or i == 0:
            samples_per_sec = (current_bs / total_iter_time) if current_bs else float('nan')
            logger.info(
                f"    🔄 Epoch [{epoch + 1}], Batch [{i + 1}/{len(loader)}], "
                f"BS: {current_bs}, Loss: {loss.item():.6f}, "
                f"data: {data_time*1000:.1f} ms, h2d: {copy_time*1000:.1f} ms, "
                f"compute: {compute_time*1000:.1f} ms, total: {total_iter_time*1000:.1f} ms, "
                f"samples/s: {samples_per_sec:.1f}"
            )
        
        # 准备下一次迭代的data计时起点
        data_start = time.time()
    
    # 使用已处理的有效batch数计算平均损失；若为0则返回inf避免除零
    return (total_loss / processed_batches) if processed_batches > 0 else float("inf")

def _evaluate_model(model, loader, criterion, device, debug, epoch, attention_type, save_dir, max_batches=None, use_amp: bool = False):
    model.eval()
    total_loss = 0
    autocast_ctx = (lambda: torch.amp.autocast('cuda')) if use_amp else nullcontext
    processed_batches = 0
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if batch is None:
                continue
            in_press, out_pressure, time_steps = batch
            # Early exit if max_batches limit reached
            if max_batches is not None and i >= max_batches:
                break
                
            in_press, out_pressure, time_steps = (
                in_press.to(device, non_blocking=True),
                out_pressure.to(device, non_blocking=True),
                time_steps.to(device, non_blocking=True),
            )
            with autocast_ctx():
                if debug:
                    model_out, attention_weights = model(
                        in_press, time_steps, return_attention=True
                    )
                    # 仅在处理到的首个有效batch（且满足可视化周期）进行一次可视化
                    if processed_batches == 0 and (epoch + 1) % 100 == 0:
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
            processed_batches += 1
            
    # 使用已处理的有效batch数计算平均损失；若为0则返回inf避免除零
    return (total_loss / processed_batches) if processed_batches > 0 else float("inf")

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
    collector=None,  # 可选的研究数据收集器，轻量级集成
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

    # 初始化硬件监控器
    hardware_monitor = HardwareMonitor(
        log_dir=str(result_dir / "hardware_logs"),
        enable_gpu_monitoring=cfg.get("hardware_monitoring", {}).get("enable_gpu_monitoring", True)
    )
    hardware_monitor.start_training()

    # 标准 MSELoss（保证横向可比）
    import torch.nn as nn

    mse_loss = nn.MSELoss()

    # 配置参数直接来自cfg
    vis_config = cfg.get("visualization", {})
    vis_enabled = vis_config.get("enabled", False)
    vis_interval = vis_config.get("interval", 100)
    max_samples = vis_config.get("max_samples", 5)
    
    # Add max_batches parameter for fast mode
    training_config = cfg.get("training", {})
    max_batches_per_epoch = training_config.get("max_batches_per_epoch", None)
    use_amp = bool(training_config.get("use_amp", False)) and str(device).startswith("cuda")
    try:
        scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    except Exception:
        # Fallback for environments where torch.amp.GradScaler signature/device arg is unsupported
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

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
    checkpoint_path = result_dir / "checkpoint_{}.pth".format(attention_type)

    logger.info("写入loss_log.txt到：{}".format(loss_log_path))

    # ========== 恢复断点 ==========
    start_epoch = 0
    if os.path.exists(checkpoint_path):
        logger.info("检测到断点文件，自动恢复：{}".format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        train_loss_history = checkpoint.get("train_loss_history", [])
        valid_loss_history = checkpoint.get("valid_loss_history", [])
        test_loss_history = checkpoint.get("test_loss_history", [])
        best_valid_loss = checkpoint.get("best_valid_loss", float("inf"))
        patience_counter = checkpoint.get("patience_counter", 0)
        start_epoch = checkpoint.get("epoch", 0) + 1
        logger.info("已恢复到 epoch {}，best_valid_loss={}".format(start_epoch, best_valid_loss))
    else:
        with open(loss_log_path, "w") as log_file:
            log_file.write("Epoch, Train Loss, Valid Loss, Test Loss\n")

    # 训练前轻量收集模型复杂度（基于配置控制）
    try:
        if collector is not None:
            research_config = cfg.get("research_data_collection", {})
            if research_config.get("collect_model_complexity", True):
                sample_batch = next(iter(train_loader))
                sample_input = sample_batch[0].to(device) if isinstance(sample_batch, (list, tuple)) else sample_batch.to(device)
                # 仅使用数据形状，避免多余前向开销
                collector.collect_model_complexity(model, sample_input, attention_type)
    except Exception:
        pass

    # ========== 主训练循环 ==========
    for epoch in range(start_epoch, num_epochs):
        # 开始epoch监控
        hardware_monitor.start_epoch(epoch + 1)
        
        avg_train_loss = _train_epoch(
            model, train_loader, criterion, optimizer, device, debug, epoch, attention_type, result_dir, hardware_monitor, max_batches_per_epoch, use_amp, scaler
        )
        train_loss_history.append(avg_train_loss)
        writer.add_scalar('Loss/train', avg_train_loss, epoch)
        
        avg_valid_loss = _evaluate_model(
            model, valid_loader, mse_loss, device, debug, epoch, attention_type, save_dir, max_batches_per_epoch, use_amp
        )
        valid_loss_history.append(avg_valid_loss)
        writer.add_scalar('Loss/valid', avg_valid_loss, epoch)

        avg_test_loss = _evaluate_model(model, test_loader, mse_loss, device, False, 0, '', '', max_batches_per_epoch, use_amp)
        test_loss_history.append(avg_test_loss)
        writer.add_scalar('Loss/test', avg_test_loss, epoch)

        # 记录训练指标（在评估完成后，避免频繁IO；基于配置控制频率）
        try:
            if collector is not None:
                research_config = cfg.get("research_data_collection", {})
                log_every_n = max(1, int(research_config.get("log_every_n_epochs", 1)))
                if (epoch + 1) % log_every_n == 0 and research_config.get("collect_training_metrics", True):
                    current_lr = None
                    try:
                        # 获取当前学习率
                        current_lr = optimizer.param_groups[0].get('lr', None)
                    except Exception:
                        current_lr = None
                    collector.log_training_metrics(epoch + 1, avg_train_loss, avg_valid_loss, avg_test_loss, current_lr)
        except Exception:
            pass

        # 结束epoch监控
        hardware_monitor.end_epoch(epoch + 1, avg_train_loss, avg_valid_loss, avg_test_loss)

        logger.info(
            "🎯 Epoch [{}/{}], Train Loss: {:.6f}, Valid Loss: {:.6f}, Test Loss: {:.6f}".format(
                epoch + 1, num_epochs, avg_train_loss, avg_valid_loss, avg_test_loss
            )
        )

        with open(loss_log_path, "a") as log_file:
            log_file.write(
                "{}, {:.6f}, {:.6f}, {:.6f}\n".format(
                    epoch + 1, avg_train_loss, avg_valid_loss, avg_test_loss
                )
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
            logger.warning("⚠️ 早停计数: {}/{}".format(patience_counter, early_stop_patience))

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
            loss_fig_path = plot_dir / "loss_curve_epoch_{}.png".format(epoch + 1)
            plot_losses(
                train_loss_history,
                valid_loss_history,
                test_loss_history,
                save_path=loss_fig_path,
            )

    # 结束训练监控
    hardware_monitor.end_training()
    
    # 记录训练摘要统计
    summary_stats = hardware_monitor.get_summary_stats()
    logger.info("📊 训练统计摘要:")
    for key, value in summary_stats.items():
        logger.info(f"  {key}: {value}")
    
    # 生成训练可视化报告
    try:
        visualizer = TrainingVisualizer(str(result_dir / "hardware_logs"))
        visualizer.generate_complete_report()
        logger.info(f"训练可视化报告已生成到: {result_dir / 'hardware_logs' / 'visualizations'}")
    except Exception as e:
        logger.warning(f"生成训练可视化报告失败: {e}")
    
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
