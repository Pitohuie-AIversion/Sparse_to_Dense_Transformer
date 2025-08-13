"""Utility script to rerun failed attention mechanisms."""

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib
import torch

from .utils.config import load_config
from .utils.system import create_timestamped_dir, set_cuda_memory_limit, apply_performance_optimizations

from .data.dataloader import get_adaptive_loaders
from .mymodels.transformer import (
    TransformerFlowReconstructionModel,
)
from .training.trainer import train_model, test_model
from .mymodels.components.attention_factory import ATTENTION_MODULES
from .utils.visualization import plot_losses

matplotlib.use("Agg")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# 减少 CUDA allocator 分片导致的 OOM 概率（需在首次使用 CUDA 前设置）
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
# 使用更好的GPU内存管理策略，启用内存池回收
os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "0")  # 改为0提高性能
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:64,expandable_segments:True,roundup_power2_divisions:16")
# 启用GPU内存缓存释放策略
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "garbage_collection_threshold:0.6,max_split_size_mb:64")


def parse_arguments():
    """解析命令行参数"""
    current_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Retest failed attention types")
    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        default=current_dir / "configs" / "config.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "-r",
        "--results-dir",
        type=Path,
        default=current_dir / "attention_results",
        help="Directory to save retest results",
    )
    return parser.parse_args()

def setup_environment(args):
    """加载配置、设置目录和设备"""
    try:
        cfg = load_config(args.config)
    except FileNotFoundError:
        print(f"❌ 配置文件未找到: {args.config}")
        return None, None, None

    parent_dir = create_timestamped_dir(args.results_dir)
    print(f"📂 Results will be saved in: {parent_dir}")

    device = torch.device(cfg["global"]["device"])
    print(f"🖥️ Using device: {device}")
    
    # 优化GPU内存管理
    if device.type == "cuda":
        # 设置内存池的最大分配比例
        try:
            torch.cuda.set_per_process_memory_fraction(0.85)  # 留15%给系统和其他进程
            torch.cuda.empty_cache()
            print(f"✓ 设置GPU内存分配比例为85%")
        except Exception as e:
            print(f"⚠️ GPU内存设置失败: {e}")
    
    # 新增：显式打印本次使用的配置文件与 epochs 设置
    print(f"✅ Loaded config file: {args.config}")
    print(f"🗓️ Training epochs: {cfg['training']['epochs']}, early_stop_patience: {cfg['training']['early_stop_patience']}")

    return cfg, parent_dir, device

def create_model(cfg, attention_type, device):
    """创建并返回模型"""
    low_memory = cfg.get("performance", {}).get("low_memory_mode", True)
    return TransformerFlowReconstructionModel(
        input_dim=cfg["model"]["input_dim"],
        output_dim=cfg["model"]["output_dim"],
        num_heads=cfg["model"]["num_heads"],
        num_layers=cfg["model"]["num_layers"],
        d_model=cfg["model"]["d_model"],
        max_time_steps=cfg["model"]["max_time_steps"],
        attention_type=attention_type,
        seq_len=cfg["model"].get("seq_len", 49),
        grid_height=cfg["model"].get("grid_height", 7),
        grid_width=cfg["model"].get("grid_width", 7),
        use_2d_embedding=cfg["model"].get("use_2d_embedding", True),
        low_memory=low_memory,
    ).to(device)

def save_model(model, path):
    """保存模型状态"""
    torch.save(model.state_dict(), path)

def plot_and_save_losses(train_loss, valid_loss, test_loss, path):
    """绘制并保存损失曲线"""
    plot_losses(train_loss, valid_loss, test_loss)
    plt.savefig(path)
    plt.close()

def save_test_results(attn_type, loss, path):
    """保存测试结果"""
    with open(path, "w") as f:
        f.write(f"Test Loss for {attn_type}: {loss}\n")

def run_attention_trial(attn_type, cfg, loaders, device, parent_dir):
    """对单个 attention 类型进行训练和测试"""
    train_loader, valid_loader, test_loader = loaders
    vis_enabled = cfg.get("visualization", {}).get("enabled", False)

    # 对于特定的重量级attention，启用强制低内存模式和CPU回退
    heavy_attentions = ["emsa", "crisscross", "psa", "danet"]
    force_cpu_fallback = attn_type in heavy_attentions
    
    if force_cpu_fallback:
        print(f"⚠️ {attn_type} 被识别为重量级模块，启用强制CPU回退模式")
        # 临时降低显存限制至50%，强制更激进的内存管理
        if torch.cuda.is_available():
            try:
                torch.cuda.set_per_process_memory_fraction(0.5)
                torch.cuda.empty_cache()
                print(f"✓ 临时降低显存限制至50%")
            except Exception as e:
                print(f"⚠️ 显存限制调整失败: {e}")

    model = None
    try:
        model = create_model(cfg, attn_type, device)
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(
            model.parameters(), lr=cfg["training"]["learning_rate"]
        )

        result_dir = parent_dir / attn_type
        result_dir.mkdir(exist_ok=True)
        # 新增：在进入训练前打印关键信息，便于核对 epoch 与输出目录
        print(f"➡️ Start training {attn_type} for {cfg['training']['epochs']} epochs, result_dir: {result_dir}")

        trained_model, train_loss, valid_loss, test_loss = train_model(
            model, train_loader, valid_loader, test_loader, criterion, optimizer,
            cfg["training"]["epochs"], device, cfg["training"]["early_stop_patience"],
            result_dir=result_dir, cfg=cfg, attention_type=attn_type
        )

        save_model(trained_model, result_dir / f"best_model_{attn_type}.pt")

        if vis_enabled:
            plot_and_save_losses(
                train_loss, valid_loss, test_loss, 
                result_dir / f"loss_curve_{attn_type}.png"
            )

        final_test_loss = test_model(
            trained_model, test_loader, criterion, device,
            attention_type=attn_type, parent_dir=parent_dir, cfg=cfg
        )

        save_test_results(
            attn_type, final_test_loss, 
            result_dir / f"test_result_{attn_type}.txt"
        )

        print(f"✅ {attn_type} 测试完成！")
        return True

    except Exception as e:
        # 若显存不足，尝试自动降批量重试一次
        if isinstance(e, torch.cuda.OutOfMemoryError) or ("out of memory" in str(e).lower() and "cuda" in str(e).lower()):
            print(f"⚠️ {attn_type} 触发 CUDA OOM，尝试使用更小批量重试 (batch_size=1)...")
            try:
                # 释放当前模型与显存
                try:
                    if model is not None:
                        del model
                except Exception:
                    pass
                if torch.cuda.is_available():
                    try:
                        torch.cuda.empty_cache()
                    except Exception:
                        pass
                
                # 构造降批量配置与数据加载器
                cfg_small = dict(cfg)
                cfg_small_data = dict(cfg_small.get("data", {}))
                cfg_small_data["batch_size"] = 1
                cfg_small["data"] = cfg_small_data
                small_loaders = get_adaptive_loaders(cfg_small, batch_size=1)
                small_train_loader, small_valid_loader, small_test_loader = small_loaders
                
                # 重新构建模型并训练
                model = create_model(cfg_small, attn_type, device)
                criterion = torch.nn.MSELoss()
                optimizer = torch.optim.Adam(model.parameters(), lr=cfg_small["training"]["learning_rate"])
                result_dir = Path(parent_dir) / (str(attn_type) + "_retry_bs1")
                result_dir.mkdir(exist_ok=True)
                print(f"🔁 Retry {attn_type} with batch_size=1, result_dir: {result_dir}")
                
                trained_model, train_loss, valid_loss, test_loss = train_model(
                    model, small_train_loader, small_valid_loader, small_test_loader, criterion, optimizer,
                    cfg_small["training"]["epochs"], device, cfg_small["training"]["early_stop_patience"],
                    result_dir=result_dir, cfg=cfg_small, attention_type=attn_type
                )
                
                save_model(trained_model, result_dir / f"best_model_{attn_type}.pt")
                final_test_loss = test_model(
                    trained_model, small_test_loader, criterion, device,
                    attention_type=attn_type, parent_dir=parent_dir, cfg=cfg_small
                )
                save_test_results(attn_type, final_test_loss, result_dir / f"test_result_{attn_type}.txt")
                print(f"✅ {attn_type} 降批量重试成功！")
                return True
            except Exception as e2:
                print(f"❌ {attn_type} 降批量重试仍失败：{e2}")
                return False
        else:
            print(f"❌ {attn_type} 出现错误，跳过")
            print(f"⚠️ 错误详情: {str(e)}")
            return False
    finally:
        # 清理显存，避免不同 trial 之间的内存累积
        try:
            if model is not None:
                del model
        except Exception:
            pass
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

def log_failed_attentions(failed_list, log_path):
    """记录失败的 attention 类型"""
    if failed_list:
        with open(log_path, "w") as f:
            for attn in failed_list:
                f.write(f"{attn}\n")
        print("\n⚠️ 以下 Attention 机制仍失败，log 已更新：")
        print("\n".join(failed_list))
    else:
        print("\n🎉 本次所有 Attention 均已成功")
        if log_path.exists():
            log_path.unlink()
            print("✅ 已清除 failed_attention_log.txt")

def run_all_trials(cfg, loaders, device, parent_dir):
    """运行所有指定的 attention 类型的测试"""
    attention_types = cfg.get("attention_test", {}).get("types", [])
    if not attention_types:
        print("\n⚠️ 在 config.yaml 中没有找到要测试的 attention 类型 (attention_test.types)，退出")
        return []
    print(f"🔁 本次将测试以下 attention: {attention_types}")

    failed_attention_types = []
    for attn_type in attention_types:
        if attn_type not in ATTENTION_MODULES:
            print(f"⚠️ 注意力机制 '{attn_type}' 在 attention_factory.py 中未定义，跳过测试。")
            continue
        print(f"\n=========== 测试注意力机制: {attn_type} ===========")
        success = run_attention_trial(attn_type, cfg, loaders, device, parent_dir)
        if not success:
            failed_attention_types.append(attn_type)
    return failed_attention_types

def main():
    """主函数，协调整个测试流程"""
    args = parse_arguments()
    cfg, parent_dir, device = setup_environment(args)
    if not cfg:
        return
    
    # 在设备初始化之后立即应用显存和性能优化
    import logging
    logger = logging.getLogger(__name__)
    
    # 应用显存限制
    memory_fraction = cfg["global"].get("max_memory_fraction", 0.8)
    if torch.cuda.is_available():
        try:
            set_cuda_memory_limit(memory_fraction)
            logger.info(f"✓ CUDA 显存限制设为 {memory_fraction*100:.0f}%")
        except Exception as e:
            logger.warning(f"⚠️ 显存限制设置失败: {e}")
    
    # 应用性能优化（如 TF32、AMP 等）
    try:
        perf_config = cfg.get("performance", {})
        if not perf_config:
            # 如果配置中没有 performance 节，使用基础优化
            perf_config = {
                "hardware": {
                    "enable_tf32": True,
                    "enable_cudnn_benchmark": True
                }
            }
        apply_performance_optimizations(perf_config, device)
        logger.info("✓ 性能优化已应用")
    except Exception as e:
        logger.warning(f"⚠️ 性能优化应用失败: {e}")

    loaders = get_adaptive_loaders(cfg, batch_size=cfg["data"]["batch_size"])
    
    parent_dir = Path(parent_dir)
    failed_attention_types = run_all_trials(cfg, loaders, device, parent_dir)

    failed_log_path = parent_dir / "failed_attention_log.txt"
    log_failed_attentions(failed_attention_types, failed_log_path)

    # === 更新失败记录 ===
    if failed_attention_types:
        with open(failed_log_path, "w") as f:
            for attn in failed_attention_types:
                f.write(f"{attn}\n")
        print("\n⚠️ 以下 Attention 机制仍失败，log 已更新：")
        print("\n".join(failed_attention_types))
    else:
        print("\n🎉 本次所有 Attention 均已成功")
        if failed_log_path.exists():
            failed_log_path.unlink()
            print("✅ 已清除 failed_attention_log.txt")


if __name__ == "__main__":
    main()
