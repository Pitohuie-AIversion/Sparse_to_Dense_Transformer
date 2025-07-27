"""Utility script to rerun failed attention mechanisms."""

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib
import torch

from .utils.config import load_config
from .utils.system import create_timestamped_dir

from modify_multi_attention_svd10_results.data.dataloader import get_loaders
from modify_multi_attention_svd10_results.mymodels.transformer import (
    TransformerFlowReconstructionModel,
)
from modify_multi_attention_svd10_results.training.trainer import train_model, test_model
from modify_multi_attention_svd10_results.mymodels.components.attention_factory import ATTENTION_MODULES
from modify_multi_attention_svd10_results.utils.visualization import plot_losses

matplotlib.use("Agg")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


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

    return cfg, parent_dir, device

def create_model(cfg, attention_type, device):
    """创建并返回模型"""
    return TransformerFlowReconstructionModel(
        input_dim=cfg["model"]["input_dim"],
        output_dim=cfg["model"]["output_dim"],
        num_heads=cfg["model"]["num_heads"],
        num_layers=cfg["model"]["num_layers"],
        d_model=cfg["model"]["d_model"],
        max_time_steps=cfg["model"]["max_time_steps"],
        attention_type=attention_type
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

    try:
        model = create_model(cfg, attn_type, device)
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(
            model.parameters(), lr=cfg["training"]["learning_rate"]
        )

        result_dir = parent_dir / attn_type
        result_dir.mkdir(exist_ok=True)

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
        print(f"❌ {attn_type} 出现错误，跳过")
        # if "optimizer got an empty parameter list" in str(e):
        #     print("--- Model Structure ---")
        #     print(model)
        #     print("--- Model Parameters ---")
        #     print(list(model.parameters()))
        print(f"⚠️ 错误详情: {str(e)}")
        return False

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

    loaders = get_loaders(cfg["data"]["path"], cfg["data"]["batch_size"])
    
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
