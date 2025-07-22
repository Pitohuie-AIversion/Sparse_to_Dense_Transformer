"""Train and evaluate the VIVTransformer with various attention mechanisms.

This script loads configuration from a YAML file and allows overriding the
location of the output results directory via command-line arguments.
"""

import argparse
import os
import sys
from pathlib import Path
import logging

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from data.dataloader import get_loaders, get_adaptive_loaders
from mymodels.model_factory import create_model
from training.experiment import run_experiment
from utils.config import load_config
from utils.logger import setup_logger
from utils.svd10_loss import TotalLossWithSVD
from utils.system import set_seed, set_cuda_memory_limit

matplotlib.use("Agg")

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def parse_loss_idx_from_argv():
    for i, arg in enumerate(sys.argv):
        if arg == "--loss_idx" and i + 1 < len(sys.argv):
            return int(sys.argv[i + 1])
    return None


def main(config_path=None):
    this_file = Path(__file__).resolve()
    project_root = this_file.parent.parent

    parser = argparse.ArgumentParser(description="Train VIVTransformer")
    parser.add_argument(
        "-c",
        "--config",
        default=config_path if config_path else str(
            project_root / "modify_multi_attention" / "configs" / "config.yaml"
        ),
        help="Path to config file",
    )
    parser.add_argument(
        "-r",
        "--results-dir",
        type=Path,
        default=project_root / "attention_results",
        help="Directory to save training results",
    )
    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining

    config_path = Path(args.config)
    parent_dir = Path(args.results_dir)
    parent_dir.mkdir(exist_ok=True)

    # Setup logger
    logger = setup_logger(log_dir=parent_dir)

    logger.info("加载配置文件: %s", config_path)

    cfg = load_config(config_path)

    set_seed(cfg["global"].get("seed", 42), cfg["global"].get("deterministic", False))
    if "max_memory_fraction" in cfg["global"]:
        device_idx = int(str(cfg["global"]["device"]).split(":")[-1])
        set_cuda_memory_limit(cfg["global"]["max_memory_fraction"], device_idx)
    device = torch.device(cfg["global"]["device"])
    logger.info("Using device: %s", device)
    logger.info("Available GPUs: %s", torch.cuda.device_count())

    import yaml

    loss_configs_dir = config_path.parent / "loss_configs"
    all_loss_configs = []
    if loss_configs_dir.is_dir():
        loss_config_files = sorted(loss_configs_dir.glob("*.yaml"))
        logger.info("Found %d loss configurations in %s.", len(loss_config_files), loss_configs_dir)
        for f_path in loss_config_files:
            with f_path.open('r', encoding='utf-8') as f:
                all_loss_configs.append(yaml.safe_load(f))
    else:
        logger.warning(
            "`loss_configs` directory not found at %s. "
            "Falling back to `loss_configs` key in main config file.", 
            loss_configs_dir
        )
        all_loss_configs = cfg.get("loss_configs", [])

    if not all_loss_configs:
        logger.error("No loss configurations found. Exiting.")
        sys.exit(1)

    loss_idx = parse_loss_idx_from_argv()
    if loss_idx is not None:
        if loss_idx < len(all_loss_configs):
            logger.info("只运行 loss_config_%s", loss_idx)
            loss_configs = [all_loss_configs[loss_idx]]
            loss_config_ids = [f"loss_config_{loss_idx}"]
        else:
            logger.error("Error: loss_idx %d is out of range. Found %d configs.", loss_idx, len(all_loss_configs))
            sys.exit(1)
    else:
        loss_configs = all_loss_configs
        loss_config_ids = [f"loss_config_{i}" for i in range(len(loss_configs))]

    ATTENTION_TYPES = cfg["attention_test"]["types"]
    failed_attention_types = []

    # 使用自适应数据加载器，支持PDEBench数据集
    train_loader, valid_loader, test_loader = get_adaptive_loaders(
        cfg,
        batch_size=cfg["data"]["batch_size"],
        use_augmentation=cfg["data"].get("use_augmentation", False)
    )
    logger.info("Data loaders created successfully.")

    for loss_cfg, loss_config_id in zip(loss_configs, loss_config_ids):
        logger.info(
            "===== 当前loss设置 [%s]: base_weight=%s, svd_weights=%s, topk=%s =====",
            loss_config_id,
            loss_cfg.get("base_weight", 0.5),
            loss_cfg.get("svd_weights", None),
            loss_cfg.get("topk", 10),
        )

        logger.info("Starting experiments...")
        for attn_type in ATTENTION_TYPES:
            failed_attn_type, min_valid_loss = run_experiment(
                cfg, loss_cfg, loss_config_id, attn_type, parent_dir, 
                train_loader, valid_loader, test_loader, device, logger,
                debug=True
            )
            if failed_attn_type:
                failed_attention_types.append((failed_attn_type, min_valid_loss))

    if failed_attention_types:
        with open(parent_dir / "failed_attention_log.txt", "w") as f:
            for info in failed_attention_types:
                f.write(f"{info}\n")
        logger.warning(
            "\n⚠️ 以下loss+注意力机制训练失败，并已记录在 failed_attention_log.txt："
        )
        for info in failed_attention_types:
            logger.warning(info)
    else:
        logger.info("\n🎉 所有loss配置和注意力机制均运行成功！")

    # This part is tricky because we run multiple experiments.
    # For hyperparameter tuning, we should probably focus on a single experiment run.
    # Let's assume for now we return the result of the first experiment.
    # A more sophisticated approach would be needed for a real-world scenario.
    # We'll need to modify `run_experiment` to return the value we want to optimize.
    # For now, let's return a placeholder. We will adjust this later.
    # A better approach would be to refactor the main loop to run only one experiment for tuning.
    return 0.0 # Placeholder


if __name__ == "__main__":
    main()
