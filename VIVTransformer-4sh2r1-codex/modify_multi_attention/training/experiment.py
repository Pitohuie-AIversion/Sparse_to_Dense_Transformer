import torch
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

from mymodels.model_factory import create_model
from utils.svd10_loss import TotalLossWithSVD
from training.trainer import train_model, test_model
from utils.visualization import plot_losses
from utils.research_data_collector import ResearchDataCollector


def run_experiment(cfg, loss_cfg, loss_config_id, attn_type, parent_dir, train_loader, valid_loader, test_loader, device, logger, debug=False):
    """Run a single experiment with a given configuration."""
    logger.info(
        "=========== 当前测试注意力机制: %s（%s） ==========",
        attn_type,
        loss_config_id,
    )
    try:
        result_dir = parent_dir / loss_config_id / attn_type
        result_dir.mkdir(parents=True, exist_ok=True)

        model = create_model(cfg, attn_type, device)

        # 读取研究数据收集配置
        research_config = cfg.get("research_data_collection", {})
        collector_enabled = research_config.get("enabled", True)
        
        # 初始化研究数据收集器（基于配置控制功能）
        collector = None
        if collector_enabled:
            # 自动生成实验名称
            experiment_name = research_config.get("experiment_name")
            if experiment_name is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                experiment_name = f"{attn_type}_{timestamp}"
            
            # 创建收集器实例（传入完整配置）
            collector = ResearchDataCollector(result_dir=result_dir, experiment_name=experiment_name, config=research_config)
            
            # 根据配置选择性收集数据
            if research_config.get("collect_system_info", True):
                collector.save_system_info()
            
            # 总是保存实验配置（轻量级操作）
            collector.save_experiment_config(cfg, attention_type=attn_type, loss_config_id=loss_config_id)
            
            if research_config.get("collect_dataset_info", True):
                collector.collect_dataset_info(train_loader, valid_loader, test_loader)

        # Determine whether to enable SVD-based loss from config
        use_svd_loss = cfg.get("loss", {}).get("svd_enabled", True)
        if not use_svd_loss:
            criterion = torch.nn.MSELoss()
        else:
            # Extract grid configuration for optimized SVD loss
            model_cfg = cfg.get("model", {})
            grid_height = model_cfg.get("grid_height", None)
            grid_width = model_cfg.get("grid_width", None)
            
            criterion = TotalLossWithSVD(
                base_weight=loss_cfg.get("base_weight", 0.5),
                svd_weights=loss_cfg.get("svd_weights", None),
                topk=loss_cfg.get("topk", 10),
                grid_height=grid_height,
                grid_width=grid_width,
                svd_enabled=use_svd_loss
            )

        # Optimizer selection with fused AdamW if enabled
        perf_cfg = cfg.get("performance", {})
        opt_cfg = perf_cfg.get("optimizer", {})
        lr = cfg["training"]["learning_rate"]
        use_fused = bool(opt_cfg.get("use_fused", False))
        foreach = bool(opt_cfg.get("foreach", True))
        differentiable = bool(opt_cfg.get("differentiable", False))
        capturable = bool(opt_cfg.get("capturable", False))

        try:
            if use_fused and torch.cuda.is_available():
                optimizer = torch.optim.AdamW(
                    model.parameters(), lr=lr, fused=True, foreach=foreach,
                    differentiable=differentiable, capturable=capturable
                )
            else:
                optimizer = torch.optim.Adam(
                    model.parameters(), lr=lr
                )
        except TypeError:
            # Some PyTorch versions may not support fused=True or foreach
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=lr
            )

        trained_model, train_loss, valid_loss, test_loss = train_model(
            model,
            train_loader,
            valid_loader,
            test_loader,
            criterion,
            optimizer,
            cfg["training"]["epochs"],
            device,
            cfg["training"]["early_stop_patience"],
            attention_type=attn_type,
            result_dir=result_dir,
            cfg=cfg,
            debug=debug,
            collector=collector,  # 传递collector用于轻量级数据收集
        )

        best_model_path = result_dir / f"best_model_{attn_type}.pt"
        torch.save(trained_model.state_dict(), best_model_path)

        if cfg["visualization"]["enabled"]:
            plot_losses(train_loss, valid_loss, test_loss)
            loss_fig_path = result_dir / f"loss_curve_{attn_type}.png"
            plt.savefig(loss_fig_path)
            plt.close()

        final_test_loss = test_model(
            trained_model,
            test_loader,
            criterion,
            device,
            attention_type=attn_type,
            parent_dir=result_dir,
            cfg=cfg,
            debug=debug,
        )  # 保持test_model签名不变，collector仅在训练阶段使用

        test_result_file = result_dir / f"test_result_{attn_type}.txt"
        with open(test_result_file, "w") as f:
            f.write(f"Final Test Loss: {final_test_loss}")

        # 结束时保存收敛性与总结（基于配置控制）
        if collector is not None:
            try:
                collector.finalize_experiment(attention_type=attn_type, final_test_loss=final_test_loss)
            except Exception as e:
                logger.warning(f"研究数据收集器完成实验时出错: {e}")
                pass

        logger.info("=========== 注意力机制 %s 测试完成 ==========", attn_type)

        min_valid_loss = min(valid_loss) if valid_loss else float('inf')
        return None, min_valid_loss

    except Exception as e:
        logger.error("运行注意力机制 %s 时出错: %s", attn_type, e, exc_info=True)
        return attn_type, float('inf')