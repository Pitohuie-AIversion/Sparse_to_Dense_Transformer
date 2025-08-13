import torch
import matplotlib.pyplot as plt
from pathlib import Path

from mymodels.model_factory import create_model
from utils.svd10_loss import TotalLossWithSVD
from training.trainer import train_model, test_model
from utils.visualization import plot_losses

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

        # Determine whether to enable SVD-based loss from config (default True)
        use_svd_loss = cfg.get("loss", {}).get("svd_enabled", True)
        if not use_svd_loss:
            criterion = torch.nn.MSELoss()
        else:
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

        optimizer = torch.optim.Adam(
            model.parameters(), lr=cfg["training"]["learning_rate"]
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
        )

        test_result_file = result_dir / f"test_result_{attn_type}.txt"
        with open(test_result_file, "w") as f:
            f.write(f"Final Test Loss: {final_test_loss}")

        logger.info("=========== 注意力机制 %s 测试完成 ==========", attn_type)

        min_valid_loss = min(valid_loss) if valid_loss else float('inf')
        return None, min_valid_loss

    except Exception as e:
        logger.error("运行注意力机制 %s 时出错: %s", attn_type, e, exc_info=True)
        return attn_type, float('inf')