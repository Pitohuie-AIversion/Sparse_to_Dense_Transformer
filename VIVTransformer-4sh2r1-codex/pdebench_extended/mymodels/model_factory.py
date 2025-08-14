import torch
from mymodels.transformer import TransformerFlowReconstructionModel

def create_model(config, attention_type, device):
    """Create and configure the Transformer model."""
    model = TransformerFlowReconstructionModel(
        input_dim=config['model']['input_dim'],
        output_dim=config['model']['output_dim'],
        num_heads=config['model']['num_heads'],
        num_layers=config['model']['num_layers'],
        d_model=config['model']['d_model'],
        max_time_steps=config['model']['max_time_steps'],
        attention_type=attention_type,
        seq_len=config['model'].get('seq_len', 49),
    )

    if config.get('use_dataparallel', False) and torch.cuda.device_count() > 1:
        # 如果模型包含被标记为 _force_cpu 的子模块，则不要使用 DataParallel
        has_force_cpu_modules = any(
            getattr(m, '_force_cpu', False) for _, m in model.named_modules()
        )
        if has_force_cpu_modules:
            print("[WARN] 模型包含强制CPU子模块，跳过DataParallel以避免设备不一致")
        else:
            print(f"Using DataParallel on {torch.cuda.device_count()} GPUs!")
            model = torch.nn.DataParallel(model)
    
    return model.to(device)