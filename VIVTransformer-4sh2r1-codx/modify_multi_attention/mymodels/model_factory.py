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

    if config.get('global', {}).get('use_dataparallel', False) and torch.cuda.device_count() > 1:
        print(f"Using DataParallel on {torch.cuda.device_count()} GPUs!")
        model = torch.nn.DataParallel(model)
    
    return model.to(device)