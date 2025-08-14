import torch
import logging
from .transformer import TransformerFlowReconstructionModel

def create_model(config, attention_type, device):
    """Create and configure the Transformer model with performance optimizations."""
    logger = logging.getLogger(__name__)
    
    model = TransformerFlowReconstructionModel(
        input_dim=config['model']['input_dim'],
        output_dim=config['model']['output_dim'],
        num_heads=config['model']['num_heads'],
        num_layers=config['model']['num_layers'],
        d_model=config['model']['d_model'],
        max_time_steps=config['model']['max_time_steps'],
        attention_type=attention_type,
        seq_len=config['model'].get('seq_len', 49),
        grid_height=config['model'].get('grid_height', 7),
        grid_width=config['model'].get('grid_width', 7),
        use_2d_embedding=config['model'].get('use_2d_embedding', True),
    )

    # Move to device first
    model = model.to(device)
    
    # Apply performance optimizations
    perf_config = config.get('performance', {})
    
    # Apply channels_last memory format if enabled and supported
    memory_format_config = perf_config.get('memory_format', {})
    if memory_format_config.get('apply_to_model', False):
        try:
            # Note: channels_last typically works best with Conv2d models
            # For Transformer models, this might not provide benefits
            model = model.to(memory_format=torch.channels_last)
            logger.info("✓ Model converted to channels_last memory format")
        except Exception as e:
            logger.warning(f"Failed to apply channels_last to model: {e}")
    
    # Apply torch.compile if enabled
    compile_config = perf_config.get('torch_compile', {})
    # Force-disable torch.compile to avoid Triton/inductor dependency issues during smoke tests
    compile_enabled = False
    if compile_enabled:
        try:
            compile_mode = compile_config.get('mode', 'default')
            dynamic = compile_config.get('dynamic', False)
            fullgraph = compile_config.get('fullgraph', False)
            backend = compile_config.get('backend', 'inductor')
            
            model = torch.compile(
                model,
                mode=compile_mode,
                dynamic=dynamic,
                fullgraph=fullgraph,
                backend=backend
            )
            logger.info(f"✓ Model compiled with mode={compile_mode}, backend={backend}")
        except Exception as e:
            logger.warning(f"torch.compile failed, falling back to eager mode: {e}")
    
    # Apply DataParallel if enabled
    if config.get('global', {}).get('use_dataparallel', False) and torch.cuda.device_count() > 1:
        logger.info(f"Using DataParallel on {torch.cuda.device_count()} GPUs!")
        model = torch.nn.DataParallel(model)
    
    return model