import torch
import torch.nn as nn
from mymodels.components.attention_factory import get_attention_module
from mymodels.components.attention_adapter import get_attention_adapter

# 测试剩余失败的模块
failed_types = ["gfnet", "s2", "danet", "outlook", "vip"]

for att_type in failed_types:
    print(f"\n=== Testing {att_type} ===")
    try:
        raw_module, adapter_type = get_attention_module(att_type, d_model=256, num_heads=8, spatial_dim=7)
        module = get_attention_adapter(raw_module, adapter_type)
        
        # 序列输入
        x_seq = torch.randn(2, 49, 256)
        print(f"Module type: {type(module)}, Adapter: {adapter_type}")
        
        try:
            out = module(x_seq)
            print(f"Sequence input success: {out.shape}")
        except Exception as e:
            print(f"Sequence input failed: {e}")
            
        # 图像输入 - 测试适配器是否支持
        if hasattr(module, 'attention') and hasattr(module.attention, '__name__'):
            print(f"Core module: {module.attention.__class__.__name__}")
            
    except Exception as e:
        print(f"Module creation failed: {e}")