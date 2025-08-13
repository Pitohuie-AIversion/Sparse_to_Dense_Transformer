import torch
import torch.nn as nn
import traceback
from mymodels.components.attention_factory import get_attention_module
from mymodels.components.attention_adapter import get_attention_adapter
from fightingcv_attention.attention.OutlookAttention import OutlookAttention
from fightingcv_attention.attention.ViP import WeightedPermuteMLP
from fightingcv_attention.attention.S2Attention import S2Attention
from fightingcv_attention.attention.DANet import DAModule

def test_outlook_attention():
    """测试 OutlookAttention 的不同构造参数组合"""
    print("="*60)
    print("🔧 Testing OutlookAttention configurations...")
    
    d_model = 256
    batch_size = 2
    seq_len = 49  # 7x7
    
    # 测试序列输入
    x = torch.randn(batch_size, seq_len, d_model)
    
    configs = [
        {"dim": d_model},
        {"dim": d_model, "num_heads": 4},
        {"dim": d_model, "num_heads": 4, "kernel_size": 3},
        {"dim": d_model, "num_heads": 4, "kernel_size": 3, "padding": 1, "stride": 1},
        {"dim": d_model, "num_heads": 8, "kernel_size": 3, "padding": 1, "stride": 1},
        {d_model},  # 位置参数
        {d_model, 4, 3, 1, 1},  # 位置参数 dim, num_heads, kernel_size, padding, stride
    ]
    
    for i, config in enumerate(configs):
        try:
            print(f"  Config {i+1}: {config}")
            if isinstance(config, dict):
                outlook = OutlookAttention(**config)
            else:
                outlook = OutlookAttention(*config)
            
            # 测试序列输入（需要转换为 4D）
            B, N, C = x.shape
            H = W = int(N**0.5)
            x_4d = x.view(B, H, W, C)  # BHWC 格式
            
            print(f"    Input shape: {x_4d.shape}")
            output = outlook(x_4d)
            print(f"    Output shape: {output.shape}")
            print(f"    ✅ Config {i+1} succeeded")
            return config, outlook
            
        except Exception as e:
            print(f"    ❌ Config {i+1} failed: {e}")
    
    return None, None

def test_vip_mlp():
    """测试 WeightedPermuteMLP 的不同构造参数组合"""
    print("="*60)
    print("🔧 Testing WeightedPermuteMLP configurations...")
    
    d_model = 256
    batch_size = 2
    seq_len = 49  # 7x7
    
    # 测试序列输入
    x = torch.randn(batch_size, seq_len, d_model)
    
    # 找到能整除 d_model 的分段
    valid_segs = [s for s in [1,2,4,8,16,32,64] if d_model % s == 0]
    print(f"  Valid seg_dims for d_model={d_model}: {valid_segs}")
    
    configs = [
        {"dim": d_model, "seg_dim": valid_segs[0] if valid_segs else 1},
        {"dim": d_model, "seg_dim": valid_segs[1] if len(valid_segs) > 1 else valid_segs[0]},
        {d_model, valid_segs[0] if valid_segs else 1},  # 位置参数
    ]
    
    for i, config in enumerate(configs):
        try:
            print(f"  Config {i+1}: {config}")
            if isinstance(config, dict):
                vip = WeightedPermuteMLP(**config)
            else:
                vip = WeightedPermuteMLP(*config)
            
            # 测试序列输入（需要转换为 4D）
            B, N, C = x.shape
            H = W = int(N**0.5)
            x_4d = x.view(B, H, W, C)  # BHWC 格式
            
            print(f"    Input shape: {x_4d.shape}")
            output = vip(x_4d)
            print(f"    Output shape: {output.shape}")
            print(f"    ✅ Config {i+1} succeeded")
            return config, vip
            
        except Exception as e:
            print(f"    ❌ Config {i+1} failed: {e}")
    
    return None, None

def test_s2_attention():
    """测试 S2Attention 的不同构造参数组合"""
    print("="*60)
    print("🔧 Testing S2Attention configurations...")
    
    d_model = 256
    batch_size = 2
    spatial_dim = 7
    
    # CNN 输入
    x_cnn = torch.randn(batch_size, d_model, spatial_dim, spatial_dim)
    
    configs = [
        {"channels": d_model},
        {"channels": d_model, "heads": 4},
        {"channels": d_model, "heads": 8},
        {d_model},  # 位置参数
        {d_model, 4},  # 位置参数 channels, heads
    ]
    
    for i, config in enumerate(configs):
        try:
            print(f"  Config {i+1}: {config}")
            if isinstance(config, dict):
                s2 = S2Attention(**config)
            else:
                s2 = S2Attention(*config)
            
            print(f"    Input shape: {x_cnn.shape}")
            output = s2(x_cnn)
            print(f"    Output type: {type(output)}")
            if isinstance(output, (tuple, list)):
                print(f"    Output tuple/list length: {len(output)}")
                if len(output) > 0:
                    print(f"    First element shape: {output[0].shape}")
            else:
                print(f"    Output shape: {output.shape}")
            print(f"    ✅ Config {i+1} succeeded")
            return config, s2
            
        except Exception as e:
            print(f"    ❌ Config {i+1} failed: {e}")
    
    return None, None

def test_danet_module():
    """测试 DAModule 的不同构造参数组合"""
    print("="*60)
    print("🔧 Testing DAModule configurations...")
    
    d_model = 256
    batch_size = 2
    spatial_dim = 7
    
    # CNN 输入
    x_cnn = torch.randn(batch_size, d_model, spatial_dim, spatial_dim)
    
    configs = [
        {"in_channels": d_model},
        {d_model},  # 位置参数
    ]
    
    for i, config in enumerate(configs):
        try:
            print(f"  Config {i+1}: {config}")
            if isinstance(config, dict):
                danet = DAModule(**config)
            else:
                danet = DAModule(*config)
            
            print(f"    Input shape: {x_cnn.shape}")
            output = danet(x_cnn)
            print(f"    Output type: {type(output)}")
            if isinstance(output, (tuple, list)):
                print(f"    Output tuple/list length: {len(output)}")
                if len(output) > 0:
                    print(f"    First element shape: {output[0].shape}")
            else:
                print(f"    Output shape: {output.shape}")
            print(f"    ✅ Config {i+1} succeeded")
            return config, danet
            
        except Exception as e:
            print(f"    ❌ Config {i+1} failed: {e}")
    
    return None, None

def main():
    print("🚀 开始逐一恢复特定注意力模块...")
    
    # 测试每个模块
    results = {}
    
    # 1. OutlookAttention
    outlook_config, outlook_module = test_outlook_attention()
    results['outlook'] = (outlook_config, outlook_module)
    
    # 2. WeightedPermuteMLP (ViP)
    vip_config, vip_module = test_vip_mlp()
    results['vip'] = (vip_config, vip_module)
    
    # 3. S2Attention
    s2_config, s2_module = test_s2_attention()
    results['s2'] = (s2_config, s2_module)
    
    # 4. DAModule (DANet)
    danet_config, danet_module = test_danet_module()
    results['danet'] = (danet_config, danet_module)
    
    # 总结
    print("="*60)
    print("📋 测试结果总结:")
    for name, (config, module) in results.items():
        if config is not None:
            print(f"  ✅ {name}: 成功配置 {config}")
        else:
            print(f"  ❌ {name}: 所有配置都失败")
    
    print("\n💡 可以恢复的模块建议:")
    for name, (config, module) in results.items():
        if config is not None:
            print(f"  - {name}: 使用配置 {config}")

if __name__ == "__main__":
    main()