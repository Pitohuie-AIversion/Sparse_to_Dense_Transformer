import torch
import torch.nn as nn
from fightingcv_attention.attention.gfnet import GFNet
from fightingcv_attention.attention.S2Attention import S2Attention
from fightingcv_attention.attention.DANet import DAModule

# 快速检查这些模块的实际行为
def test_gfnet():
    print("=== Testing GFNet ===")
    try:
        model = GFNet()
        print(f"GFNet created: {model}")
        x = torch.randn(1, 3, 16, 16)
        out = model(x)
        print(f"GFNet output shape: {type(out)}, {out.shape if hasattr(out, 'shape') else 'not tensor'}")
    except Exception as e:
        print(f"GFNet error: {e}")

def test_s2():
    print("\n=== Testing S2Attention ===") 
    try:
        model = S2Attention(channels=256)
        print(f"S2Attention created: {model}")
        x = torch.randn(1, 256, 7, 7)
        out = model(x)
        print(f"S2Attention output: {type(out)}, shape: {out.shape if hasattr(out, 'shape') else 'not tensor'}")
    except Exception as e:
        print(f"S2Attention error: {e}")

def test_danet():
    print("\n=== Testing DAModule ===")
    try:
        model = DAModule(in_channels=256)
        print(f"DAModule created: {model}")
        x = torch.randn(1, 256, 7, 7)
        out = model(x)
        print(f"DAModule output: {type(out)}, shape: {out.shape if hasattr(out, 'shape') else 'not tensor'}")
        if isinstance(out, (tuple, list)):
            print(f"Tuple/list length: {len(out)}")
            for i, o in enumerate(out):
                if hasattr(o, 'shape'):
                    print(f"  [{i}]: {o.shape}")
    except Exception as e:
        print(f"DAModule error: {e}")

if __name__ == "__main__":
    test_gfnet()
    test_s2()
    test_danet()