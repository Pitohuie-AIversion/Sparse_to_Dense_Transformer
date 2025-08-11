#!/usr/bin/env python3
"""Test MOA attention mechanism directly."""

import torch
import sys
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from mymodels.components.attention_factory import get_attention_module

def test_moa_attention():
    """Test MOA attention mechanism."""
    print("Testing MOA attention mechanism...")
    
    try:
        # Get MOA attention module
        attention_module, adapter_type = get_attention_module('moa', d_model=256, num_heads=4)
        print(f"✅ MOA attention module created successfully")
        print(f"   Module type: {type(attention_module)}")
        print(f"   Adapter type: {adapter_type}")
        
        # Create test input - MOA expects 4D input (B, C, H, W) with 3 channels and 224x224 image size
        batch_size, channels, height, width = 2, 3, 224, 224
        x = torch.randn(batch_size, channels, height, width)
        print(f"✅ Test input created: {x.shape} (4D format for MOA with 3 channels and 224x224 size)")
        
        # Test forward pass
        with torch.no_grad():
            output = attention_module(x)
            print(f"✅ Forward pass successful")
            print(f"   Output shape: {output.shape}")
            print(f"   Output type: {type(output)}")
            
            if isinstance(output, tuple):
                print(f"   Output is tuple with {len(output)} elements")
                for i, elem in enumerate(output):
                    if elem is not None:
                        print(f"   Element {i}: {elem.shape if hasattr(elem, 'shape') else type(elem)}")
                    else:
                        print(f"   Element {i}: None")
            
        print("\n🎉 MOA attention mechanism test passed!")
        return True
        
    except Exception as e:
        print(f"❌ MOA attention test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_moa_attention()
    sys.exit(0 if success else 1)