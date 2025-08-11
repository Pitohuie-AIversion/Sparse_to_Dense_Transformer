#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test all attention mechanisms for compatibility with sequence data.
"""

import torch
import sys
sys.path.append('.')
from mymodels.components.attention_factory import get_attention_module, ATTENTION_MODULES

def test_attention_mechanism(attention_type, d_model=256, num_heads=4, seq_len=49):
    """
    Test a specific attention mechanism with sequence data.
    """
    try:
        print(f"\n🔍 Testing {attention_type} attention mechanism...")
        
        # Get attention module
        attention_module, adapter_type = get_attention_module(attention_type, d_model=d_model, num_heads=num_heads)
        print(f"   ✅ Module created successfully")
        print(f"   Adapter type: {adapter_type}")
        
        # Create test input based on adapter type
        batch_size = 2
        
        if adapter_type.name == 'SINGLE_INPUT':
            # For single input adapters, try sequence format first
            x = torch.randn(batch_size, seq_len, d_model)
            print(f"   📝 Test input created: {x.shape} (sequence format)")
            
            try:
                output = attention_module(x)
                print(f"   ✅ Forward pass successful with sequence input: {output.shape}")
                return True, "sequence_compatible"
            except Exception as e:
                print(f"   ⚠️ Sequence format failed: {e}")
                
                # Try 4D image format for vision transformers
                try:
                    # Calculate square root for image dimensions
                    img_size = int(seq_len ** 0.5)
                    if img_size * img_size == seq_len:
                        x_4d = torch.randn(batch_size, 3, img_size * 32, img_size * 32)  # Use larger image size
                        print(f"   📝 Trying 4D image format: {x_4d.shape}")
                        output = attention_module(x_4d)
                        print(f"   ✅ Forward pass successful with 4D image input: {output.shape}")
                        return True, "image_only"
                    else:
                        print(f"   ❌ Cannot create square image from seq_len={seq_len}")
                        return False, str(e)
                except Exception as e2:
                    print(f"   ❌ Both sequence and image formats failed: {e2}")
                    return False, f"sequence: {e}, image: {e2}"
        
        elif adapter_type.name == 'QKV':
            # For QKV adapters
            query = torch.randn(batch_size, seq_len, d_model)
            key = torch.randn(batch_size, seq_len, d_model)
            value = torch.randn(batch_size, seq_len, d_model)
            print(f"   📝 Test input created: Q{query.shape}, K{key.shape}, V{value.shape}")
            
            output = attention_module(query, key, value)
            print(f"   ✅ Forward pass successful: {output.shape}")
            return True, "qkv_compatible"
        
        elif adapter_type.name == 'CNN':
            # For CNN adapters
            img_size = int(seq_len ** 0.5)
            if img_size * img_size == seq_len:
                x = torch.randn(batch_size, d_model, img_size, img_size)
                print(f"   📝 Test input created: {x.shape} (CNN format)")
                
                output = attention_module(x)
                print(f"   ✅ Forward pass successful: {output.shape}")
                return True, "cnn_compatible"
            else:
                print(f"   ❌ Cannot create square image from seq_len={seq_len}")
                return False, "invalid_seq_len_for_cnn"
        
        else:
            print(f"   ❌ Unknown adapter type: {adapter_type}")
            return False, f"unknown_adapter_type: {adapter_type}"
            
    except Exception as e:
        print(f"   ❌ Failed to create or test {attention_type}: {e}")
        return False, str(e)

def main():
    """
    Test all attention mechanisms.
    """
    print("🚀 Testing all attention mechanisms for compatibility...")
    
    results = {}
    compatible_mechanisms = []
    incompatible_mechanisms = []
    
    for attention_type in ATTENTION_MODULES.keys():
        success, details = test_attention_mechanism(attention_type)
        results[attention_type] = (success, details)
        
        if success:
            compatible_mechanisms.append((attention_type, details))
        else:
            incompatible_mechanisms.append((attention_type, details))
    
    # Summary
    print("\n" + "="*60)
    print("📊 COMPATIBILITY SUMMARY")
    print("="*60)
    
    print(f"\n✅ Compatible mechanisms ({len(compatible_mechanisms)}):")
    for mechanism, details in compatible_mechanisms:
        print(f"   • {mechanism}: {details}")
    
    print(f"\n❌ Incompatible mechanisms ({len(incompatible_mechanisms)}):")
    for mechanism, details in incompatible_mechanisms:
        print(f"   • {mechanism}: {details}")
    
    print(f"\n📈 Success rate: {len(compatible_mechanisms)}/{len(results)} ({len(compatible_mechanisms)/len(results)*100:.1f}%)")
    
    # Save results to file
    with open('attention_compatibility_report.txt', 'w') as f:
        f.write("Attention Mechanism Compatibility Report\n")
        f.write("="*50 + "\n\n")
        
        f.write(f"Compatible mechanisms ({len(compatible_mechanisms)}):\n")
        for mechanism, details in compatible_mechanisms:
            f.write(f"  {mechanism}: {details}\n")
        
        f.write(f"\nIncompatible mechanisms ({len(incompatible_mechanisms)}):\n")
        for mechanism, details in incompatible_mechanisms:
            f.write(f"  {mechanism}: {details}\n")
        
        f.write(f"\nSuccess rate: {len(compatible_mechanisms)}/{len(results)} ({len(compatible_mechanisms)/len(results)*100:.1f}%)\n")
    
    print("\n💾 Results saved to 'attention_compatibility_report.txt'")
    return len(compatible_mechanisms) > 0

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 At least some attention mechanisms are compatible!")
    else:
        print("\n😞 No attention mechanisms are compatible with current setup.")
    
    sys.exit(0 if success else 1)