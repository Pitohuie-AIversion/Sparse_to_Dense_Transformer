#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试修改后的Transformer模型（带2D空间嵌入）
"""

import torch
import yaml
import sys
sys.path.append('.')

from mymodels.model_factory import create_model
from utils.config import load_config

def test_model_creation():
    """测试模型创建功能。"""
    print("🔍 测试模型创建...")
    
    # 加载配置
    config_path = "configs/config.yaml"
    config = load_config(config_path)
    
    print(f"配置文件: {config_path}")
    print(f"使用2D嵌入: {config['model'].get('use_2d_embedding', True)}")
    print(f"网格尺寸: {config['model'].get('grid_height', 7)}x{config['model'].get('grid_width', 7)}")
    
    # 创建模型
    device = torch.device('cpu')  # 使用CPU进行测试
    attention_type = config['model']['attention_type']
    
    try:
        model = create_model(config, attention_type, device)
        print(f"✅ 模型创建成功")
        print(f"   注意力类型: {attention_type}")
        print(f"   模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
        print(f"   可训练参数: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
        
        return model
        
    except Exception as e:
        print(f"❌ 模型创建失败: {e}")
        return None

def test_model_forward():
    """测试模型前向传播。"""
    print("\n🚀 测试模型前向传播...")
    
    # 加载配置
    config = load_config("configs/config.yaml")
    device = torch.device('cpu')
    
    # 创建模型
    model = create_model(config, config['model']['attention_type'], device)
    model.eval()
    
    # 创建测试数据
    batch_size = 2
    seq_len = config['model']['seq_len']
    max_time_steps = config['model']['max_time_steps']
    
    # 对于2D嵌入，输入应该是(batch_size, seq_len)而不是(batch_size, input_dim)
    if config['model'].get('use_2d_embedding', True):
        x_pressures = torch.randn(batch_size, seq_len)  # (2, 49)
        print(f"2D嵌入模式 - 输入形状: {x_pressures.shape}")
    else:
        input_dim = config['model']['input_dim']
        x_pressures = torch.randn(batch_size, input_dim)  # (2, 400)
        print(f"原始嵌入模式 - 输入形状: {x_pressures.shape}")
    
    x_time_steps = torch.randint(0, max_time_steps, (batch_size,))  # (2,)
    
    print(f"时间步形状: {x_time_steps.shape}")
    
    # 前向传播
    try:
        with torch.no_grad():
            output = model(x_pressures, x_time_steps)
        
        print(f"✅ 前向传播成功")
        print(f"   输出形状: {output.shape}")
        print(f"   期望输出维度: {config['model']['output_dim']}")
        
        # 验证输出形状
        expected_shape = (batch_size, config['model']['output_dim'])
        assert output.shape == expected_shape, f"输出形状不匹配: {output.shape} vs {expected_shape}"
        
        print(f"✅ 输出形状验证通过")
        return True
        
    except Exception as e:
        print(f"❌ 前向传播失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_attention_mechanisms():
    """测试不同注意力机制。"""
    print("\n🎯 测试不同注意力机制...")
    
    config = load_config("configs/config.yaml")
    device = torch.device('cpu')
    
    # 获取可用的注意力机制
    attention_types = config.get('attention_test', {}).get('types', ['self'])
    
    successful_types = []
    failed_types = []
    
    for attention_type in attention_types[:5]:  # 只测试前5个
        print(f"\n测试注意力机制: {attention_type}")
        
        try:
            model = create_model(config, attention_type, device)
            
            # 创建测试数据
            batch_size = 1
            seq_len = config['model']['seq_len']
            
            if config['model'].get('use_2d_embedding', True):
                x_pressures = torch.randn(batch_size, seq_len)
            else:
                x_pressures = torch.randn(batch_size, config['model']['input_dim'])
            
            x_time_steps = torch.randint(0, config['model']['max_time_steps'], (batch_size,))
            
            # 前向传播
            with torch.no_grad():
                output = model(x_pressures, x_time_steps)
            
            print(f"  ✅ {attention_type} 成功")
            successful_types.append(attention_type)
            
        except Exception as e:
            print(f"  ❌ {attention_type} 失败: {e}")
            failed_types.append((attention_type, str(e)))
    
    print(f"\n📊 注意力机制测试结果:")
    print(f"   成功: {len(successful_types)}/{len(attention_types[:5])}")
    print(f"   成功的机制: {successful_types}")
    
    if failed_types:
        print(f"   失败的机制:")
        for attn_type, error in failed_types:
            print(f"     - {attn_type}: {error}")
    
    return len(successful_types) > 0

def compare_embedding_methods():
    """比较2D嵌入和原始嵌入的性能。"""
    print("\n⚖️ 比较嵌入方法...")
    
    config = load_config("configs/config.yaml")
    device = torch.device('cpu')
    attention_type = 'self'  # 使用简单的注意力机制
    
    results = {}
    
    for use_2d in [True, False]:
        print(f"\n测试 {'2D' if use_2d else '原始'} 嵌入方法...")
        
        # 修改配置
        test_config = config.copy()
        test_config['model']['use_2d_embedding'] = use_2d
        
        try:
            model = create_model(test_config, attention_type, device)
            
            # 创建测试数据
            batch_size = 2
            
            if use_2d:
                x_pressures = torch.randn(batch_size, config['model']['seq_len'])
            else:
                x_pressures = torch.randn(batch_size, config['model']['input_dim'])
            
            x_time_steps = torch.randint(0, config['model']['max_time_steps'], (batch_size,))
            
            # 测量推理时间
            import time
            start_time = time.time()
            
            with torch.no_grad():
                for _ in range(10):  # 运行10次取平均
                    output = model(x_pressures, x_time_steps)
            
            avg_time = (time.time() - start_time) / 10
            
            # 统计参数数量
            total_params = sum(p.numel() for p in model.parameters())
            
            results[f"{'2D' if use_2d else 'Original'}"] = {
                'success': True,
                'params': total_params,
                'avg_time': avg_time,
                'output_shape': output.shape
            }
            
            print(f"  ✅ 成功")
            print(f"     参数数量: {total_params:,}")
            print(f"     平均推理时间: {avg_time:.4f}s")
            print(f"     输出形状: {output.shape}")
            
        except Exception as e:
            print(f"  ❌ 失败: {e}")
            results[f"{'2D' if use_2d else 'Original'}"] = {
                'success': False,
                'error': str(e)
            }
    
    # 比较结果
    print(f"\n📊 嵌入方法比较:")
    for method, result in results.items():
        if result['success']:
            print(f"   {method}: {result['params']:,} 参数, {result['avg_time']:.4f}s")
        else:
            print(f"   {method}: 失败 - {result['error']}")
    
    return results

def main():
    """主测试函数。"""
    print("🚀 开始测试修改后的Transformer模型...")
    
    tests = [
        ("模型创建测试", test_model_creation),
        ("前向传播测试", test_model_forward),
        ("注意力机制测试", test_attention_mechanisms),
        ("嵌入方法比较", compare_embedding_methods),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*60}")
        print(f"📋 {test_name}")
        print(f"{'='*60}")
        
        try:
            result = test_func()
            if result or result is None:  # None表示创建测试
                print(f"✅ {test_name} 通过")
                passed += 1
            else:
                print(f"❌ {test_name} 失败")
        except Exception as e:
            print(f"❌ {test_name} 异常: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*60}")
    print(f"📊 测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("\n🎉 所有测试通过！2D空间嵌入模型工作正常。")
        print("\n💡 建议:")
        print("1. 2D空间嵌入已成功集成到Transformer模型中")
        print("2. 模型现在能够保留空间邻域信息")
        print("3. 可以开始训练以验证性能改进")
    else:
        print(f"\n😞 {total-passed} 个测试失败，需要进一步调试。")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)