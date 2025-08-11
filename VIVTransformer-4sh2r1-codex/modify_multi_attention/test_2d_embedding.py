#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试2D空间嵌入模块
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('.')

from mymodels.embedding_2d import EmbeddingAndEncoding2D, EmbeddingAndEncodingOriginal

def test_2d_embedding_basic():
    """测试2D嵌入的基本功能。"""
    print("🔍 测试2D嵌入模块基本功能...")
    
    # 测试参数
    batch_size = 2
    input_dim = 1  # 每个网格点一个压力值
    d_model = 256
    max_time_steps = 100
    grid_height, grid_width = 7, 7
    seq_len = grid_height * grid_width  # 49
    
    # 创建2D嵌入模块
    embedding_2d = EmbeddingAndEncoding2D(
        input_dim=input_dim,
        d_model=d_model,
        max_time_steps=max_time_steps,
        seq_len=seq_len,
        grid_height=grid_height,
        grid_width=grid_width
    )
    
    # 创建测试数据
    # 模拟7x7网格的压力数据，展平为49个值
    x_pressures = torch.randn(batch_size, seq_len)  # (2, 49)
    x_time_steps = torch.randint(0, max_time_steps, (batch_size,))  # (2,)
    
    print(f"输入形状: {x_pressures.shape}")
    print(f"时间步形状: {x_time_steps.shape}")
    
    # 前向传播
    try:
        output = embedding_2d(x_pressures, x_time_steps)
        print(f"✅ 2D嵌入输出形状: {output.shape}")
        print(f"✅ 期望形状: ({batch_size}, {seq_len}, {d_model})")
        
        # 验证输出形状
        expected_shape = (batch_size, seq_len, d_model)
        assert output.shape == expected_shape, f"输出形状不匹配: {output.shape} vs {expected_shape}"
        
        return True
        
    except Exception as e:
        print(f"❌ 2D嵌入测试失败: {e}")
        return False

def test_coordinate_mapping():
    """测试坐标映射功能。"""
    print("\n🗺️ 测试坐标映射...")
    
    grid_height, grid_width = 7, 7
    seq_len = grid_height * grid_width
    
    embedding_2d = EmbeddingAndEncoding2D(
        input_dim=1,
        d_model=64,
        max_time_steps=10,
        seq_len=seq_len,
        grid_height=grid_height,
        grid_width=grid_width
    )
    
    # 显示坐标映射
    embedding_2d.visualize_coordinate_mapping()
    
    # 获取坐标网格
    coords = embedding_2d.get_spatial_coordinates()
    print(f"\n坐标网格形状: {coords.shape}")
    
    # 验证坐标范围
    y_coords = coords[:, 0]
    x_coords = coords[:, 1]
    
    print(f"Y坐标范围: {y_coords.min().item()} - {y_coords.max().item()}")
    print(f"X坐标范围: {x_coords.min().item()} - {x_coords.max().item()}")
    
    # 验证坐标的正确性
    assert y_coords.min() == 0 and y_coords.max() == grid_height - 1
    assert x_coords.min() == 0 and x_coords.max() == grid_width - 1
    
    print("✅ 坐标映射验证通过")
    return True

def compare_embeddings():
    """比较原始嵌入和2D嵌入的差异。"""
    print("\n⚖️ 比较原始嵌入与2D嵌入...")
    
    # 参数设置
    batch_size = 1
    input_dim = 49  # 原始方法：所有49个值作为一个特征向量
    d_model = 128
    max_time_steps = 10
    seq_len = 49
    
    # 创建两种嵌入模块
    embedding_original = EmbeddingAndEncodingOriginal(
        input_dim=input_dim,
        d_model=d_model,
        max_time_steps=max_time_steps,
        seq_len=seq_len
    )
    
    embedding_2d = EmbeddingAndEncoding2D(
        input_dim=1,  # 每个网格点一个值
        d_model=d_model,
        max_time_steps=max_time_steps,
        seq_len=seq_len,
        grid_height=7,
        grid_width=7
    )
    
    # 创建测试数据
    x_flat = torch.randn(batch_size, input_dim)  # 原始方法的输入
    x_grid = x_flat.view(batch_size, seq_len)    # 2D方法的输入
    x_time = torch.randint(0, max_time_steps, (batch_size,))
    
    # 前向传播
    try:
        output_original = embedding_original(x_flat, x_time)
        output_2d = embedding_2d(x_grid, x_time)
        
        print(f"原始嵌入输出形状: {output_original.shape}")
        print(f"2D嵌入输出形状: {output_2d.shape}")
        
        # 计算参数数量
        params_original = sum(p.numel() for p in embedding_original.parameters())
        params_2d = sum(p.numel() for p in embedding_2d.parameters())
        
        print(f"原始嵌入参数数量: {params_original:,}")
        print(f"2D嵌入参数数量: {params_2d:,}")
        
        # 分析空间信息保留
        print("\n📊 空间信息分析:")
        print("原始方法: 将所有49个值作为单一特征向量处理，丢失空间位置信息")
        print("2D方法: 为每个网格点单独嵌入，并添加2D位置编码，保留空间邻域关系")
        
        return True
        
    except Exception as e:
        print(f"❌ 比较测试失败: {e}")
        return False

def visualize_spatial_attention_pattern():
    """可视化空间注意力模式（模拟）。"""
    print("\n🎨 可视化空间注意力模式...")
    
    grid_height, grid_width = 7, 7
    
    # 创建一个模拟的注意力权重矩阵
    # 假设中心点(3,3)对周围点有不同的注意力权重
    attention_weights = np.zeros((grid_height, grid_width))
    center_y, center_x = 3, 3
    
    for y in range(grid_height):
        for x in range(grid_width):
            # 距离中心点越近，注意力权重越高
            distance = np.sqrt((y - center_y)**2 + (x - center_x)**2)
            attention_weights[y, x] = np.exp(-distance / 2)
    
    # 可视化
    plt.figure(figsize=(10, 4))
    
    # 子图1：注意力权重热图
    plt.subplot(1, 2, 1)
    plt.imshow(attention_weights, cmap='hot', interpolation='nearest')
    plt.colorbar(label='Attention Weight')
    plt.title('空间注意力权重\n(中心点对周围的注意力)')
    plt.xlabel('X坐标')
    plt.ylabel('Y坐标')
    
    # 添加网格和数值
    for y in range(grid_height):
        for x in range(grid_width):
            plt.text(x, y, f'{attention_weights[y, x]:.2f}', 
                    ha='center', va='center', color='white', fontsize=8)
    
    # 子图2：展平后的序列索引
    plt.subplot(1, 2, 2)
    sequence_indices = np.arange(grid_height * grid_width).reshape(grid_height, grid_width)
    plt.imshow(sequence_indices, cmap='viridis', interpolation='nearest')
    plt.colorbar(label='Sequence Index')
    plt.title('展平后的序列索引\n(2D -> 1D映射)')
    plt.xlabel('X坐标')
    plt.ylabel('Y坐标')
    
    # 添加序列索引
    for y in range(grid_height):
        for x in range(grid_width):
            plt.text(x, y, f'{sequence_indices[y, x]}', 
                    ha='center', va='center', color='white', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('spatial_attention_visualization.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("✅ 空间注意力可视化已保存为 'spatial_attention_visualization.png'")
    
    return True

def main():
    """主测试函数。"""
    print("🚀 开始测试2D空间嵌入模块...")
    
    tests = [
        ("基本功能测试", test_2d_embedding_basic),
        ("坐标映射测试", test_coordinate_mapping),
        ("嵌入方法比较", compare_embeddings),
        ("空间注意力可视化", visualize_spatial_attention_pattern),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        print(f"📋 {test_name}")
        print(f"{'='*50}")
        
        try:
            if test_func():
                print(f"✅ {test_name} 通过")
                passed += 1
            else:
                print(f"❌ {test_name} 失败")
        except Exception as e:
            print(f"❌ {test_name} 异常: {e}")
    
    print(f"\n{'='*50}")
    print(f"📊 测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("\n🎉 所有测试通过！2D空间嵌入模块工作正常。")
        print("\n💡 建议:")
        print("1. 在transformer.py中导入并使用EmbeddingAndEncoding2D")
        print("2. 确保输入数据格式为(batch_size, seq_len)而不是完全展平")
        print("3. 利用2D位置信息可以更好地学习空间模式")
    else:
        print(f"\n😞 {total-passed} 个测试失败，需要进一步调试。")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)