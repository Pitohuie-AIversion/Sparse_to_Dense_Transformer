#!/usr/bin/env python3
"""检查DarcyFlow数据集结构的脚本

这个脚本用于检查PDEBench中的DarcyFlow数据集是否可以用我们的自定义数据集适配器加载。
"""

import h5py
import numpy as np
import sys
from pathlib import Path

def inspect_hdf5_file(file_path):
    """检查HDF5文件的结构和内容"""
    print(f"正在检查文件: {file_path}")
    print("=" * 60)
    
    try:
        with h5py.File(file_path, 'r') as f:
            print("\n📁 文件结构:")
            print_structure(f, "")
            
            # 检查数据集的详细信息
            print("\n📊 数据集详细信息:")
            for key in f.keys():
                dataset = f[key]
                if isinstance(dataset, h5py.Dataset):
                    print(f"\n🔹 数据集: {key}")
                    print(f"   形状: {dataset.shape}")
                    print(f"   数据类型: {dataset.dtype}")
                    print(f"   大小: {dataset.size:,} 个元素")
                    
                    # 显示数据的统计信息
                    if dataset.size > 0:
                        sample_data = dataset[0] if len(dataset.shape) > 0 else dataset[()]
                        if isinstance(sample_data, np.ndarray) and sample_data.size > 0:
                            print(f"   数据范围: [{np.min(sample_data):.6f}, {np.max(sample_data):.6f}]")
                            print(f"   均值: {np.mean(sample_data):.6f}")
                            print(f"   标准差: {np.std(sample_data):.6f}")
                        
                        # 如果是多维数据，显示前几个样本的形状
                        if len(dataset.shape) > 1:
                            print(f"   前3个样本形状:")
                            for i in range(min(3, dataset.shape[0])):
                                sample = dataset[i]
                                print(f"     样本 {i}: {sample.shape if hasattr(sample, 'shape') else type(sample)}")
            
            # 检查属性
            print("\n🏷️ 文件属性:")
            for attr_name, attr_value in f.attrs.items():
                print(f"   {attr_name}: {attr_value}")
                
    except Exception as e:
        print(f"❌ 读取文件时出错: {e}")
        return False
    
    return True

def print_structure(group, indent):
    """递归打印HDF5文件结构"""
    for key in group.keys():
        item = group[key]
        if isinstance(item, h5py.Group):
            print(f"{indent}📁 {key}/")
            print_structure(item, indent + "  ")
        elif isinstance(item, h5py.Dataset):
            print(f"{indent}📄 {key} {item.shape} {item.dtype}")

def check_compatibility_with_custom_adapter():
    """检查数据集是否与我们的自定义适配器兼容"""
    print("\n🔍 兼容性检查:")
    print("=" * 40)
    
    file_path = "PDEBench/pdebench/data_download/data/2D/DarcyFlow/2D_DarcyFlow_beta0.1_Train.hdf5"
    
    if not Path(file_path).exists():
        print(f"❌ 文件不存在: {file_path}")
        return False
    
    try:
        with h5py.File(file_path, 'r') as f:
            keys = list(f.keys())
            print(f"✅ 文件可以正常打开")
            print(f"✅ 包含 {len(keys)} 个顶级数据集/组")
            
            # 检查是否有常见的数据键
            common_keys = ['input', 'output', 'target', 'data', 'solution', 'u', 'a']
            found_keys = []
            for key in keys:
                if key.lower() in [k.lower() for k in common_keys]:
                    found_keys.append(key)
            
            if found_keys:
                print(f"✅ 找到可能的数据键: {found_keys}")
            else:
                print(f"⚠️ 未找到常见的数据键，可用键: {keys}")
            
            # 检查数据形状
            for key in keys:
                dataset = f[key]
                if isinstance(dataset, h5py.Dataset):
                    shape = dataset.shape
                    if len(shape) >= 3:  # 至少需要 [N, T, spatial] 或 [N, spatial, T]
                        print(f"✅ 数据集 '{key}' 形状 {shape} 可能适合时间序列处理")
                    elif len(shape) == 2:
                        print(f"⚠️ 数据集 '{key}' 形状 {shape} 可能需要重塑")
                    else:
                        print(f"❌ 数据集 '{key}' 形状 {shape} 不适合")
            
            return True
            
    except Exception as e:
        print(f"❌ 兼容性检查失败: {e}")
        return False

def suggest_configuration():
    """建议配置参数"""
    print("\n💡 配置建议:")
    print("=" * 40)
    
    file_path = "PDEBench/pdebench/data_download/data/2D/DarcyFlow/2D_DarcyFlow_beta0.1_Train.hdf5"
    
    try:
        with h5py.File(file_path, 'r') as f:
            keys = list(f.keys())
            
            print("基于数据集结构，建议的配置:")
            print("```yaml")
            print("custom_dataset:")
            print(f'  data_root: "PDEBench/pdebench/data_download/data/2D/DarcyFlow"')
            print(f'  dataset_type: "darcy_flow"')
            print("  file_paths:")
            print(f'    - "2D_DarcyFlow_beta0.1_Train.hdf5"')
            print("  ")
            
            # 分析第一个数据集来推断配置
            first_key = keys[0]
            dataset = f[first_key]
            if isinstance(dataset, h5py.Dataset):
                shape = dataset.shape
                print(f"  # 基于数据集 '{first_key}' 形状 {shape}")
                
                if len(shape) == 3:
                    # 假设是 [N, spatial_x, spatial_y] 或 [N, T, spatial]
                    if shape[1] == shape[2]:  # 可能是空间维度
                        print(f"  spatial_resolution: [{shape[1]}, {shape[2]}]")
                        print(f"  sequence_length: 1  # 静态数据，需要人工设置")
                    else:
                        print(f"  sequence_length: {shape[1]}")
                        print(f"  spatial_resolution: [{int(np.sqrt(shape[2]))}, {int(np.sqrt(shape[2]))}]  # 假设是平方网格")
                elif len(shape) == 4:
                    # 假设是 [N, T, spatial_x, spatial_y]
                    print(f"  sequence_length: {shape[1]}")
                    print(f"  spatial_resolution: [{shape[2]}, {shape[3]}]")
                
            print("  ")
            print("  data_keys:")
            for i, key in enumerate(keys[:3]):  # 只显示前3个键
                if i == 0:
                    print(f'    input: "{key}"')
                elif i == 1:
                    print(f'    target: "{key}"')
                else:
                    print(f'    # {key}: "{key}"')
            
            print("  ")
            print("  normalization_method: \"minmax\"")
            print("  data_split_ratios: [0.7, 0.2, 0.1]")
            print("```")
            
    except Exception as e:
        print(f"❌ 无法生成配置建议: {e}")

def main():
    """主函数"""
    print("🔍 DarcyFlow数据集检查工具")
    print("=" * 60)
    
    # 检查文件是否存在
    file_path = "PDEBench/pdebench/data_download/data/2D/DarcyFlow/2D_DarcyFlow_beta0.1_Train.hdf5"
    
    if not Path(file_path).exists():
        print(f"❌ 文件不存在: {file_path}")
        print("\n请确保:")
        print("1. 文件路径正确")
        print("2. 已下载PDEBench数据集")
        print("3. 当前工作目录正确")
        return False
    
    # 检查文件结构
    success = inspect_hdf5_file(file_path)
    
    if success:
        # 检查兼容性
        check_compatibility_with_custom_adapter()
        
        # 提供配置建议
        suggest_configuration()
        
        print("\n✅ 检查完成！")
        print("\n📝 下一步:")
        print("1. 根据上面的建议创建配置文件")
        print("2. 使用 test_custom_dataset.py 测试数据加载")
        print("3. 开始训练")
        
        return True
    else:
        print("\n❌ 检查失败！")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)