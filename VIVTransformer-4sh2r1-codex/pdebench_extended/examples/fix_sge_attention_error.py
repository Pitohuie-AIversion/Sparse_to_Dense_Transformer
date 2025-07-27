#!/usr/bin/env python3
"""
修复SGE注意力机制的序列长度错误

问题分析：
1. SGE注意力使用CNN适配器，要求序列长度必须能形成正方形空间维度
2. 配置文件中seq_len=49（7x7），但实际输入数据的序列长度可能不是49
3. 数据加载器可能产生了不同的序列长度，导致维度不匹配

解决方案：
1. 检查实际数据的维度
2. 修改配置文件中的seq_len以匹配实际数据
3. 或者修改数据预处理以匹配配置的seq_len
"""

import torch
import yaml
import numpy as np
from pathlib import Path
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def analyze_data_dimensions(config_path):
    """
    分析数据的实际维度
    """
    # 加载配置
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    logger.info("=== 数据维度分析 ===")
    logger.info(f"配置文件中的seq_len: {config['model']['seq_len']}")
    logger.info(f"配置文件中的input_dim: {config['model']['input_dim']}")
    logger.info(f"配置文件中的output_dim: {config['model']['output_dim']}")
    
    # 计算期望的序列长度
    input_dim = config['model']['input_dim']
    output_dim = config['model']['output_dim']
    
    # 从input_dim推算实际的空间分辨率
    # input_dim = H * W * C，假设C=1
    input_spatial_size = int(np.sqrt(input_dim))
    output_spatial_size = int(np.sqrt(output_dim))
    
    logger.info(f"从input_dim推算的输入空间尺寸: {input_spatial_size}x{input_spatial_size}")
    logger.info(f"从output_dim推算的输出空间尺寸: {output_spatial_size}x{output_spatial_size}")
    
    # 检查是否为完全平方数
    if input_spatial_size * input_spatial_size != input_dim:
        logger.warning(f"输入维度 {input_dim} 不是完全平方数！")
        # 寻找最接近的完全平方数
        closest_square = int(np.sqrt(input_dim)) ** 2
        logger.info(f"最接近的完全平方数: {closest_square} ({int(np.sqrt(closest_square))}x{int(np.sqrt(closest_square))})")
    
    if output_spatial_size * output_spatial_size != output_dim:
        logger.warning(f"输出维度 {output_dim} 不是完全平方数！")
        closest_square = int(np.sqrt(output_dim)) ** 2
        logger.info(f"最接近的完全平方数: {closest_square} ({int(np.sqrt(closest_square))}x{int(np.sqrt(closest_square))})")
    
    return {
        'config_seq_len': config['model']['seq_len'],
        'input_dim': input_dim,
        'output_dim': output_dim,
        'input_spatial_size': input_spatial_size,
        'output_spatial_size': output_spatial_size
    }

def suggest_fixes(analysis_result):
    """
    根据分析结果建议修复方案
    """
    logger.info("\n=== 修复建议 ===")
    
    config_seq_len = analysis_result['config_seq_len']
    input_dim = analysis_result['input_dim']
    input_spatial_size = analysis_result['input_spatial_size']
    
    # 方案1：修改seq_len以匹配实际数据
    actual_seq_len = input_spatial_size * input_spatial_size
    if actual_seq_len != config_seq_len:
        logger.info(f"方案1：修改配置文件中的seq_len")
        logger.info(f"  当前seq_len: {config_seq_len}")
        logger.info(f"  建议seq_len: {actual_seq_len} ({input_spatial_size}x{input_spatial_size})")
    
    # 方案2：使用不同的注意力机制
    logger.info(f"\n方案2：更换注意力机制")
    logger.info(f"  SGE注意力要求序列长度为完全平方数")
    logger.info(f"  建议使用以下注意力机制：")
    logger.info(f"    - 'muse': 多尺度注意力，适合多尺度任务")
    logger.info(f"    - 'relative': 相对位置注意力，对序列长度要求较宽松")
    logger.info(f"    - 'se': 通道注意力，也使用CNN适配器但更灵活")
    
    # 方案3：修改数据预处理
    logger.info(f"\n方案3：修改数据预处理")
    logger.info(f"  调整数据加载器，确保输入数据的序列长度为完全平方数")
    
    return {
        'suggested_seq_len': actual_seq_len,
        'alternative_attention_types': ['muse', 'relative', 'se', 'cbam']
    }

def create_fixed_config(original_config_path, output_config_path, fix_type='seq_len'):
    """
    创建修复后的配置文件
    """
    # 加载原始配置
    with open(original_config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    if fix_type == 'seq_len':
        # 修复seq_len
        input_dim = config['model']['input_dim']
        input_spatial_size = int(np.sqrt(input_dim))
        new_seq_len = input_spatial_size * input_spatial_size
        
        config['model']['seq_len'] = new_seq_len
        logger.info(f"修改seq_len: {config['model']['seq_len']} -> {new_seq_len}")
        
    elif fix_type == 'attention':
        # 更换注意力机制
        old_attention = config['model']['attention_type']
        config['model']['attention_type'] = 'muse'  # 使用多尺度注意力
        logger.info(f"修改注意力机制: {old_attention} -> muse")
        
        # 添加备注
        config['_fix_note'] = {
            'issue': 'SGE attention requires square sequence length',
            'solution': 'Changed to MUSE attention which is more flexible',
            'original_attention': old_attention
        }
    
    # 保存修复后的配置
    with open(output_config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True, indent=2)
    
    logger.info(f"修复后的配置文件已保存到: {output_config_path}")

def main():
    """
    主函数
    """
    config_path = "multiscale/improved_multiscale_config.yaml"
    
    logger.info("SGE注意力错误修复工具")
    logger.info("=" * 50)
    
    # 分析数据维度
    analysis = analyze_data_dimensions(config_path)
    
    # 建议修复方案
    suggestions = suggest_fixes(analysis)
    
    # 创建修复后的配置文件
    logger.info("\n=== 生成修复配置文件 ===")
    
    # 方案1：修改seq_len
    output_path_1 = "multiscale/improved_multiscale_config_fixed_seqlen.yaml"
    create_fixed_config(config_path, output_path_1, 'seq_len')
    
    # 方案2：更换注意力机制
    output_path_2 = "multiscale/improved_multiscale_config_fixed_attention.yaml"
    create_fixed_config(config_path, output_path_2, 'attention')
    
    logger.info("\n=== 使用建议 ===")
    logger.info("1. 首先尝试使用修改seq_len的配置文件:")
    logger.info(f"   python train_configurable_multiscale.py --config_file {output_path_1}")
    logger.info("\n2. 如果仍有问题，使用更换注意力机制的配置文件:")
    logger.info(f"   python train_configurable_multiscale.py --config_file {output_path_2}")
    logger.info("\n3. 推荐的注意力机制（按优先级）:")
    for i, attention_type in enumerate(suggestions['alternative_attention_types'], 1):
        logger.info(f"   {i}. {attention_type}")

if __name__ == "__main__":
    main()