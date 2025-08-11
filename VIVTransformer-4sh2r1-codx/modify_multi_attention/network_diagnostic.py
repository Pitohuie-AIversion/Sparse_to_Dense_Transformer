#!/usr/bin/env python3
"""
网络诊断脚本 - 检测和报告当前Transformer网络的问题
"""

import logging
import sys
import traceback
from pathlib import Path

import torch
import torch.nn as nn
import yaml

# 设置日志记录
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_config_integrity():
    """检查配置文件完整性"""
    logger.info("🔍 检查配置文件完整性...")
    
    config_files = [
        'configs/config.yaml',
        'configs/config_test.yaml', 
        'configs/config_old.yaml'
    ]
    
    issues = []
    
    for config_path in config_files:
        if not Path(config_path).exists():
            issues.append(f"配置文件不存在: {config_path}")
            continue
            
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                cfg = yaml.safe_load(f)
                
            # 检查必需的键
            required_keys = {
                'global': ['device'],
                'model': ['attention_type', 'd_model', 'num_heads'],
                'training': ['epochs', 'learning_rate']
            }
            
            for section, keys in required_keys.items():
                if section not in cfg:
                    issues.append(f"{config_path}: 缺少section '{section}'")
                    continue
                    
                for key in keys:
                    if key not in cfg[section]:
                        issues.append(f"{config_path}: 缺少key '{section}.{key}'")
                        
            # 检查use_dataparallel的位置
            if 'use_dataparallel' in cfg and 'global' in cfg:
                if 'use_dataparallel' not in cfg['global']:
                    issues.append(f"{config_path}: use_dataparallel应该在global section中")
                    
        except Exception as e:
            issues.append(f"{config_path}: 解析错误 - {e}")
    
    return issues


def check_model_architecture():
    """检查模型架构问题"""
    logger.info("🏗️ 检查模型架构...")
    
    issues = []
    
    try:
        from mymodels.transformer import TransformerFlowReconstructionModel
        from mymodels.embedding import EmbeddingAndEncoding
        
        # 测试基本模型构建
        test_config = {
            'input_dim': 400,
            'output_dim': 40000,
            'num_heads': 4,
            'num_layers': 2,  # 减少层数用于测试
            'd_model': 256,
            'max_time_steps': 100,
            'seq_len': 49
        }
        
        # 测试embedding层
        embedding = EmbeddingAndEncoding(
            input_dim=test_config['input_dim'],
            d_model=test_config['d_model'],
            max_time_steps=test_config['max_time_steps'],
            seq_len=test_config['seq_len']
        )
        
        # 测试输入
        batch_size = 2
        test_input = torch.randn(batch_size, test_config['input_dim'])
        test_time_steps = torch.randint(0, test_config['max_time_steps'], (batch_size,))
        
        try:
            embedded = embedding(test_input, test_time_steps)
            expected_shape = (batch_size, test_config['seq_len'], test_config['d_model'])
            if embedded.shape != expected_shape:
                issues.append(f"Embedding输出形状错误: 期望{expected_shape}, 实际{embedded.shape}")
        except Exception as e:
            issues.append(f"Embedding层测试失败: {e}")
        
        # 测试完整模型
        try:
            model = TransformerFlowReconstructionModel(
                attention_type='self',  # 使用简单的注意力机制
                **test_config
            )
            
            with torch.no_grad():
                output = model(test_input, test_time_steps)
                expected_output_shape = (batch_size, test_config['output_dim'])
                if output.shape != expected_output_shape:
                    issues.append(f"模型输出形状错误: 期望{expected_output_shape}, 实际{output.shape}")
                    
        except Exception as e:
            issues.append(f"模型构建或前向传播失败: {e}")
            
    except ImportError as e:
        issues.append(f"模型导入失败: {e}")
        
    return issues


def check_attention_mechanisms():
    """检查注意力机制"""
    logger.info("⚡ 检查注意力机制...")
    
    issues = []
    
    try:
        from mymodels.components.attention_factory import ATTENTION_MODULES, get_attention_module
        from mymodels.components.attention_adapter import get_attention_adapter
        
        # 测试基本注意力机制
        test_mechanisms = ['self', 'relative', 'sparse']
        
        for attn_type in test_mechanisms:
            if attn_type not in ATTENTION_MODULES:
                issues.append(f"注意力机制 '{attn_type}' 未在ATTENTION_MODULES中定义")
                continue
                
            try:
                attention_module, adapter_type = get_attention_module(
                    attention_type=attn_type,
                    d_model=256,
                    num_heads=4
                )
                adapter = get_attention_adapter(attention_module, adapter_type)
                
                # 测试前向传播
                test_input = torch.randn(2, 49, 256)
                with torch.no_grad():
                    output = adapter(test_input)
                    if output.shape != test_input.shape:
                        issues.append(f"注意力机制 '{attn_type}' 输出形状不匹配")
                        
            except Exception as e:
                issues.append(f"注意力机制 '{attn_type}' 测试失败: {e}")
                
    except ImportError as e:
        issues.append(f"注意力模块导入失败: {e}")
        
    return issues


def check_data_pipeline():
    """检查数据管道"""
    logger.info("📊 检查数据管道...")
    
    issues = []
    
    try:
        # 检查数据文件
        with open('configs/config.yaml', 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
            
        data_path = cfg.get('data', {}).get('path', '')
        if data_path and not Path(data_path).exists():
            issues.append(f"数据文件不存在: {data_path}")
            
    except Exception as e:
        issues.append(f"数据检查失败: {e}")
        
    return issues


def check_cuda_environment():
    """检查CUDA环境"""
    logger.info("🚀 检查CUDA环境...")
    
    issues = []
    
    if not torch.cuda.is_available():
        issues.append("CUDA不可用")
    else:
        logger.info(f"CUDA版本: {torch.version.cuda}")
        logger.info(f"可用GPU数量: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            logger.info(f"GPU {i}: {props.name} ({props.total_memory // 1024**3}GB)")
            
        # 测试基本CUDA操作
        try:
            test_tensor = torch.randn(100, 100).cuda()
            result = torch.matmul(test_tensor, test_tensor.t())
            del test_tensor, result
            torch.cuda.empty_cache()
        except Exception as e:
            issues.append(f"CUDA操作测试失败: {e}")
            
    return issues


def generate_diagnostic_report():
    """生成诊断报告"""
    logger.info("🔬 开始网络诊断...")
    
    all_issues = []
    
    # 运行各项检查
    checks = [
        ("配置文件", check_config_integrity),
        ("模型架构", check_model_architecture), 
        ("注意力机制", check_attention_mechanisms),
        ("数据管道", check_data_pipeline),
        ("CUDA环境", check_cuda_environment)
    ]
    
    for check_name, check_func in checks:
        try:
            issues = check_func()
            if issues:
                all_issues.extend([f"[{check_name}] {issue}" for issue in issues])
            else:
                logger.info(f"✅ {check_name} 检查通过")
        except Exception as e:
            all_issues.append(f"[{check_name}] 检查过程出错: {e}")
            logger.error(f"❌ {check_name} 检查失败: {e}")
    
    # 生成报告
    report_path = Path("network_diagnostic_report.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=== 网络诊断报告 ===\n\n")
        
        if all_issues:
            f.write("🚨 发现的问题:\n")
            for i, issue in enumerate(all_issues, 1):
                f.write(f"{i}. {issue}\n")
        else:
            f.write("✅ 未发现问题\n")
            
        f.write(f"\n诊断完成时间: {Path(__file__).stat().st_mtime}\n")
    
    logger.info(f"📋 诊断报告已保存到: {report_path}")
    
    # 输出摘要
    if all_issues:
        logger.warning(f"🚨 发现 {len(all_issues)} 个问题")
        for issue in all_issues[:5]:  # 只显示前5个
            logger.warning(f"  - {issue}")
        if len(all_issues) > 5:
            logger.warning(f"  ... 还有 {len(all_issues) - 5} 个问题，详见报告文件")
    else:
        logger.info("🎉 所有检查都通过了！")
    
    return all_issues


if __name__ == "__main__":
    try:
        issues = generate_diagnostic_report()
        sys.exit(1 if issues else 0)
    except KeyboardInterrupt:
        logger.info("诊断被用户中断")
        sys.exit(1)
    except Exception as e:
        logger.error(f"诊断过程发生未预期错误: {e}")
        traceback.print_exc()
        sys.exit(1)