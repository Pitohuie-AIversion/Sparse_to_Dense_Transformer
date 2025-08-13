#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
性能优化演示脚本

该脚本演示如何启用和使用各种性能优化功能来提升 VIVTransformer 的训练效率。
"""

import argparse
import time
from pathlib import Path
import torch
import yaml

def create_optimized_config():
    """创建优化配置文件"""
    optimized_config = {
        'performance': {
            # PyTorch 编译优化
            'torch_compile': {
                'enabled': True,
                'mode': 'default',  # 可选: 'default', 'reduce-overhead', 'max-autotune'
                'dynamic': False,
                'fullgraph': False,
                'backend': 'inductor'
            },
            
            # 硬件优化
            'hardware': {
                'enable_tf32': True,  # Ampere GPU (RTX 30/40 系列)
                'set_float32_matmul_precision': 'high',  # 'highest', 'high', 'medium'
                'enable_cudnn_benchmark': True
            },
            
            # 优化器配置
            'optimizer': {
                'use_fused': True,  # Fused AdamW
                'foreach': True,    # 批量参数更新
                'differentiable': False,
                'capturable': False
            },
            
            # 内存格式优化（对 Transformer 可能效果有限）
            'memory_format': {
                'apply_to_model': False,  # channels_last 对 Transformer 效果有限
                'apply_to_inputs': False
            },
            
            # 注意力机制优化
            'attention': {
                'use_sdpa': True,       # PyTorch Scaled Dot-Product Attention
                'flash_attention': False, # 需要额外安装
                'compile_attention': True
            },
            
            # 数据加载优化
            'data_loading': {
                'prefetch_factor': 2,    # 预取批次数
                'drop_last': False,      # 保留所有数据
                'pin_memory_device': ''  # 使用默认设备
            },
            
            # 内存优化
            'memory': {
                'max_split_size_mb': 512,  # CUDA 内存分配器配置
                'empty_cache_freq': 10     # 每 N 个 epoch 清理缓存
            },
            
            # 实验性优化
            'experimental': {
                'use_deterministic_algorithms': False,  # 确定性算法（会降低性能）
                'enable_anomaly_detection': False,     # 梯度异常检测（调试用）
                'warn_only_deterministic': True
            },
            
            # CPU 特定优化
            'hardware_specific': {
                'cpu': {
                    'num_threads': 8,      # 根据 CPU 核心数调整
                    'enable_mkldnn': True  # Intel MKL-DNN
                }
            }
        }
    }
    
    return optimized_config

def benchmark_performance():
    """简单的性能基准测试"""
    print("🔥 开始性能基准测试...")
    
    # 创建测试张量
    batch_size = 32
    seq_len = 49
    d_model = 512
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"使用 GPU: {torch.cuda.get_device_name()}")
    else:
        device = torch.device('cpu')
        print("使用 CPU")
    
    # 创建测试数据
    x = torch.randn(batch_size, seq_len, d_model, device=device)
    
    # 简单的线性层测试
    layer = torch.nn.Linear(d_model, d_model).to(device)
    
    # 预热
    for _ in range(10):
        _ = layer(x)
    
    # 基准测试
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start_time = time.time()
    
    for _ in range(100):
        output = layer(x)
    
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    end_time = time.time()
    
    duration = end_time - start_time
    throughput = 100 / duration
    
    print(f"⚡ 前向传播性能: {throughput:.2f} 次/秒")
    print(f"📊 平均延迟: {duration * 1000 / 100:.2f} ms")
    
    return throughput

def main():
    parser = argparse.ArgumentParser(description="性能优化演示脚本")
    parser.add_argument(
        "--create-config", 
        action="store_true", 
        help="创建优化配置文件"
    )
    parser.add_argument(
        "--benchmark", 
        action="store_true", 
        help="运行性能基准测试"
    )
    parser.add_argument(
        "--run-training", 
        action="store_true", 
        help="运行优化训练"
    )
    parser.add_argument(
        "--optimization-level", 
        choices=['minimal', 'balanced', 'aggressive'], 
        default='balanced',
        help="优化级别"
    )
    
    args = parser.parse_args()
    
    print("🚀 VIVTransformer 性能优化演示")
    print("=" * 50)
    
    # 显示系统信息
    print(f"PyTorch 版本: {torch.__version__}")
    print(f"CUDA 可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA 版本: {torch.version.cuda}")
        print(f"GPU 设备: {torch.cuda.get_device_name()}")
        print(f"GPU 内存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print("-" * 50)
    
    if args.create_config:
        print("📝 创建性能优化配置文件...")
        config_path = Path("configs/performance_config.yaml")
        config_path.parent.mkdir(exist_ok=True)
        
        optimized_config = create_optimized_config()
        
        # 根据优化级别调整配置
        if args.optimization_level == 'minimal':
            optimized_config['performance']['torch_compile']['enabled'] = False
            optimized_config['performance']['optimizer']['use_fused'] = False
        elif args.optimization_level == 'aggressive':
            optimized_config['performance']['torch_compile']['mode'] = 'max-autotune'
            optimized_config['performance']['memory_format']['apply_to_model'] = True
        
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(optimized_config, f, default_flow_style=False, allow_unicode=True)
        
        print(f"✓ 配置文件已创建: {config_path}")
        print(f"优化级别: {args.optimization_level}")
    
    if args.benchmark:
        throughput = benchmark_performance()
        print(f"\n📈 基准测试完成，吞吐量: {throughput:.2f} 次/秒")
    
    if args.run_training:
        print("🎯 启动优化训练...")
        import subprocess
        import sys
        
        # 确保配置文件存在
        perf_config_path = Path("configs/performance_config.yaml")
        if not perf_config_path.exists():
            print("⚠️ 性能配置文件不存在，正在创建...")
            optimized_config = create_optimized_config()
            perf_config_path.parent.mkdir(exist_ok=True)
            with open(perf_config_path, 'w', encoding='utf-8') as f:
                yaml.dump(optimized_config, f, default_flow_style=False, allow_unicode=True)
        
        # 运行主训练脚本
        cmd = [sys.executable, "main.py", "-c", "configs/config.yaml"]
        print(f"执行命令: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print("✓ 训练完成")
            print(result.stdout)
        except subprocess.CalledProcessError as e:
            print(f"✗ 训练失败: {e}")
            print(f"错误输出: {e.stderr}")
    
    print("\n🎉 演示完成！")
    print("\n💡 使用提示:")
    print("- 对于首次使用，建议先运行 --create-config --optimization-level minimal")
    print("- 然后运行 --benchmark 测试基础性能")
    print("- 最后使用 --run-training 开始优化训练")
    print("- 根据硬件配置调整优化级别：minimal < balanced < aggressive")

if __name__ == "__main__":
    main()