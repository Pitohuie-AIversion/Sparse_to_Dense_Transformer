#!/usr/bin/env python3
"""
创建测试数据文件
"""

import torch
import numpy as np
from pathlib import Path

def create_test_data(save_path: str, num_samples: int = 100):
    """
    创建测试用的压力场数据
    
    Args:
        save_path: 保存路径
        num_samples: 样本数量
    """
    print(f"创建 {num_samples} 个测试样本...")
    
    # 创建随机数据
    # 输入: 20x20 = 400 (先创建为2D，然后展平)
    in_pressure_2d = torch.randn(num_samples, 20, 20)
    in_pressure = in_pressure_2d.view(num_samples, -1)  # 展平为 [N, 400]
    
    # 输出: 200x200 = 40000 (先创建为2D，然后展平)
    pressure_2d = torch.randn(num_samples, 200, 200)
    pressure = pressure_2d.view(num_samples, -1)  # 展平为 [N, 40000]
    
    # 时间步
    time_steps = torch.linspace(0, 1, num_samples)
    
    # 保存数据
    data = {
        'in_pressure': in_pressure,
        'pressure': pressure,
        'time_steps': time_steps
    }
    
    torch.save(data, save_path)
    print(f"测试数据已保存到: {save_path}")
    
    # 打印数据信息
    print(f"数据形状:")
    print(f"  输入压力场: {in_pressure.shape}")
    print(f"  输出压力场: {pressure.shape}")
    print(f"  时间步: {time_steps.shape}")
    
    return data

if __name__ == "__main__":
    # 创建数据目录
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    # 创建测试数据
    test_data_path = data_dir / "test_pressure_field_data.pt"
    create_test_data(str(test_data_path), num_samples=100)
    
    print("\n测试数据创建完成！")
    print(f"现在可以使用以下命令进行训练测试:")
    print(f"python train_pressure_field.py --config configs/pressure_field_training.yaml --data_path {test_data_path}")