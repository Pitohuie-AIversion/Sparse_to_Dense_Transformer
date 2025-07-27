#!/usr/bin/env python3
"""
测试SVD损失函数的确定性
验证修复后的SVD函数是否能产生一致的结果
"""

import torch
import numpy as np
from utils.loss import get_svd_modes, TotalLossWithSVD
from utils.system import set_seed

def test_svd_deterministic():
    """测试SVD函数的确定性"""
    print("🧪 测试SVD损失函数的确定性...")
    
    # 设置随机种子
    set_seed(42, deterministic=True)
    
    # 创建测试数据
    batch_size = 4
    height, width = 32, 32
    test_tensor = torch.randn(batch_size, height, width)
    
    # 第一次计算
    set_seed(42, deterministic=True)
    modes1 = get_svd_modes(test_tensor, topk=3)
    
    # 第二次计算（重新设置相同种子）
    set_seed(42, deterministic=True)
    modes2 = get_svd_modes(test_tensor, topk=3)
    
    # 检查结果是否一致
    all_equal = True
    for k in range(3):
        if not torch.allclose(modes1[k], modes2[k], atol=1e-6):
            all_equal = False
            print(f"❌ 模态 {k} 不一致")
            print(f"   最大差异: {torch.max(torch.abs(modes1[k] - modes2[k])).item():.2e}")
    
    if all_equal:
        print("✅ SVD函数产生确定性结果")
    else:
        print("❌ SVD函数结果不确定")
    
    return all_equal

def test_loss_deterministic():
    """测试损失函数的确定性"""
    print("\n🧪 测试损失函数的确定性...")
    
    # 创建损失函数
    loss_fn = TotalLossWithSVD(
        base_weight=0.5,
        svd_weights=[0.1, 0.2, 0.2],
        topk=3
    )
    
    # 创建测试数据
    batch_size = 4
    height, width = 32, 32
    pred = torch.randn(batch_size, height, width, requires_grad=True)
    target = torch.randn(batch_size, height, width)
    
    # 第一次计算
    set_seed(42, deterministic=True)
    loss1 = loss_fn(pred, target)
    
    # 第二次计算
    set_seed(42, deterministic=True)
    loss2 = loss_fn(pred, target)
    
    # 检查结果是否一致
    if torch.allclose(loss1, loss2, atol=1e-6):
        print("✅ 损失函数产生确定性结果")
        print(f"   损失值: {loss1.item():.6f}")
        return True
    else:
        print("❌ 损失函数结果不确定")
        print(f"   损失1: {loss1.item():.6f}")
        print(f"   损失2: {loss2.item():.6f}")
        print(f"   差异: {abs(loss1.item() - loss2.item()):.2e}")
        return False

def test_gradient_deterministic():
    """测试梯度计算的确定性"""
    print("\n🧪 测试梯度计算的确定性...")
    
    # 创建损失函数
    loss_fn = TotalLossWithSVD(
        base_weight=0.5,
        svd_weights=[0.1, 0.2, 0.2],
        topk=3
    )
    
    # 创建测试数据
    batch_size = 4
    height, width = 32, 32
    target = torch.randn(batch_size, height, width)
    
    # 第一次计算梯度
    set_seed(42, deterministic=True)
    pred1 = torch.randn(batch_size, height, width, requires_grad=True)
    loss1 = loss_fn(pred1, target)
    loss1.backward()
    grad1 = pred1.grad.clone()
    
    # 第二次计算梯度
    set_seed(42, deterministic=True)
    pred2 = torch.randn(batch_size, height, width, requires_grad=True)
    loss2 = loss_fn(pred2, target)
    loss2.backward()
    grad2 = pred2.grad.clone()
    
    # 检查梯度是否一致
    if torch.allclose(grad1, grad2, atol=1e-6):
        print("✅ 梯度计算产生确定性结果")
        print(f"   梯度范数: {torch.norm(grad1).item():.6f}")
        return True
    else:
        print("❌ 梯度计算结果不确定")
        print(f"   梯度1范数: {torch.norm(grad1).item():.6f}")
        print(f"   梯度2范数: {torch.norm(grad2).item():.6f}")
        print(f"   梯度差异范数: {torch.norm(grad1 - grad2).item():.2e}")
        return False

def main():
    """主测试函数"""
    print("🔧 SVD损失函数确定性测试")
    print("=" * 50)
    
    # 运行所有测试
    test1_passed = test_svd_deterministic()
    test2_passed = test_loss_deterministic()
    test3_passed = test_gradient_deterministic()
    
    print("\n" + "=" * 50)
    print("📊 测试结果总结:")
    print(f"   SVD模态计算: {'✅ 通过' if test1_passed else '❌ 失败'}")
    print(f"   损失函数计算: {'✅ 通过' if test2_passed else '❌ 失败'}")
    print(f"   梯度计算: {'✅ 通过' if test3_passed else '❌ 失败'}")
    
    if all([test1_passed, test2_passed, test3_passed]):
        print("\n🎉 所有测试通过！SVD损失函数现在是确定性的。")
        print("💡 这意味着训练结果现在可以完全复现。")
    else:
        print("\n⚠️  部分测试失败，需要进一步检查。")
    
    return all([test1_passed, test2_passed, test3_passed])

if __name__ == "__main__":
    main()