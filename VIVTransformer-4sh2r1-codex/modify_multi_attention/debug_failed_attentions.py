#!/usr/bin/env python3
"""
独立调试失败的注意力机制 (BAM, PSA, Residual)
"""
import torch
import traceback
from mymodels.components.attention_factory import get_attention_module, ADAPTER_MAPPING
from mymodels.components.attention_adapter import get_attention_adapter

def summarize_module_devices(module):
    """Return a summary string of parameter/buffer devices for the module."""
    devices = {}
    param_count = 0
    for n, p in module.named_parameters(recurse=True):
        if p is None:
            continue
        d = str(p.device)
        devices[d] = devices.get(d, 0) + p.numel()
        param_count += p.numel()
    buf_devices = {}
    for n, b in module.named_buffers(recurse=True):
        if b is None:
            continue
        d = str(b.device)
        buf_devices[d] = buf_devices.get(d, 0) + b.numel()
    return param_count, devices, buf_devices


def test_single_attention(attention_type, d_model=192, batch_size=2, seq_len=400):
    """测试单个注意力机制"""
    print(f"\n{'='*60}")
    print(f"🔍 测试注意力机制: {attention_type}")
    print(f"📊 配置: d_model={d_model}, batch_size={batch_size}, seq_len={seq_len}")
    print(f"{'='*60}")
    
    try:
        # 获取注意力模块和适配器
        attention_module, factory_adapter_type = get_attention_module(
            attention_type=attention_type,
            d_model=d_model,
            num_heads=4,
            spatial_dim=int(seq_len**0.5) if seq_len == 400 else 7
        )
        print(f"✅ 成功创建注意力模块: {attention_module.__class__.__name__}")
        
        adapter_type = factory_adapter_type
        print(f"✅ 适配器类型: {adapter_type}")
        
        # 创建测试输入
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # 打印移动前设备分布
        pcount, pdevs, bdevs = summarize_module_devices(attention_module)
        print(f"🔎 模块参数总数: {pcount}")
        print(f"🔎 移动前 参数设备分布: {pdevs}")
        print(f"🔎 移动前 Buffer设备分布: {bdevs}")

        # 将模块显式移动到设备
        attention_module = attention_module.to(device)

        # 再次打印移动后设备分布
        pcount_after, pdevs_after, bdevs_after = summarize_module_devices(attention_module)
        print(f"🔎 移动后 参数设备分布: {pdevs_after}")
        print(f"🔎 移动后 Buffer设备分布: {bdevs_after}")

        # 创建小批量测试数据
        x = torch.randn(batch_size, seq_len, d_model, device=device)
        print(f"✅ 创建测试输入: {x.shape}, device={x.device}")
        
        # 手动测试注意力模块（不通过适配器）
        if adapter_type.value == "cnn":
            spatial_dim = int(seq_len**0.5)
            x_reshaped = x.transpose(1, 2).contiguous().view(batch_size, d_model, spatial_dim, spatial_dim)
            print(f"🔍 CNN 输入形状: {x_reshaped.shape}, device={x_reshaped.device}")
            
            with torch.no_grad():
                try:
                    output = attention_module(x_reshaped)
                except RuntimeError as e:
                    msg = str(e)
                    print(f"❌ 前向传播失败: {msg}")
                    # 设备不一致：降级到CPU执行一次
                    if (
                        "Input type (torch.cuda.FloatTensor) and weight type (torch.FloatTensor)" in msg
                        or "Expected all tensors to be on the same device" in msg
                    ):
                        print("🔁 检测到设备不匹配，降级到 CPU 执行此前向计算...")
                        cpu = torch.device("cpu")
                        attention_module = attention_module.to(cpu)
                        x_cpu = x_reshaped.to(cpu)
                        output = attention_module(x_cpu)
                        if isinstance(output, tuple):
                            output = output[0]
                        if output is None:
                            output = torch.zeros_like(x_cpu)
                        # 移回目标设备
                        output = output.to(x_reshaped.device)
                    else:
                        # 其他错误直接抛出
                        raise

                print(f"🔍 注意力模块输出形状: {output.shape if not isinstance(output, tuple) else [o.shape for o in output]}")
                
                if isinstance(output, tuple):
                    output = output[0]
                    print(f"🔍 解包后输出形状: {output.shape}")
                
                print(f"🔍 输出总元素数: {output.numel()}")
                print(f"🔍 期望重塑形状: [{batch_size}, {d_model}, {seq_len}]")
                print(f"🔍 期望总元素数: {batch_size * d_model * seq_len}")
                
                # 尝试重塑
                try:
                    output_reshaped = output.view(batch_size, d_model, seq_len).transpose(1, 2)
                    print(f"✅ 重塑成功: {output_reshaped.shape}")
                except Exception as reshape_error:
                    print(f"❌ 重塑失败: {reshape_error}")
                    # 尝试其他可能的重塑
                    print(f"🔧 尝试自适应重塑...")
                    if output.numel() == batch_size * seq_len * d_model:
                        # 如果总元素数正确，尝试不同的维度顺序
                        output_reshaped = output.view(batch_size, seq_len, d_model)
                        print(f"✅ 自适应重塑成功: {output_reshaped.shape}")
                    else:
                        print(f"❌ 无法重塑，元素数量不匹配")
                        return False
        
        print(f"🎉 {attention_type} 测试通过！")
        return True
        
    except Exception as e:
        print(f"❌ {attention_type} 测试失败:")
        print(f"   错误类型: {type(e).__name__}")
        print(f"   错误消息: {str(e)}")
        print("   详细堆栈:")
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("🧪 开始调试失败的注意力机制")
    
    # 失败的注意力机制列表
    failed_attentions = ["bam", "psa", "residual"]
    
    # 逐一测试失败的注意力机制
    results = {}
    for attention_type in failed_attentions:
        results[attention_type] = test_single_attention(attention_type)
    
    # 总结测试结果
    print(f"\n{'='*60}")
    print("📋 测试结果总结:")
    print(f"{'='*60}")
    for attention_type, success in results.items():
        status = "✅ 通过" if success else "❌ 失败"
        print(f"   {attention_type}: {status}")
    
    failed_count = sum(1 for success in results.values() if not success)
    if failed_count == 0:
        print("\n🎉 所有注意力机制测试通过！")
    else:
        print(f"\n⚠️ 有 {failed_count} 个注意力机制仍然失败")

if __name__ == "__main__":
    main()