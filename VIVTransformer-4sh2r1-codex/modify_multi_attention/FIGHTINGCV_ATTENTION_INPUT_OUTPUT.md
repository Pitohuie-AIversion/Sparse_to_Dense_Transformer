# FightingCV 注意力层输入输出张量参数详细说明

本文档详细描述了项目中集成的每个 fightingcv 注意力层如何处理输入输出张量参数。

## 适配器类型概述

项目中使用了三种适配器类型来统一不同注意力机制的接口：

### 1. QKVAttentionAdapter（QKV型）
- **输入格式**：`forward(x, memory=None, return_attention=False)`
- **内部调用**：`attention(x, memory, memory, return_attention=return_attention)`
- **张量形状**：
  - `x`: `(batch_size, seq_len, d_model)`
  - `memory`: `(batch_size, seq_len, d_model)` 或 None
  - 输出: `(batch_size, seq_len, d_model)`

### 2. CNNStyleAttentionAdapter（CNN型）  
- **输入格式**：`forward(x, memory=None, return_attention=False)`
- **张量变换过程**：
  1. 输入: `x` 形状 `(batch_size, seq_len, d_model)`
  2. 重塑为: `(batch_size, d_model, spatial_dim, spatial_dim)`
     - `spatial_dim = int(seq_len**0.5)` 
     - 要求 `spatial_dim * spatial_dim == seq_len`
  3. 调用注意力: `attention(x_reshaped)`
  4. 输出重塑回: `(batch_size, seq_len, d_model)`

### 3. SingleInputAttentionAdapter（单输入型）
- **输入格式**：`forward(x, memory=None, return_attention=False)`
- **内部调用**：`attention(x)` （忽略 memory 参数）
- **张量形状**：保持 `(batch_size, seq_len, d_model)`

## 各注意力层详细说明

### QKV 兼容注意力层

#### 1. self（ScaledDotProductAttention）
- **适配器类型**：QKV
- **实例化参数**：`d_model=d_model, d_k=d_model, d_v=d_model, h=num_heads`
- **输入张量**：
  - `x`: `(batch_size, seq_len, d_model)`
  - `memory`: `(batch_size, seq_len, d_model)`
- **输出张量**：`(batch_size, seq_len, d_model)`

#### 2. simplified_self（SimplifiedScaledDotProductAttention）
- **适配器类型**：QKV
- **实例化参数**：`d_model=d_model, h=num_heads`
- **输入/输出张量**：同 self

#### 3. muse（MUSEAttention）
- **适配器类型**：QKV
- **实例化参数**：`d_model=d_model, d_k=d_model, d_v=d_model, h=num_heads`
- **输入/输出张量**：同 self

#### 4. ufo（UFOAttention）
- **适配器类型**：QKV
- **实例化参数**：`d_model=d_model, d_k=d_model, d_v=d_model, h=num_heads`
- **输入/输出张量**：同 self

#### 5. relative（RelativePositionSelfAttention）
- **适配器类型**：QKV
- **实例化参数**：`d_model, num_heads`
- **输入/输出张量**：同 self

#### 6. sparse（SparseSelfAttention）
- **适配器类型**：QKV
- **实例化参数**：`d_model, num_heads`
- **输入/输出张量**：同 self

#### 7. lsh（LSHSelfAttention）
- **适配器类型**：QKV
- **实例化参数**：`d_model, num_heads`
- **输入/输出张量**：同 self

### CNN 兼容注意力层

#### 8. se（SEAttention）
- **适配器类型**：CNN
- **实例化参数**：`channel=d_model, reduction=8`
- **输入张量变换**：
  - 原始输入: `(batch_size, seq_len, d_model)`
  - CNN输入: `(batch_size, d_model, spatial_dim, spatial_dim)`
  - CNN输出: `(batch_size, d_model, spatial_dim, spatial_dim)`
  - 最终输出: `(batch_size, seq_len, d_model)`

#### 9. sk（SKAttention）
- **适配器类型**：CNN
- **实例化参数**：`channel=d_model, reduction=8`
- **张量变换过程**：同 se

#### 10. cbam（CBAMBlock）
- **适配器类型**：CNN
- **实例化参数**：`channel=d_model, reduction=8`
- **张量变换过程**：同 se

#### 11. bam（BAMBlock）
- **适配器类型**：CNN
- **实例化参数**：`channel=d_model, reduction=8, dia_val=1`
- **特殊处理**：
  - 设置 `dia_val=1` 确保空间分辨率不变
  - 自动移动到 CUDA 设备（如果可用）
- **张量变换过程**：同 se

#### 12. eca（ECAAttention）
- **适配器类型**：CNN
- **实例化参数**：`kernel_size=3`
- **张量变换过程**：同 se

#### 13. shuffle（ShuffleAttention）
- **适配器类型**：CNN
- **实例化参数**：`channel=d_model, reduction=8`
- **张量变换过程**：同 se

#### 14. psa（PSA）
- **适配器类型**：CNN
- **实例化参数**：`channel=d_model, reduction=4`
- **特殊处理**：
  - 自动移动到 CUDA 设备（如果可用）
  - 内置 CPU 回退机制处理设备不匹配
- **张量变换过程**：同 se

#### 15. cot（CoTAttention）
- **适配器类型**：CNN
- **实例化参数**：`dim=d_model, kernel_size=3`
- **张量变换过程**：同 se

#### 16. polarized（SequentialPolarizedSelfAttention）
- **适配器类型**：CNN
- **实例化参数**：`channel=d_model`
- **张量变换过程**：同 se

#### 17. halo（HaloAttention）
- **适配器类型**：CNN
- **实例化参数**：`dim=d_model, block_size=1, halo_size=1`
- **张量变换过程**：同 se

#### 18. a2（DoubleAttention）
- **适配器类型**：CNN
- **实例化参数**：`d_model, 128, 128, True`
- **张量变换过程**：同 se

#### 19. parnet（ParNetAttention）
- **适配器类型**：CNN
- **实例化参数**：`channel=d_model`
- **张量变换过程**：同 se

#### 20. residual（SimpleResidualAttention）
- **适配器类型**：CNN
- **自定义实现**：包含通道注意力 + 残差连接
- **实例化参数**：`channel=d_model, reduction=8`
- **张量变换过程**：同 se

### 单输入兼容注意力层

#### 21. external（ExternalAttention）
- **适配器类型**：SINGLE_INPUT
- **实例化参数**：`d_model=d_model, S=8`
- **输入张量**：`(batch_size, seq_len, d_model)`
- **输出张量**：`(batch_size, seq_len, d_model)`

#### 22. aft（AFT_FULL）
- **适配器类型**：SINGLE_INPUT
- **实例化参数**：`d_model=d_model, n=spatial_dim*spatial_dim`
- **输入张量**：`(batch_size, seq_len, d_model)`
- **输出张量**：`(batch_size, seq_len, d_model)`

## 设备管理和错误处理

### 自动设备同步
- **CNN 适配器**：自动将注意力模块移动到输入张量所在设备
- **特殊模块**（BAM, PSA）：在实例化时预移动到 CUDA 设备

### CPU 回退机制
当检测到设备不匹配错误时，系统自动：
1. 将模块和输入临时移动到 CPU
2. 在 CPU 上执行前向传播
3. 将输出移回原始设备
4. 输出警告信息：`[WARN] {module_name} running on CPU due to device-mismatch`

### 张量形状验证
- **CNN 适配器**：验证 `seq_len` 能否构成正方形空间维度
- **错误处理**：无效形状时抛出 `ValueError`

## 使用示例

```python
# 获取注意力模块和适配器类型
module, adapter_type = get_attention_module(
    attention_type="se",  # 或其他任意类型
    d_model=512,
    num_heads=8,
    spatial_dim=7  # 对于 seq_len=49 的情况
)

# 创建适配器
adapter = get_attention_adapter(module, adapter_type)

# 前向传播
x = torch.randn(32, 49, 512)  # (batch_size, seq_len, d_model)
output = adapter(x)  # 输出形状: (32, 49, 512)
```

## 总结

项目通过适配器模式统一了不同注意力机制的接口，主要处理三类张量变换：
1. **QKV型**：直接传递 Q,K,V 张量
2. **CNN型**：2D↔3D 张量重塑 + 空间维度处理
3. **单输入型**：直接传递单个张量

所有注意力层的最终输入输出格式都统一为 `(batch_size, seq_len, d_model)`，确保了与 Transformer 架构的完美兼容。