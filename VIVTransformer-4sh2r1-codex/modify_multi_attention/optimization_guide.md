# 训练优化指南

## 批次大小 (Batch Size) 优化

### 问题诊断
你的日志显示 `batch_size` 在训练过程中不稳定 (170 → 106 → 170)，这通常由以下原因导致：

1. **最后一个小批次问题**：DataLoader 默认会包含数据集剩余的样本，导致最后一个batch比设定的小
2. **多卡分配不均**：DataParallel 在不同 batch 大小下的分配开销不同

### 解决方案

#### 1. 固定 Batch Size
- ✅ **已设置 `drop_last=True`**：确保训练时所有批次都是固定大小
- ✅ **验证/测试保持 `drop_last=False`**：确保使用全部数据进行评估

#### 2. DataLoader 优化
```python
# 训练集
train_loader = DataLoader(
    dataset, 
    batch_size=batch_size,
    shuffle=True,
    drop_last=True,          # 🔑 关键：固定batch大小
    pin_memory=True,         # GPU加速数据传输
    num_workers=4,           # 多进程加载（Windows设为0）
    prefetch_factor=2,       # 预取缓冲
    persistent_workers=True  # 保持worker进程
)

# 验证/测试集
eval_loader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=False,
    drop_last=False,         # 保持全部数据用于评估
    # ... 其他参数同上
)
```

### 批次大小选择策略

#### "甜点区"方法
1. **起始点**：从较小的安全值开始（如 16、32）
2. **逐步翻倍**：16 → 32 → 64 → 128 → 256
3. **监控指标**：
   - 吞吐量 (samples/sec)
   - 显存使用率 (70-85% 最佳)
   - 训练稳定性

#### 多卡配置建议 (2x46GB)
- **目标显存使用**: 每卡 32-40GB (70-85%)
- **推荐 batch_size**: 64-128 per GPU
- **总有效批次**: 使用梯度累积扩大到 512-1024

### 学习率调整
当改变批次大小时，需要相应调整学习率：
```python
# 线性缩放法则
new_lr = base_lr * (new_batch_size / base_batch_size)

# 配合 warmup
warmup_epochs = 5
final_lr = new_lr
warmup_lr = final_lr / warmup_epochs
```

## 性能监控

### 新增的日志指标
```
🔄 Epoch [1], Batch [1/100], BS: 64, Loss: 0.123456, 
   data: 15.2 ms, h2d: 2.1 ms, compute: 45.8 ms, total: 63.1 ms, samples/s: 1014.2
```

**指标解读**：
- `BS`: 当前批次大小（应该固定）
- `data`: 数据加载时间
- `h2d`: 主机到设备传输时间
- `compute`: 模型计算时间
- `total`: 总迭代时间
- `samples/s`: 吞吐量

### 瓶颈诊断
- **data 时间过长** → 增加 `num_workers`，启用 `pin_memory`
- **h2d 时间过长** → 检查数据类型，启用 `non_blocking=True`
- **compute 时间过长** → 启用混合精度 (AMP)，优化模型
- **samples/s 下降** → 批次过大，减小batch_size

## 高级优化

### 1. 混合精度训练
```python
use_amp = True
scaler = torch.cuda.amp.GradScaler()

with torch.cuda.amp.autocast():
    output = model(input)
    loss = criterion(output, target)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### 2. 梯度累积
```python
accumulation_steps = 8  # 有效batch = batch_size * accumulation_steps

for i, batch in enumerate(loader):
    loss = model(batch) / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### 3. 升级到 DDP (推荐)
```python
# 替换 DataParallel
model = torch.nn.parallel.DistributedDataParallel(model)

# 启动命令
python -m torch.distributed.launch --nproc_per_node=2 train.py
```

## 推荐配置

### 当前环境 (2x46GB GPU)
```python
# 最佳起始配置
batch_size_per_gpu = 64        # 每卡批次
total_batch_size = 128         # 双卡总批次  
accumulation_steps = 4         # 梯度累积
effective_batch = 512          # 有效批次

# 学习率调整
base_lr = 1e-4
effective_lr = base_lr * (effective_batch / 32)  # 假设基础batch=32
```

### 监控重点
1. 确认每个epoch的batch数量固定
2. samples/sec 在不同batch_size下的表现
3. 显存使用率稳定在 70-85%
4. Loss收敛稳定性