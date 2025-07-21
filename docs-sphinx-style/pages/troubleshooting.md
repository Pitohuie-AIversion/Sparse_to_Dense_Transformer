---
layout: default
title: Troubleshooting
description: 故障排除和问题解决
nav_order: 18
parent: 获取帮助
permalink: /pages/troubleshooting/
---

# 故障排除指南 {#故障排除指南}

本文档提供VIVTransformer项目常见问题的解决方案和调试技巧。

## 📋 目录 {#目录}

- [安装问题](#安装问题)
- [配置问题](#配置问题)
- [训练问题](#训练问题)
- [推理问题](#推理问题)
- [性能问题](#性能问题)
- [内存问题](#内存问题)
- [注意力机制问题](#注意力机制问题)
- [数据处理问题](#数据处理问题)
- [调试工具](#调试工具)
- [常见错误代码](#常见错误代码)

## 安装问题 {#安装问题}

### ❌ 依赖包安装失败 {#依赖包安装失败}

**问题描述**: 运行 `pip install -r requirements.txt` 时出现错误

**常见原因**:
1. Python版本不兼容
2. CUDA版本不匹配
3. 网络连接问题
4. 权限不足

**解决方案**:

```bash
# 1. 检查Python版本（需要3.8+） {#1-检查python版本-需要3-8}
python --version

# 2. 升级pip {#2-升级pip}
python -m pip install --upgrade pip

# 3. 使用国内镜像源 {#3-使用国内镜像源}
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/

# 4. 分步安装关键依赖 {#4-分步安装关键依赖}
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers
pip install -r requirements.txt

# 5. 使用conda环境（推荐） {#5-使用conda环境-推荐}
conda create -n vivtransformer python=3.9
conda activate vivtransformer
pip install -r requirements.txt
```

### ❌ CUDA相关错误 {#cuda相关错误}

**问题描述**: `RuntimeError: CUDA out of memory` 或 `CUDA device not found`

**解决方案**:

```python
# 检查CUDA可用性 {#检查cuda可用性}
import torch
print(f"CUDA可用: {torch.cuda.is_available()}")
print(f"CUDA版本: {torch.version.cuda}")
print(f"GPU数量: {torch.cuda.device_count()}")

# 如果CUDA不可用，强制使用CPU {#如果cuda不可用-强制使用cpu}
device = torch.device('cpu')
model = model.to(device)

# 清理GPU内存 {#清理gpu内存}
torch.cuda.empty_cache()
```

**CUDA版本兼容性**:
```bash
# 检查系统CUDA版本 {#检查系统cuda版本}
nvcc --version

# 安装对应的PyTorch版本 {#安装对应的pytorch版本}
# CUDA 11.8 {#cuda-11-8}
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1 {#cuda-12-1}
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# CPU版本 {#cpu版本}
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

## 配置问题 {#配置问题}

### ❌ 配置文件格式错误 {#配置文件格式错误}

**问题描述**: `yaml.scanner.ScannerError` 或配置解析失败

**解决方案**:

```python
# 验证YAML格式 {#验证yaml格式}
import yaml

def validate_config(config_path):
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        print("配置文件格式正确")
        return config
    except yaml.YAMLError as e:
        print(f"YAML格式错误: {e}")
        return None
    except Exception as e:
        print(f"配置文件错误: {e}")
        return None

# 使用示例 {#使用示例}
config = validate_config('config/default_config.yaml')
if config is None:
    print("请检查配置文件格式")
```

**常见YAML格式问题**:
```yaml
# ❌ 错误：缩进不一致 {#错误-缩进不一致}
model:
  name: vivtransformer
   hidden_size: 512  # 缩进错误

# ✅ 正确：统一使用2空格缩进 {#正确-统一使用2空格缩进}
model:
  name: vivtransformer
  hidden_size: 512

# ❌ 错误：冒号后没有空格 {#错误-冒号后没有空格}
learning_rate:0.001

# ✅ 正确：冒号后有空格 {#正确-冒号后有空格}
learning_rate: 0.001

# ❌ 错误：字符串包含特殊字符未加引号 {#错误-字符串包含特殊字符未加引号}
data_path: C:\Users\data  # Windows路径

# ✅ 正确：使用引号或正斜杠 {#正确-使用引号或正斜杠}
data_path: "C:\\Users\\data"
# 或 {#或}
data_path: C:/Users/data
```

### ❌ 参数验证失败 {#参数验证失败}

**问题描述**: 配置参数超出有效范围或类型错误

**解决方案**:

```python
# 配置验证函数 {#配置验证函数}
def validate_training_config(config):
    """验证训练配置"""
    errors = []
    
    # 检查学习率
    lr = config.get('learning_rate', 0)
    if not isinstance(lr, (int, float)) or lr <= 0:
        errors.append(f"学习率必须为正数，当前值: {lr}")
    
    # 检查批次大小
    batch_size = config.get('batch_size', 0)
    if not isinstance(batch_size, int) or batch_size <= 0:
        errors.append(f"批次大小必须为正整数，当前值: {batch_size}")
    
    # 检查注意力机制
    attention_type = config.get('attention_type', '')
    valid_attentions = ['standard', 'multi_head', 'sparse', 'linear']
    if attention_type not in valid_attentions:
        errors.append(f"无效的注意力类型: {attention_type}，有效选项: {valid_attentions}")
    
    # 检查设备配置
    device = config.get('device', 'cpu')
    if device == 'cuda' and not torch.cuda.is_available():
        errors.append("配置使用CUDA但CUDA不可用，将自动切换到CPU")
        config['device'] = 'cpu'
    
    if errors:
        print("配置验证失败:")
        for error in errors:
            print(f"  - {error}")
        return False
    
    print("配置验证通过")
    return True

# 使用示例 {#使用示例}
if not validate_training_config(config):
    print("请修正配置文件后重试")
    exit(1)
```

## 训练问题 {#训练问题}

### ❌ 损失不收敛 {#损失不收敛}

**问题描述**: 训练损失不下降或震荡严重

**诊断步骤**:

```python
# 1. 检查学习率 {#1-检查学习率}
def diagnose_learning_rate(model, dataloader, lr_range=(1e-6, 1e-1)):
    """学习率诊断"""
    import matplotlib.pyplot as plt
    
    lrs = []
    losses = []
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_range[0])
    criterion = torch.nn.MSELoss()
    
    for i, (data, target) in enumerate(dataloader):
        if i > 100:  # 只测试100个批次
            break
        
        # 指数增长学习率
        lr = lr_range[0] * (lr_range[1] / lr_range[0]) ** (i / 100)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        lrs.append(lr)
        losses.append(loss.item())
    
    # 绘制学习率-损失曲线
    plt.figure(figsize=(10, 6))
    plt.semilogx(lrs, losses)
    plt.xlabel('Learning Rate')
    plt.ylabel('Loss')
    plt.title('Learning Rate vs Loss')
    plt.grid(True)
    plt.savefig('lr_diagnosis.png')
    plt.show()
    
    # 找到最优学习率
    min_loss_idx = losses.index(min(losses))
    optimal_lr = lrs[min_loss_idx]
    print(f"建议学习率: {optimal_lr:.2e}")
    
    return optimal_lr

# 2. 检查梯度 {#2-检查梯度}
def check_gradients(model):
    """检查梯度状态"""
    total_norm = 0
    param_count = 0
    zero_grad_count = 0
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
            param_count += 1
            
            if param_norm.item() == 0:
                zero_grad_count += 1
                print(f"警告: {name} 的梯度为零")
        else:
            print(f"警告: {name} 没有梯度")
    
    total_norm = total_norm ** (1. / 2)
    
    print(f"总梯度范数: {total_norm:.6f}")
    print(f"零梯度参数数量: {zero_grad_count}/{param_count}")
    
    if total_norm < 1e-7:
        print("⚠️ 梯度过小，可能存在梯度消失问题")
    elif total_norm > 100:
        print("⚠️ 梯度过大，可能存在梯度爆炸问题")
    
    return total_norm

# 3. 数据检查 {#3-数据检查}
def check_data_distribution(dataloader):
    """检查数据分布"""
    all_data = []
    all_targets = []
    
    for data, target in dataloader:
        all_data.append(data)
        all_targets.append(target)
        if len(all_data) > 10:  # 只检查前10个批次
            break
    
    data_tensor = torch.cat(all_data, dim=0)
    target_tensor = torch.cat(all_targets, dim=0)
    
    print(f"数据统计:")
    print(f"  输入范围: [{data_tensor.min():.4f}, {data_tensor.max():.4f}]")
    print(f"  输入均值: {data_tensor.mean():.4f}")
    print(f"  输入标准差: {data_tensor.std():.4f}")
    print(f"  目标范围: [{target_tensor.min():.4f}, {target_tensor.max():.4f}]")
    print(f"  目标均值: {target_tensor.mean():.4f}")
    print(f"  目标标准差: {target_tensor.std():.4f}")
    
    # 检查是否需要标准化
    if abs(data_tensor.mean()) > 1 or data_tensor.std() > 10:
        print("⚠️ 建议对输入数据进行标准化")
    
    if abs(target_tensor.mean()) > 1 or target_tensor.std() > 10:
        print("⚠️ 建议对目标数据进行标准化")
```

**解决方案**:

```python
# 1. 调整学习率 {#1-调整学习率}
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=10, verbose=True
)

# 2. 梯度裁剪 {#2-梯度裁剪}
max_grad_norm = 1.0
torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

# 3. 权重初始化 {#3-权重初始化}
def init_weights(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)
    elif isinstance(m, torch.nn.LayerNorm):
        torch.nn.init.ones_(m.weight)
        torch.nn.init.zeros_(m.bias)

model.apply(init_weights)

# 4. 使用预热学习率 {#4-使用预热学习率}
class WarmupScheduler:
    def __init__(self, optimizer, warmup_steps, max_lr):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.max_lr = max_lr
        self.step_count = 0
    
    def step(self):
        self.step_count += 1
        if self.step_count <= self.warmup_steps:
            lr = self.max_lr * self.step_count / self.warmup_steps
        else:
            lr = self.max_lr
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

warmup_scheduler = WarmupScheduler(optimizer, warmup_steps=1000, max_lr=1e-3)
```

### ❌ 过拟合问题 {#过拟合问题}

**问题描述**: 训练损失下降但验证损失上升

**解决方案**:

```python
# 1. 添加正则化 {#1-添加正则化}
class RegularizedModel(torch.nn.Module):
    def __init__(self, base_model, dropout_rate=0.1, weight_decay=1e-4):
        super().__init__()
        self.base_model = base_model
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.weight_decay = weight_decay
    
    def forward(self, x):
        x = self.dropout(x)
        return self.base_model(x)
    
    def l2_regularization(self):
        l2_reg = torch.tensor(0.0)
        for param in self.parameters():
            l2_reg += torch.norm(param, 2)
        return self.weight_decay * l2_reg

# 2. 早停机制 {#2-早停机制}
class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
    
    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
        
        return self.counter >= self.patience

# 使用示例 {#使用示例}
early_stopping = EarlyStopping(patience=15)

for epoch in range(num_epochs):
    # 训练...
    val_loss = validate(model, val_dataloader)
    
    if early_stopping(val_loss):
        print(f"早停在第 {epoch} 轮")
        break

# 3. 数据增强 {#3-数据增强}
from torchvision import transforms

transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
```

### ❌ 训练速度慢 {#训练速度慢}

**问题描述**: 训练时间过长，效率低下

**性能优化**:

```python
# 1. 混合精度训练 {#1-混合精度训练}
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for data, target in dataloader:
    optimizer.zero_grad()
    
    with autocast():
        output = model(data)
        loss = criterion(output, target)
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

# 2. 数据加载优化 {#2-数据加载优化}
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=4,  # 增加工作进程
    pin_memory=True,  # 固定内存
    persistent_workers=True  # 持久化工作进程
)

# 3. 模型编译（PyTorch 2.0+） {#3-模型编译-pytorch-2-0}
model = torch.compile(model)

# 4. 梯度累积 {#4-梯度累积}
accumulation_steps = 4

for i, (data, target) in enumerate(dataloader):
    output = model(data)
    loss = criterion(output, target) / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()

# 5. 检查点保存优化 {#5-检查点保存优化}
def save_checkpoint(model, optimizer, epoch, loss, filepath):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, filepath)

# 每N个epoch保存一次 {#每n个epoch保存一次}
if epoch % save_interval == 0:
    save_checkpoint(model, optimizer, epoch, loss, f'checkpoint_epoch_{epoch}.pth')
```

## 推理问题 {#推理问题}

### ❌ 推理结果异常 {#推理结果异常}

**问题描述**: 模型推理输出NaN、Inf或不合理的值

**诊断和解决**:

```python
# 1. 检查模型状态 {#1-检查模型状态}
def check_model_health(model):
    """检查模型健康状态"""
    for name, param in model.named_parameters():
        if torch.isnan(param).any():
            print(f"⚠️ 参数 {name} 包含NaN")
        if torch.isinf(param).any():
            print(f"⚠️ 参数 {name} 包含Inf")
        if param.abs().max() > 1e6:
            print(f"⚠️ 参数 {name} 数值过大: {param.abs().max()}")

# 2. 安全推理函数 {#2-安全推理函数}
def safe_inference(model, input_data, check_output=True):
    """安全推理，包含异常检测"""
    model.eval()
    
    # 检查输入
    if torch.isnan(input_data).any():
        raise ValueError("输入数据包含NaN")
    if torch.isinf(input_data).any():
        raise ValueError("输入数据包含Inf")
    
    with torch.no_grad():
        try:
            output = model(input_data)
            
            if check_output:
                if torch.isnan(output).any():
                    print("⚠️ 输出包含NaN")
                    return None
                if torch.isinf(output).any():
                    print("⚠️ 输出包含Inf")
                    return None
            
            return output
            
        except RuntimeError as e:
            print(f"推理错误: {e}")
            return None

# 3. 输出范围检查 {#3-输出范围检查}
def validate_output_range(output, expected_range=None):
    """验证输出范围"""
    if expected_range is None:
        expected_range = (-10, 10)  # 默认合理范围
    
    min_val, max_val = expected_range
    
    if output.min() < min_val or output.max() > max_val:
        print(f"⚠️ 输出超出预期范围 {expected_range}")
        print(f"实际范围: [{output.min():.4f}, {output.max():.4f}]")
        return False
    
    return True

# 使用示例 {#使用示例}
output = safe_inference(model, input_data)
if output is not None:
    if validate_output_range(output, (-1, 1)):
        print("推理结果正常")
    else:
        print("推理结果异常")
```

### ❌ 推理速度慢 {#推理速度慢}

**优化方案**:

```python
# 1. 模型量化 {#1-模型量化}
import torch.quantization as quantization

# 动态量化 {#动态量化}
quantized_model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)

# 2. 模型剪枝 {#2-模型剪枝}
import torch.nn.utils.prune as prune

# 结构化剪枝 {#结构化剪枝}
for module in model.modules():
    if isinstance(module, torch.nn.Linear):
        prune.l1_unstructured(module, name='weight', amount=0.2)

# 3. 批量推理 {#3-批量推理}
def batch_inference(model, data_list, batch_size=32):
    """批量推理"""
    results = []
    
    for i in range(0, len(data_list), batch_size):
        batch = data_list[i:i+batch_size]
        batch_tensor = torch.stack(batch)
        
        with torch.no_grad():
            batch_output = model(batch_tensor)
            results.extend(batch_output.cpu().numpy())
    
    return results

# 4. TorchScript优化 {#4-torchscript优化}
scripted_model = torch.jit.script(model)
scripted_model.save('optimized_model.pt')

# 加载优化模型 {#加载优化模型}
optimized_model = torch.jit.load('optimized_model.pt')
```

## 内存问题 {#内存问题}

### ❌ GPU内存不足 {#gpu内存不足}

**问题描述**: `CUDA out of memory`

**解决方案**:

```python
# 1. 内存监控 {#1-内存监控}
def monitor_gpu_memory():
    """监控GPU内存使用"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        reserved = torch.cuda.memory_reserved() / 1024**3   # GB
        print(f"GPU内存 - 已分配: {allocated:.2f}GB, 已保留: {reserved:.2f}GB")
    else:
        print("CUDA不可用")

# 2. 内存清理 {#2-内存清理}
def clear_gpu_memory():
    """清理GPU内存"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        print("GPU内存已清理")

# 3. 梯度检查点 {#3-梯度检查点}
from torch.utils.checkpoint import checkpoint

class CheckpointedModel(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, x):
        # 使用梯度检查点节省内存
        return checkpoint(self.model, x)

# 4. 动态批次大小 {#4-动态批次大小}
def find_optimal_batch_size(model, input_shape, max_batch_size=128):
    """找到最优批次大小"""
    model.eval()
    
    for batch_size in range(1, max_batch_size + 1):
        try:
            # 创建测试输入
            test_input = torch.randn(batch_size, *input_shape[1:])
            if torch.cuda.is_available():
                test_input = test_input.cuda()
                model = model.cuda()
            
            with torch.no_grad():
                _ = model(test_input)
            
            print(f"批次大小 {batch_size} 可用")
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"批次大小 {batch_size} 内存不足")
                return batch_size - 1
            else:
                raise e
        
        finally:
            clear_gpu_memory()
    
    return max_batch_size

# 5. 内存高效的数据加载 {#5-内存高效的数据加载}
class MemoryEfficientDataset(torch.utils.data.Dataset):
    def __init__(self, data_paths):
        self.data_paths = data_paths
    
    def __len__(self):
        return len(self.data_paths)
    
    def __getitem__(self, idx):
        # 动态加载数据，而不是预加载到内存
        data = torch.load(self.data_paths[idx])
        return data

# 使用示例 {#使用示例}
optimal_batch_size = find_optimal_batch_size(model, (1, 3, 224, 224))
print(f"建议批次大小: {optimal_batch_size}")
```

## 注意力机制问题 {#注意力机制问题}

### ❌ 注意力权重异常 {#注意力权重异常}

**问题描述**: 注意力权重全为0、全为1或分布不合理

**诊断和修复**:

```python
# 1. 注意力权重诊断 {#1-注意力权重诊断}
def diagnose_attention_weights(attention_weights):
    """诊断注意力权重"""
    # attention_weights: [batch_size, num_heads, seq_len, seq_len]
    
    print("注意力权重诊断:")
    print(f"  形状: {attention_weights.shape}")
    print(f"  范围: [{attention_weights.min():.6f}, {attention_weights.max():.6f}]")
    print(f"  均值: {attention_weights.mean():.6f}")
    print(f"  标准差: {attention_weights.std():.6f}")
    
    # 检查是否包含NaN或Inf
    if torch.isnan(attention_weights).any():
        print("⚠️ 注意力权重包含NaN")
    if torch.isinf(attention_weights).any():
        print("⚠️ 注意力权重包含Inf")
    
    # 检查是否正确归一化
    row_sums = attention_weights.sum(dim=-1)
    if not torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-6):
        print("⚠️ 注意力权重未正确归一化")
        print(f"  行和范围: [{row_sums.min():.6f}, {row_sums.max():.6f}]")
    
    # 检查稀疏性
    zero_ratio = (attention_weights == 0).float().mean()
    print(f"  零值比例: {zero_ratio:.4f}")
    
    # 检查对角线优势（自注意力）
    if attention_weights.shape[-2] == attention_weights.shape[-1]:
        diag_values = torch.diagonal(attention_weights, dim1=-2, dim2=-1)
        diag_mean = diag_values.mean()
        off_diag_mean = (attention_weights.sum(dim=(-2, -1)) - diag_values.sum(dim=-1)) / (attention_weights.shape[-1] * (attention_weights.shape[-1] - 1))
        print(f"  对角线均值: {diag_mean:.6f}")
        print(f"  非对角线均值: {off_diag_mean:.6f}")

# 2. 修复注意力计算 {#2-修复注意力计算}
class RobustAttention(torch.nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1, temperature=1.0):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.temperature = temperature
        
        self.w_q = torch.nn.Linear(d_model, d_model)
        self.w_k = torch.nn.Linear(d_model, d_model)
        self.w_v = torch.nn.Linear(d_model, d_model)
        self.w_o = torch.nn.Linear(d_model, d_model)
        
        self.dropout = torch.nn.Dropout(dropout)
        self.layer_norm = torch.nn.LayerNorm(d_model)
        
    def forward(self, query, key, value, mask=None):
        batch_size, seq_len, d_model = query.shape
        
        # 线性变换
        Q = self.w_q(query).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (math.sqrt(self.d_k) * self.temperature)
        
        # 应用掩码
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # 稳定的softmax
        attention_weights = self.stable_softmax(scores)
        
        # 应用dropout
        attention_weights = self.dropout(attention_weights)
        
        # 计算输出
        context = torch.matmul(attention_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        
        output = self.w_o(context)
        
        # 残差连接和层归一化
        output = self.layer_norm(output + query)
        
        return output, attention_weights
    
    def stable_softmax(self, x, dim=-1):
        """数值稳定的softmax"""
        # 减去最大值以提高数值稳定性
        x_max = torch.max(x, dim=dim, keepdim=True)[0]
        x_stable = x - x_max
        
        # 计算softmax
        exp_x = torch.exp(x_stable)
        sum_exp_x = torch.sum(exp_x, dim=dim, keepdim=True)
        
        # 避免除零
        sum_exp_x = torch.clamp(sum_exp_x, min=1e-8)
        
        return exp_x / sum_exp_x

# 3. 注意力可视化 {#3-注意力可视化}
def visualize_attention(attention_weights, save_path='attention_heatmap.png'):
    """可视化注意力权重"""
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # 取第一个样本的第一个头
    att_matrix = attention_weights[0, 0].cpu().numpy()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(att_matrix, cmap='Blues', cbar=True)
    plt.title('Attention Weights Heatmap')
    plt.xlabel('Key Position')
    plt.ylabel('Query Position')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"注意力热图已保存到: {save_path}")
```

## 调试工具 {#调试工具}

### 🔧 调试辅助函数 {#调试辅助函数}

```python
# 1. 模型结构检查 {#1-模型结构检查}
def print_model_structure(model, input_shape):
    """打印模型结构和参数信息"""
    from torchsummary import summary
    
    print("模型结构:")
    summary(model, input_shape)
    
    print("\n详细参数信息:")
    total_params = 0
    trainable_params = 0
    
    for name, param in model.named_parameters():
        param_count = param.numel()
        total_params += param_count
        
        if param.requires_grad:
            trainable_params += param_count
        
        print(f"{name:50} {str(param.shape):20} {param_count:>10,} {'✓' if param.requires_grad else '✗'}")
    
    print(f"\n总参数量: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")
    print(f"不可训练参数: {total_params - trainable_params:,}")

# 2. 训练监控 {#2-训练监控}
class TrainingMonitor:
    def __init__(self, log_interval=10):
        self.log_interval = log_interval
        self.losses = []
        self.learning_rates = []
        self.gradient_norms = []
        
    def log_step(self, loss, lr, grad_norm):
        self.losses.append(loss)
        self.learning_rates.append(lr)
        self.gradient_norms.append(grad_norm)
        
        if len(self.losses) % self.log_interval == 0:
            avg_loss = sum(self.losses[-self.log_interval:]) / self.log_interval
            print(f"步骤 {len(self.losses)}: 平均损失={avg_loss:.6f}, 学习率={lr:.2e}, 梯度范数={grad_norm:.6f}")
    
    def plot_metrics(self, save_path='training_metrics.png'):
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 损失曲线
        axes[0, 0].plot(self.losses)
        axes[0, 0].set_title('Training Loss')
        axes[0, 0].set_xlabel('Step')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].grid(True)
        
        # 学习率曲线
        axes[0, 1].plot(self.learning_rates)
        axes[0, 1].set_title('Learning Rate')
        axes[0, 1].set_xlabel('Step')
        axes[0, 1].set_ylabel('LR')
        axes[0, 1].set_yscale('log')
        axes[0, 1].grid(True)
        
        # 梯度范数
        axes[1, 0].plot(self.gradient_norms)
        axes[1, 0].set_title('Gradient Norm')
        axes[1, 0].set_xlabel('Step')
        axes[1, 0].set_ylabel('Norm')
        axes[1, 0].grid(True)
        
        # 损失分布
        axes[1, 1].hist(self.losses, bins=50, alpha=0.7)
        axes[1, 1].set_title('Loss Distribution')
        axes[1, 1].set_xlabel('Loss')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

# 3. 数据检查工具 {#3-数据检查工具}
def check_dataset_health(dataset, num_samples=100):
    """检查数据集健康状态"""
    print(f"检查数据集健康状态（样本数: {num_samples}）...")
    
    issues = []
    
    for i in range(min(num_samples, len(dataset))):
        try:
            data, target = dataset[i]
            
            # 检查数据类型
            if not isinstance(data, torch.Tensor):
                issues.append(f"样本 {i}: 数据不是Tensor类型")
            
            if not isinstance(target, torch.Tensor):
                issues.append(f"样本 {i}: 标签不是Tensor类型")
            
            # 检查NaN和Inf
            if torch.isnan(data).any():
                issues.append(f"样本 {i}: 数据包含NaN")
            
            if torch.isinf(data).any():
                issues.append(f"样本 {i}: 数据包含Inf")
            
            if torch.isnan(target).any():
                issues.append(f"样本 {i}: 标签包含NaN")
            
            if torch.isinf(target).any():
                issues.append(f"样本 {i}: 标签包含Inf")
            
        except Exception as e:
            issues.append(f"样本 {i}: 加载失败 - {e}")
    
    if issues:
        print(f"发现 {len(issues)} 个问题:")
        for issue in issues[:10]:  # 只显示前10个问题
            print(f"  - {issue}")
        if len(issues) > 10:
            print(f"  ... 还有 {len(issues) - 10} 个问题")
    else:
        print("✅ 数据集健康状态良好")
    
    return len(issues) == 0
```

## 常见错误代码 {#常见错误代码}

### 📋 错误代码对照表 {#错误代码对照表}

| 错误代码 | 错误描述 | 可能原因 | 解决方案 |
|---------|---------|---------|----------|
| `CUDA_ERROR_OUT_OF_MEMORY` | GPU内存不足 | 批次大小过大、模型过大 | 减小批次大小、使用梯度累积 |
| `RuntimeError: Expected all tensors to be on the same device` | 设备不匹配 | 模型和数据在不同设备 | 确保模型和数据在同一设备 |
| `ValueError: Target size mismatch` | 目标尺寸不匹配 | 输出和标签维度不一致 | 检查模型输出维度和标签格式 |
| `KeyError: 'attention_type'` | 配置键缺失 | 配置文件缺少必要参数 | 补充配置文件或使用默认值 |
| `ImportError: No module named 'xxx'` | 模块导入失败 | 依赖包未安装 | 安装缺失的依赖包 |
| `FileNotFoundError` | 文件未找到 | 路径错误或文件不存在 | 检查文件路径和权限 |
| `yaml.scanner.ScannerError` | YAML格式错误 | 配置文件格式不正确 | 检查YAML语法和缩进 |
| `torch.nn.modules.module.ModuleAttributeError` | 模块属性错误 | 模型结构定义错误 | 检查模型定义和属性名称 |

### 🔍 快速诊断脚本 {#快速诊断脚本}

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*- {#coding-utf-8}
"""
快速诊断脚本
用于检查VIVTransformer项目的常见问题
"""

import torch
import yaml
import os
import sys
from pathlib import Path

def quick_diagnosis():
    """快速诊断系统状态"""
    print("=" * 50)
    print("VIVTransformer 快速诊断")
    print("=" * 50)
    
    # 1. Python环境检查
    print(f"\n1. Python环境:")
    print(f"   Python版本: {sys.version}")
    print(f"   PyTorch版本: {torch.__version__}")
    
    # 2. CUDA检查
    print(f"\n2. CUDA环境:")
    print(f"   CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   CUDA版本: {torch.version.cuda}")
        print(f"   GPU数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
    
    # 3. 依赖包检查
    print(f"\n3. 依赖包检查:")
    required_packages = ['numpy', 'matplotlib', 'seaborn', 'tqdm', 'transformers']
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"   ✅ {package}: 已安装")
        except ImportError:
            print(f"   ❌ {package}: 未安装")
    
    # 4. 配置文件检查
    print(f"\n4. 配置文件检查:")
    config_files = ['config/default_config.yaml', 'config/training_config.yaml']
    
    for config_file in config_files:
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    yaml.safe_load(f)
                print(f"   ✅ {config_file}: 格式正确")
            except yaml.YAMLError as e:
                print(f"   ❌ {config_file}: YAML格式错误 - {e}")
            except Exception as e:
                print(f"   ❌ {config_file}: 读取失败 - {e}")
        else:
            print(f"   ⚠️ {config_file}: 文件不存在")
    
    # 5. 数据目录检查
    print(f"\n5. 数据目录检查:")
    data_dirs = ['data/', 'datasets/', 'checkpoints/']
    
    for data_dir in data_dirs:
        if os.path.exists(data_dir):
            file_count = len(list(Path(data_dir).rglob('*')))
            print(f"   ✅ {data_dir}: 存在 ({file_count} 个文件)")
        else:
            print(f"   ⚠️ {data_dir}: 不存在")
    
    # 6. 内存检查
    print(f"\n6. 内存状态:")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            total_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            allocated_memory = torch.cuda.memory_allocated(i) / 1024**3
            print(f"   GPU {i}: {allocated_memory:.2f}GB / {total_memory:.2f}GB")
    
    import psutil
    memory = psutil.virtual_memory()
    print(f"   系统内存: {memory.used / 1024**3:.2f}GB / {memory.total / 1024**3:.2f}GB ({memory.percent:.1f}%)")
    
    print(f"\n=" * 50)
    print("诊断完成")
    print("=" * 50)

if __name__ == "__main__":
    quick_diagnosis()
```

---

**💡 提示**: 遇到问题时，建议按照以下顺序进行排查：
1. 运行快速诊断脚本
2. 检查错误日志和堆栈跟踪
3. 查阅本故障排除指南
4. 在项目Issues中搜索类似问题
5. 如果问题仍未解决，请提交详细的错误报告

**🔧 调试技巧**: 
- 使用小数据集进行快速测试
- 逐步增加模型复杂度
- 保存中间结果进行分析
- 使用可视化工具辅助调试

---

*需要帮助？查看 [FAQ](faq) 或 [故障排除](troubleshooting) 页面。*
