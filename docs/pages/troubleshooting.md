---
layout: doc
title: Troubleshooting
description: Troubleshooting and problem solving
nav_order: 18
parent: Getting Help
permalink: /pages/troubleshooting/
---

<div data-lang-zh style="display: none;">
<h1 id="故障排除指南">故障排除指南</h1>

<p>本文档提供VIVTransformer项目常见问题的解决方案和调试技巧。</p>

<h2 id="目录">📋 目录</h2>

<ul>
<li><a href="#安装问题">安装问题</a></li>
<li><a href="#配置问题">配置问题</a></li>
<li><a href="#训练问题">训练问题</a></li>
<li><a href="#推理问题">推理问题</a></li>
<li><a href="#性能问题">性能问题</a></li>
<li><a href="#内存问题">内存问题</a></li>
<li><a href="#注意力机制问题">注意力机制问题</a></li>
<li><a href="#数据处理问题">数据处理问题</a></li>
<li><a href="#调试工具">调试工具</a></li>
<li><a href="#常见错误代码">常见错误代码</a></li>
</ul>

<h2 id="安装问题">安装问题</h2>

<h3 id="依赖包安装失败">❌ 依赖包安装失败</h3>

<p><strong>问题描述</strong>: 运行 <code>pip install -r requirements.txt</code> 时出现错误</p>

<p><strong>常见原因</strong>:</p>
<ol>
<li>Python版本不兼容</li>
<li>CUDA版本不匹配</li>
<li>网络连接问题</li>
<li>权限不足</li>
</ol>

<p><strong>解决方案</strong>:</p>

<pre><code class="language-bash"># 1. 检查Python版本（需要3.8+）
python --version

# 2. 升级pip
python -m pip install --upgrade pip

# 3. 使用国内镜像源
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/

# 4. 分步安装关键依赖
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers
pip install -r requirements.txt

# 5. 使用conda环境（推荐）
conda create -n vivtransformer python=3.9
conda activate vivtransformer
pip install -r requirements.txt
</code></pre>

<h3 id="cuda相关错误">❌ CUDA相关错误</h3>

<p><strong>问题描述</strong>: <code>RuntimeError: CUDA out of memory</code> 或 <code>CUDA device not found</code></p>

<p><strong>解决方案</strong>:</p>

<pre><code class="language-python"># 检查CUDA可用性
import torch
print(f"CUDA可用: {torch.cuda.is_available()}")
print(f"CUDA版本: {torch.version.cuda}")
print(f"GPU数量: {torch.cuda.device_count()}")

# 如果CUDA不可用，强制使用CPU
device = torch.device('cpu')
model = model.to(device)

# 清理GPU内存
torch.cuda.empty_cache()
</code></pre>

<p><strong>CUDA版本兼容性</strong>:</p>
<pre><code class="language-bash"># 检查系统CUDA版本
nvcc --version

# 安装对应的PyTorch版本
# CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# CPU版本
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
</code></pre>

<h2 id="配置问题">配置问题</h2>

<h3 id="配置文件格式错误">❌ 配置文件格式错误</h3>

<p><strong>问题描述</strong>: <code>yaml.scanner.ScannerError</code> 或配置解析失败</p>

<p><strong>解决方案</strong>:</p>

<pre><code class="language-python"># 验证YAML格式
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

# 使用示例
config = validate_config('config/default_config.yaml')
if config is None:
    print("请检查配置文件格式")
</code></pre>

<p><strong>常见YAML格式问题</strong>:</p>
<pre><code class="language-yaml"># ❌ 错误：缩进不一致
model:
  name: vivtransformer
   hidden_size: 512  # 缩进错误

# ✅ 正确：统一使用2空格缩进
model:
  name: vivtransformer
  hidden_size: 512

# ❌ 错误：冒号后没有空格
learning_rate:0.001

# ✅ 正确：冒号后有空格
learning_rate: 0.001

# ❌ 错误：字符串包含特殊字符未加引号
data_path: C:\Users\data  # Windows路径

# ✅ 正确：使用引号或正斜杠
data_path: "C:\\Users\\data"
# 或
data_path: C:/Users/data
</code></pre>

<h3 id="参数验证失败">❌ 参数验证失败</h3>

<p><strong>问题描述</strong>: <code>ValueError: Invalid parameter value</code></p>

<p><strong>解决方案</strong>:</p>

<pre><code class="language-python"># 参数验证工具
def validate_model_config(config):
    """验证模型配置参数"""
    errors = []
    
    # 检查必需参数
    required_params = ['d_model', 'n_heads', 'n_layers']
    for param in required_params:
        if param not in config:
            errors.append(f"缺少必需参数: {param}")
    
    # 检查参数范围
    if 'd_model' in config:
        if config['d_model'] <= 0 or config['d_model'] % config.get('n_heads', 8) != 0:
            errors.append("d_model必须是正数且能被n_heads整除")
    
    if 'learning_rate' in config:
        if not (1e-6 <= config['learning_rate'] <= 1.0):
            errors.append("learning_rate必须在1e-6到1.0之间")
    
    if errors:
        raise ValueError("\n".join(errors))
    
    return True

# 使用示例
try:
    validate_model_config(config['model'])
    print("配置验证通过")
except ValueError as e:
    print(f"配置验证失败: {e}")
</code></pre>

<h2 id="训练问题">训练问题</h2>

<h3 id="训练不收敛">❌ 训练不收敛</h3>

<p><strong>问题描述</strong>: 损失函数不下降或震荡严重</p>

<p><strong>可能原因</strong>:</p>
<ol>
<li>学习率过大或过小</li>
<li>批次大小不合适</li>
<li>梯度爆炸或消失</li>
<li>数据预处理问题</li>
</ol>

<p><strong>解决方案</strong>:</p>

<pre><code class="language-python"># 1. 学习率调试
import matplotlib.pyplot as plt

def find_learning_rate(model, dataloader, init_lr=1e-8, final_lr=10):
    """学习率范围测试"""
    lrs = []
    losses = []
    
    optimizer = torch.optim.Adam(model.parameters(), lr=init_lr)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=1.1)
    
    for batch_idx, (data, target) in enumerate(dataloader):
        optimizer.zero_grad()
        output = model(data)
        loss = F.mse_loss(output, target)
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        
        lrs.append(optimizer.param_groups[0]['lr'])
        losses.append(loss.item())
        
        if optimizer.param_groups[0]['lr'] > final_lr:
            break
    
    # 绘制学习率-损失曲线
    plt.figure(figsize=(10, 6))
    plt.semilogx(lrs, losses)
    plt.xlabel('Learning Rate')
    plt.ylabel('Loss')
    plt.title('Learning Rate Range Test')
    plt.show()
    
    return lrs, losses

# 2. 梯度监控
def monitor_gradients(model):
    """监控梯度统计"""
    total_norm = 0
    param_count = 0
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
            param_count += 1
            
            # 检查梯度异常
            if torch.isnan(param.grad).any():
                print(f"警告: {name} 包含NaN梯度")
            if torch.isinf(param.grad).any():
                print(f"警告: {name} 包含无穷梯度")
    
    total_norm = total_norm ** (1. / 2)
    print(f"梯度范数: {total_norm:.6f}")
    
    return total_norm

# 3. 损失函数调试
class DebugLoss(nn.Module):
    def __init__(self, base_loss):
        super().__init__()
        self.base_loss = base_loss
        self.loss_history = []
    
    def forward(self, pred, target):
        loss = self.base_loss(pred, target)
        
        # 记录损失统计
        self.loss_history.append(loss.item())
        
        # 检查异常值
        if torch.isnan(loss):
            print("警告: 损失为NaN")
            print(f"预测统计: min={pred.min():.6f}, max={pred.max():.6f}, mean={pred.mean():.6f}")
            print(f"目标统计: min={target.min():.6f}, max={target.max():.6f}, mean={target.mean():.6f}")
        
        return loss
</code></pre>

<h3 id="过拟合问题">❌ 过拟合问题</h3>

<p><strong>问题描述</strong>: 训练损失下降但验证损失上升</p>

<p><strong>解决方案</strong>:</p>

<pre><code class="language-python"># 1. 早停机制
class EarlyStopping:
    def __init__(self, patience=7, min_delta=0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None
    
    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1
        
        if self.counter >= self.patience:
            if self.restore_best_weights:
                model.load_state_dict(self.best_weights)
            return True
        return False
    
    def save_checkpoint(self, model):
        self.best_weights = model.state_dict().copy()

# 2. 正则化技术
class RegularizedVIVTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 基础模型
        self.base_model = VIVTransformer(config)
        
        # Dropout层
        self.dropout = nn.Dropout(config.get('dropout', 0.1))
        
        # 权重衰减将在优化器中设置
    
    def forward(self, x):
        x = self.dropout(x)
        return self.base_model(x)
    
    def get_l1_loss(self):
        """计算L1正则化损失"""
        l1_loss = 0
        for param in self.parameters():
            l1_loss += torch.sum(torch.abs(param))
        return l1_loss
    
    def get_l2_loss(self):
        """计算L2正则化损失"""
        l2_loss = 0
        for param in self.parameters():
            l2_loss += torch.sum(param ** 2)
        return l2_loss

# 3. 数据增强
def add_noise_augmentation(data, noise_factor=0.01):
    """添加噪声增强"""
    noise = torch.randn_like(data) * noise_factor
    return data + noise

def temporal_masking(data, mask_ratio=0.1):
    """时间掩蔽增强"""
    batch_size, seq_len, features = data.shape
    mask_len = int(seq_len * mask_ratio)
    
    for i in range(batch_size):
        start_idx = torch.randint(0, seq_len - mask_len, (1,))
        data[i, start_idx:start_idx + mask_len] = 0
    
    return data
</code></pre>

<h2 id="推理问题">推理问题</h2>

<h3 id="推理速度慢">❌ 推理速度慢</h3>

<p><strong>问题描述</strong>: 模型推理时间过长</p>

<p><strong>解决方案</strong>:</p>

<pre><code class="language-python"># 1. 模型优化
import torch.jit

# TorchScript编译
model.eval()
scripted_model = torch.jit.script(model)

# 或使用trace
sample_input = torch.randn(1, 100, 512)
traced_model = torch.jit.trace(model, sample_input)

# 2. 推理加速
@torch.no_grad()
def fast_inference(model, data):
    """快速推理"""
    model.eval()
    
    # 使用半精度
    if torch.cuda.is_available():
        model = model.half()
        data = data.half()
    
    # 批量推理
    results = []
    batch_size = 32
    
    for i in range(0, len(data), batch_size):
        batch = data[i:i + batch_size]
        output = model(batch)
        results.append(output)
    
    return torch.cat(results, dim=0)

# 3. 内存优化
def memory_efficient_inference(model, data, chunk_size=1000):
    """内存高效推理"""
    model.eval()
    results = []
    
    with torch.no_grad():
        for i in range(0, data.size(1), chunk_size):
            chunk = data[:, i:i + chunk_size]
            output = model(chunk)
            results.append(output.cpu())  # 移到CPU释放GPU内存
            torch.cuda.empty_cache()
    
    return torch.cat(results, dim=1)
</code></pre>

<h2 id="性能问题">性能问题</h2>

<h3 id="训练速度慢">❌ 训练速度慢</h3>

<p><strong>解决方案</strong>:</p>

<pre><code class="language-python"># 1. 数据加载优化
from torch.utils.data import DataLoader

# 优化DataLoader
dataloader = DataLoader(
    dataset,
    batch_size=32,
    num_workers=4,  # 多进程加载
    pin_memory=True,  # 固定内存
    persistent_workers=True,  # 持久化worker
    prefetch_factor=2  # 预取因子
)

# 2. 混合精度训练
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

# 3. 梯度累积
accumulation_steps = 4

for i, (data, target) in enumerate(dataloader):
    with autocast():
        output = model(data)
        loss = criterion(output, target) / accumulation_steps
    
    scaler.scale(loss).backward()
    
    if (i + 1) % accumulation_steps == 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
</code></pre>

<h2 id="内存问题">内存问题</h2>

<h3 id="gpu内存不足">❌ GPU内存不足</h3>

<p><strong>解决方案</strong>:</p>

<pre><code class="language-python"># 1. 内存监控
def monitor_gpu_memory():
    """监控GPU内存使用"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        cached = torch.cuda.memory_reserved() / 1024**3
        print(f"GPU内存 - 已分配: {allocated:.2f}GB, 已缓存: {cached:.2f}GB")

# 2. 梯度检查点
from torch.utils.checkpoint import checkpoint

class CheckpointedTransformerLayer(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer
    
    def forward(self, x):
        return checkpoint(self.layer, x)

# 3. 模型并行
class ModelParallelVIVTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        # 将不同层放在不同GPU上
        self.embedding = nn.Linear(config.input_dim, config.d_model).to('cuda:0')
        self.transformer_layers = nn.ModuleList([
            TransformerLayer(config).to(f'cuda:{i % torch.cuda.device_count()}')
            for i in range(config.n_layers)
        ])
        self.output_layer = nn.Linear(config.d_model, config.output_dim).to('cuda:0')
    
    def forward(self, x):
        x = x.to('cuda:0')
        x = self.embedding(x)
        
        for i, layer in enumerate(self.transformer_layers):
            device = f'cuda:{i % torch.cuda.device_count()}'
            x = x.to(device)
            x = layer(x)
        
        x = x.to('cuda:0')
        return self.output_layer(x)
</code></pre>

<h2 id="注意力机制问题">注意力机制问题</h2>

<h3 id="注意力权重异常">❌ 注意力权重异常</h3>

<p><strong>解决方案</strong>:</p>

<pre><code class="language-python"># 1. 注意力可视化
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_attention(attention_weights, save_path=None):
    """可视化注意力权重"""
    # attention_weights: [batch_size, n_heads, seq_len, seq_len]
    
    batch_size, n_heads, seq_len, _ = attention_weights.shape
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    for head in range(min(8, n_heads)):
        attention_map = attention_weights[0, head].detach().cpu().numpy()
        
        sns.heatmap(
            attention_map,
            ax=axes[head],
            cmap='Blues',
            cbar=True,
            square=True
        )
        axes[head].set_title(f'Head {head + 1}')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

# 2. 注意力统计分析
def analyze_attention_patterns(attention_weights):
    """分析注意力模式"""
    # 计算注意力熵
    entropy = -torch.sum(attention_weights * torch.log(attention_weights + 1e-8), dim=-1)
    
    # 计算注意力集中度
    max_attention = torch.max(attention_weights, dim=-1)[0]
    
    # 统计信息
    stats = {
        'entropy_mean': entropy.mean().item(),
        'entropy_std': entropy.std().item(),
        'max_attention_mean': max_attention.mean().item(),
        'max_attention_std': max_attention.std().item(),
    }
    
    print("注意力统计:")
    for key, value in stats.items():
        print(f"  {key}: {value:.4f}")
    
    return stats

# 3. 注意力正则化
class RegularizedAttention(nn.Module):
    def __init__(self, attention_module, entropy_weight=0.01):
        super().__init__()
        self.attention = attention_module
        self.entropy_weight = entropy_weight
    
    def forward(self, x):
        output, attention_weights = self.attention(x)
        
        # 计算熵正则化损失
        entropy = -torch.sum(attention_weights * torch.log(attention_weights + 1e-8), dim=-1)
        entropy_loss = -entropy.mean()  # 负熵，鼓励多样性
        
        return output, attention_weights, entropy_loss * self.entropy_weight
</code></pre>

<h2 id="数据处理问题">数据处理问题</h2>

<h3 id="数据加载错误">❌ 数据加载错误</h3>

<p><strong>解决方案</strong>:</p>

<pre><code class="language-python"># 1. 数据验证
def validate_data(data_path):
    """验证数据文件"""
    try:
        data = torch.load(data_path)
        
        # 检查数据类型
        if not isinstance(data, (torch.Tensor, dict, list)):
            raise ValueError(f"不支持的数据类型: {type(data)}")
        
        # 检查数据形状
        if isinstance(data, torch.Tensor):
            print(f"数据形状: {data.shape}")
            print(f"数据类型: {data.dtype}")
            print(f"数据范围: [{data.min():.4f}, {data.max():.4f}]")
            
            # 检查异常值
            if torch.isnan(data).any():
                print("警告: 数据包含NaN值")
            if torch.isinf(data).any():
                print("警告: 数据包含无穷值")
        
        return True
        
    except Exception as e:
        print(f"数据验证失败: {e}")
        return False

# 2. 数据预处理管道
class DataPreprocessor:
    def __init__(self, config):
        self.config = config
        self.scaler = None
    
    def fit_transform(self, data):
        """拟合并转换数据"""
        # 标准化
        if self.config.get('normalize', True):
            self.scaler = StandardScaler()
            data = self.scaler.fit_transform(data)
        
        # 异常值处理
        if self.config.get('clip_outliers', True):
            data = self.clip_outliers(data)
        
        # 填充缺失值
        if self.config.get('fill_missing', True):
            data = self.fill_missing_values(data)
        
        return torch.tensor(data, dtype=torch.float32)
    
    def clip_outliers(self, data, std_threshold=3):
        """裁剪异常值"""
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        
        lower_bound = mean - std_threshold * std
        upper_bound = mean + std_threshold * std
        
        return np.clip(data, lower_bound, upper_bound)
    
    def fill_missing_values(self, data, method='mean'):
        """填充缺失值"""
        if method == 'mean':
            mask = np.isnan(data)
            data[mask] = np.nanmean(data, axis=0)[mask[0]]
        elif method == 'forward_fill':
            data = pd.DataFrame(data).fillna(method='ffill').values
        
        return data
</code></pre>

<h2 id="调试工具">调试工具</h2>

<h3>🔧 调试工具集</h3>

<pre><code class="language-python"># 1. 模型调试器
class ModelDebugger:
    def __init__(self, model):
        self.model = model
        self.hooks = []
        self.activations = {}
        self.gradients = {}
    
    def register_hooks(self):
        """注册调试钩子"""
        def forward_hook(name):
            def hook(module, input, output):
                self.activations[name] = {
                    'input_shape': input[0].shape if input else None,
                    'output_shape': output.shape if hasattr(output, 'shape') else None,
                    'output_mean': output.mean().item() if hasattr(output, 'mean') else None,
                    'output_std': output.std().item() if hasattr(output, 'std') else None,
                }
            return hook
        
        def backward_hook(name):
            def hook(module, grad_input, grad_output):
                if grad_output[0] is not None:
                    self.gradients[name] = {
                        'grad_norm': grad_output[0].norm().item(),
                        'grad_mean': grad_output[0].mean().item(),
                        'grad_std': grad_output[0].std().item(),
                    }
            return hook
        
        for name, module in self.model.named_modules():
            if len(list(module.children())) == 0:  # 叶子模块
                self.hooks.append(module.register_forward_hook(forward_hook(name)))
                self.hooks.append(module.register_backward_hook(backward_hook(name)))
    
    def print_debug_info(self):
        """打印调试信息"""
        print("=== 激活统计 ===")
        for name, stats in self.activations.items():
            print(f"{name}: {stats}")
        
        print("\n=== 梯度统计 ===")
        for name, stats in self.gradients.items():
            print(f"{name}: {stats}")
    
    def remove_hooks(self):
        """移除钩子"""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

# 2. 训练监控器
class TrainingMonitor:
    def __init__(self, log_dir='./logs'):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        self.metrics = defaultdict(list)
        self.start_time = time.time()
    
    def log_metrics(self, epoch, **kwargs):
        """记录指标"""
        self.metrics['epoch'].append(epoch)
        self.metrics['timestamp'].append(time.time() - self.start_time)
        
        for key, value in kwargs.items():
            self.metrics[key].append(value)
    
    def plot_metrics(self, metrics_to_plot=None):
        """绘制指标曲线"""
        if metrics_to_plot is None:
            metrics_to_plot = ['train_loss', 'val_loss']
        
        fig, axes = plt.subplots(len(metrics_to_plot), 1, figsize=(10, 6 * len(metrics_to_plot)))
        if len(metrics_to_plot) == 1:
            axes = [axes]
        
        for i, metric in enumerate(metrics_to_plot):
            if metric in self.metrics:
                axes[i].plot(self.metrics['epoch'], self.metrics[metric])
                axes[i].set_title(f'{metric.title()}')
                axes[i].set_xlabel('Epoch')
                axes[i].set_ylabel(metric)
                axes[i].grid(True)
        
        plt.tight_layout()
        plt.savefig(self.log_dir / 'training_metrics.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_logs(self):
        """保存日志"""
        log_file = self.log_dir / 'training_log.json'
        with open(log_file, 'w') as f:
            json.dump(dict(self.metrics), f, indent=2)
</code></pre>

<h2 id="常见错误代码">常见错误代码</h2>

<h3>📋 错误代码对照表</h3>

<table>
<thead>
<tr>
<th>错误代码</th>
<th>错误描述</th>
<th>可能原因</th>
<th>解决方案</th>
</tr>
</thead>
<tbody>
<tr>
<td>VIV-001</td>
<td>配置文件解析失败</td>
<td>YAML格式错误</td>
<td>检查配置文件语法</td>
</tr>
<tr>
<td>VIV-002</td>
<td>模型初始化失败</td>
<td>参数不兼容</td>
<td>验证模型配置参数</td>
</tr>
<tr>
<td>VIV-003</td>
<td>数据加载失败</td>
<td>文件路径错误或格式不支持</td>
<td>检查数据文件路径和格式</td>
</tr>
<tr>
<td>VIV-004</td>
<td>CUDA内存不足</td>
<td>批次大小过大或模型过大</td>
<td>减小批次大小或使用梯度累积</td>
</tr>
<tr>
<td>VIV-005</td>
<td>梯度爆炸</td>
<td>学习率过大或模型不稳定</td>
<td>降低学习率或添加梯度裁剪</td>
</tr>
<tr>
<td>VIV-006</td>
<td>注意力计算错误</td>
<td>序列长度不匹配</td>
<td>检查输入序列维度</td>
</tr>
<tr>
<td>VIV-007</td>
<td>损失函数返回NaN</td>
<td>数值不稳定</td>
<td>检查数据预处理和模型输出</td>
</tr>
<tr>
<td>VIV-008</td>
<td>模型保存失败</td>
<td>磁盘空间不足或权限问题</td>
<td>检查存储空间和文件权限</td>
</tr>
</tbody>
</table>

<h3>🚨 紧急故障处理</h3>

<pre><code class="language-python"># 紧急恢复脚本
def emergency_recovery(checkpoint_dir, config_path):
    """紧急恢复训练"""
    try:
        # 1. 加载最新检查点
        checkpoints = list(Path(checkpoint_dir).glob('*.pth'))
        if not checkpoints:
            print("未找到检查点文件")
            return None
        
        latest_checkpoint = max(checkpoints, key=lambda x: x.stat().st_mtime)
        print(f"加载检查点: {latest_checkpoint}")
        
        # 2. 加载模型状态
        checkpoint = torch.load(latest_checkpoint, map_location='cpu')
        
        # 3. 重建模型
        config = load_config(config_path)
        model = VIVTransformer(config)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # 4. 验证模型
        model.eval()
        test_input = torch.randn(1, 100, config.input_dim)
        with torch.no_grad():
            output = model(test_input)
        
        print(f"模型恢复成功，输出形状: {output.shape}")
        return model, checkpoint
        
    except Exception as e:
        print(f"恢复失败: {e}")
        return None

# 使用示例
model, checkpoint = emergency_recovery('./checkpoints', './config/default_config.yaml')
if model is not None:
    print(f"从第 {checkpoint['epoch']} 轮恢复训练")
</code></pre>

<h2>📞 获取帮助</h2>

<p>如果以上解决方案无法解决您的问题，请通过以下方式获取帮助：</p>

<ol>
<li><strong>GitHub Issues</strong>: 在项目仓库提交详细的问题报告</li>
<li><strong>讨论区</strong>: 参与社区讨论，分享经验</li>
<li><strong>文档</strong>: 查阅完整的API文档和用户指南</li>
<li><strong>示例代码</strong>: 参考项目中的示例代码</li>
</ol>

<p>提交问题时，请包含以下信息：</p>
<ul>
<li>错误的完整堆栈跟踪</li>
<li>使用的配置文件</li>
<li>Python和依赖包版本</li>
<li>硬件环境信息</li>
<li>重现问题的最小代码示例</li>
</ul>
</div>

<div data-lang-en>
<h1 id="troubleshooting-guide">Troubleshooting Guide</h1>

<p>This document provides solutions and debugging techniques for common issues in the VIVTransformer project.</p>

<h2 id="table-of-contents">📋 Table of Contents</h2>

<ul>
<li><a href="#installation-issues">Installation Issues</a></li>
<li><a href="#configuration-issues">Configuration Issues</a></li>
<li><a href="#training-issues">Training Issues</a></li>
<li><a href="#inference-issues">Inference Issues</a></li>
<li><a href="#performance-issues">Performance Issues</a></li>
<li><a href="#memory-issues">Memory Issues</a></li>
<li><a href="#attention-mechanism-issues">Attention Mechanism Issues</a></li>
<li><a href="#data-processing-issues">Data Processing Issues</a></li>
<li><a href="#debugging-tools">Debugging Tools</a></li>
<li><a href="#common-error-codes">Common Error Codes</a></li>
</ul>

<h2 id="installation-issues">Installation Issues</h2>

<h3 id="dependency-installation-failed">❌ Dependency Installation Failed</h3>

<p><strong>Problem Description</strong>: Error occurs when running <code>pip install -r requirements.txt</code></p>

<p><strong>Common Causes</strong>:</p>
<ol>
<li>Incompatible Python version</li>
<li>CUDA version mismatch</li>
<li>Network connection issues</li>
<li>Insufficient permissions</li>
</ol>

<p><strong>Solutions</strong>:</p>

<pre><code class="language-bash"># 1. Check Python version (requires 3.8+)
python --version

# 2. Upgrade pip
python -m pip install --upgrade pip

# 3. Use mirror source
pip install -r requirements.txt -i https://pypi.org/simple/

# 4. Install key dependencies step by step
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers
pip install -r requirements.txt

# 5. Use conda environment (recommended)
conda create -n vivtransformer python=3.9
conda activate vivtransformer
pip install -r requirements.txt
</code></pre>

<h3 id="cuda-related-errors">❌ CUDA Related Errors</h3>

<p><strong>Problem Description</strong>: <code>RuntimeError: CUDA out of memory</code> or <code>CUDA device not found</code></p>

<p><strong>Solutions</strong>:</p>

<pre><code class="language-python"># Check CUDA availability
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"GPU count: {torch.cuda.device_count()}")

# Force CPU usage if CUDA unavailable
device = torch.device('cpu')
model = model.to(device)

# Clear GPU memory
torch.cuda.empty_cache()
</code></pre>

<p><strong>CUDA Version Compatibility</strong>:</p>
<pre><code class="language-bash"># Check system CUDA version
nvcc --version

# Install corresponding PyTorch version
# CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# CPU version
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
</code></pre>

<h2 id="configuration-issues">Configuration Issues</h2>

<h3 id="configuration-file-format-error">❌ Configuration File Format Error</h3>

<p><strong>Problem Description</strong>: <code>yaml.scanner.ScannerError</code> or configuration parsing failure</p>

<p><strong>Solutions</strong>:</p>

<pre><code class="language-python"># Validate YAML format
import yaml

def validate_config(config_path):
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        print("Configuration file format is correct")
        return config
    except yaml.YAMLError as e:
        print(f"YAML format error: {e}")
        return None
    except Exception as e:
        print(f"Configuration file error: {e}")
        return None

# Usage example
config = validate_config('config/default_config.yaml')
if config is None:
    print("Please check configuration file format")
</code></pre>

<p><strong>Common YAML Format Issues</strong>:</p>
<pre><code class="language-yaml"># ❌ Wrong: Inconsistent indentation
model:
  name: vivtransformer
   hidden_size: 512  # Indentation error

# ✅ Correct: Consistent 2-space indentation
model:
  name: vivtransformer
  hidden_size: 512

# ❌ Wrong: No space after colon
learning_rate:0.001

# ✅ Correct: Space after colon
learning_rate: 0.001

# ❌ Wrong: Special characters without quotes
data_path: C:\Users\data  # Windows path

# ✅ Correct: Use quotes or forward slashes
data_path: "C:\\Users\\data"
# or
data_path: C:/Users/data
</code></pre>

<h3 id="parameter-validation-failed">❌ Parameter Validation Failed</h3>

<p><strong>Problem Description</strong>: <code>ValueError: Invalid parameter value</code></p>

<p><strong>Solutions</strong>:</p>

<pre><code class="language-python"># Parameter validation tool
def validate_model_config(config):
    """Validate model configuration parameters"""
    errors = []
    
    # Check required parameters
    required_params = ['d_model', 'n_heads', 'n_layers']
    for param in required_params:
        if param not in config:
            errors.append(f"Missing required parameter: {param}")
    
    # Check parameter ranges
    if 'd_model' in config:
        if config['d_model'] <= 0 or config['d_model'] % config.get('n_heads', 8) != 0:
            errors.append("d_model must be positive and divisible by n_heads")
    
    if 'learning_rate' in config:
        if not (1e-6 <= config['learning_rate'] <= 1.0):
            errors.append("learning_rate must be between 1e-6 and 1.0")
    
    if errors:
        raise ValueError("\n".join(errors))
    
    return True

# Usage example
try:
    validate_model_config(config['model'])
    print("Configuration validation passed")
except ValueError as e:
    print(f"Configuration validation failed: {e}")
</code></pre>

<h2 id="training-issues">Training Issues</h2>

<h3 id="training-not-converging">❌ Training Not Converging</h3>

<p><strong>Problem Description</strong>: Loss function not decreasing or oscillating severely</p>

<p><strong>Possible Causes</strong>:</p>
<ol>
<li>Learning rate too large or too small</li>
<li>Inappropriate batch size</li>
<li>Gradient explosion or vanishing</li>
<li>Data preprocessing issues</li>
</ol>

<p><strong>Solutions</strong>:</p>

<pre><code class="language-python"># 1. Learning rate debugging
import matplotlib.pyplot as plt

def find_learning_rate(model, dataloader, init_lr=1e-8, final_lr=10):
    """Learning rate range test"""
    lrs = []
    losses = []
    
    optimizer = torch.optim.Adam(model.parameters(), lr=init_lr)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=1.1)
    
    for batch_idx, (data, target) in enumerate(dataloader):
        optimizer.zero_grad()
        output = model(data)
        loss = F.mse_loss(output, target)
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        
        lrs.append(optimizer.param_groups[0]['lr'])
        losses.append(loss.item())
        
        if optimizer.param_groups[0]['lr'] > final_lr:
            break
    
    # Plot learning rate vs loss curve
    plt.figure(figsize=(10, 6))
    plt.semilogx(lrs, losses)
    plt.xlabel('Learning Rate')
    plt.ylabel('Loss')
    plt.title('Learning Rate Range Test')
    plt.show()
    
    return lrs, losses

# 2. Gradient monitoring
def monitor_gradients(model):
    """Monitor gradient statistics"""
    total_norm = 0
    param_count = 0
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
            param_count += 1
            
            # Check gradient anomalies
            if torch.isnan(param.grad).any():
                print(f"Warning: {name} contains NaN gradients")
            if torch.isinf(param.grad).any():
                print(f"Warning: {name} contains infinite gradients")
    
    total_norm = total_norm ** (1. / 2)
    print(f"Gradient norm: {total_norm:.6f}")
    
    return total_norm

# 3. Loss function debugging
class DebugLoss(nn.Module):
    def __init__(self, base_loss):
        super().__init__()
        self.base_loss = base_loss
        self.loss_history = []
    
    def forward(self, pred, target):
        loss = self.base_loss(pred, target)
        
        # Record loss statistics
        self.loss_history.append(loss.item())
        
        # Check anomalies
        if torch.isnan(loss):
            print("Warning: Loss is NaN")
            print(f"Prediction stats: min={pred.min():.6f}, max={pred.max():.6f}, mean={pred.mean():.6f}")
            print(f"Target stats: min={target.min():.6f}, max={target.max():.6f}, mean={target.mean():.6f}")
        
        return loss
</code></pre>

<h3 id="overfitting-issues">❌ Overfitting Issues</h3>

<p><strong>Problem Description</strong>: Training loss decreases but validation loss increases</p>

<p><strong>Solutions</strong>:</p>

<pre><code class="language-python"># 1. Early stopping mechanism
class EarlyStopping:
    def __init__(self, patience=7, min_delta=0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None
    
    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1
        
        if self.counter >= self.patience:
            if self.restore_best_weights:
                model.load_state_dict(self.best_weights)
            return True
        return False
    
    def save_checkpoint(self, model):
        self.best_weights = model.state_dict().copy()

# 2. Regularization techniques
class RegularizedVIVTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Base model
        self.base_model = VIVTransformer(config)
        
        # Dropout layer
        self.dropout = nn.Dropout(config.get('dropout', 0.1))
        
        # Weight decay will be set in optimizer
    
    def forward(self, x):
        x = self.dropout(x)
        return self.base_model(x)
    
    def get_l1_loss(self):
        """Calculate L1 regularization loss"""
        l1_loss = 0
        for param in self.parameters():
            l1_loss += torch.sum(torch.abs(param))
        return l1_loss
    
    def get_l2_loss(self):
        """Calculate L2 regularization loss"""
        l2_loss = 0
        for param in self.parameters():
            l2_loss += torch.sum(param ** 2)
        return l2_loss

# 3. Data augmentation
def add_noise_augmentation(data, noise_factor=0.01):
    """Add noise augmentation"""
    noise = torch.randn_like(data) * noise_factor
    return data + noise

def temporal_masking(data, mask_ratio=0.1):
    """Temporal masking augmentation"""
    batch_size, seq_len, features = data.shape
    mask_len = int(seq_len * mask_ratio)
    
    for i in range(batch_size):
        start_idx = torch.randint(0, seq_len - mask_len, (1,))
        data[i, start_idx:start_idx + mask_len] = 0
    
    return data
</code></pre>

<h2 id="inference-issues">Inference Issues</h2>

<h3 id="slow-inference-speed">❌ Slow Inference Speed</h3>

<p><strong>Problem Description</strong>: Model inference takes too long</p>

<p><strong>Solutions</strong>:</p>

<pre><code class="language-python"># 1. Model optimization
import torch.jit

# TorchScript compilation
model.eval()
scripted_model = torch.jit.script(model)

# Or use trace
sample_input = torch.randn(1, 100, 512)
traced_model = torch.jit.trace(model, sample_input)

# 2. Inference acceleration
@torch.no_grad()
def fast_inference(model, data):
    """Fast inference"""
    model.eval()
    
    # Use half precision
    if torch.cuda.is_available():
        model = model.half()
        data = data.half()
    
    # Batch inference
    results = []
    batch_size = 32
    
    for i in range(0, len(data), batch_size):
        batch = data[i:i + batch_size]
        output = model(batch)
        results.append(output)
    
    return torch.cat(results, dim=0)

# 3. Memory optimization
def memory_efficient_inference(model, data, chunk_size=1000):
    """Memory efficient inference"""
    model.eval()
    results = []
    
    with torch.no_grad():
        for i in range(0, data.size(1), chunk_size):
            chunk = data[:, i:i + chunk_size]
            output = model(chunk)
            results.append(output.cpu())  # Move to CPU to free GPU memory
            torch.cuda.empty_cache()
    
    return torch.cat(results, dim=1)
</code></pre>

<h2 id="performance-issues">Performance Issues</h2>

<h3 id="slow-training-speed">❌ Slow Training Speed</h3>

<p><strong>Solutions</strong>:</p>

<pre><code class="language-python"># 1. Data loading optimization
from torch.utils.data import DataLoader

# Optimize DataLoader
dataloader = DataLoader(
    dataset,
    batch_size=32,
    num_workers=4,  # Multi-process loading
    pin_memory=True,  # Pin memory
    persistent_workers=True,  # Persistent workers
    prefetch_factor=2  # Prefetch factor
)

# 2. Mixed precision training
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

# 3. Gradient accumulation
accumulation_steps = 4

for i, (data, target) in enumerate(dataloader):
    with autocast():
        output = model(data)
        loss = criterion(output, target) / accumulation_steps
    
    scaler.scale(loss).backward()
    
    if (i + 1) % accumulation_steps == 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
</code></pre>

<h2 id="memory-issues">Memory Issues</h2>

<h3 id="gpu-memory-insufficient">❌ GPU Memory Insufficient</h3>

<p><strong>Solutions</strong>:</p>

<pre><code class="language-python"># 1. Memory monitoring
def monitor_gpu_memory():
    """Monitor GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        cached = torch.cuda.memory_reserved() / 1024**3
        print(f"GPU Memory - Allocated: {allocated:.2f}GB, Cached: {cached:.2f}GB")

# 2. Gradient checkpointing
from torch.utils.checkpoint import checkpoint

class CheckpointedTransformerLayer(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer
    
    def forward(self, x):
        return checkpoint(self.layer, x)

# 3. Model parallelism
class ModelParallelVIVTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        # Place different layers on different GPUs
        self.embedding = nn.Linear(config.input_dim, config.d_model).to('cuda:0')
        self.transformer_layers = nn.ModuleList([
            TransformerLayer(config).to(f'cuda:{i % torch.cuda.device_count()}')
            for i in range(config.n_layers)
        ])
        self.output_layer = nn.Linear(config.d_model, config.output_dim).to('cuda:0')
    
    def forward(self, x):
        x = x.to('cuda:0')
        x = self.embedding(x)
        
        for i, layer in enumerate(self.transformer_layers):
            device = f'cuda:{i % torch.cuda.device_count()}'
            x = x.to(device)
            x = layer(x)
        
        x = x.to('cuda:0')
        return self.output_layer(x)
</code></pre>

<h2 id="attention-mechanism-issues">Attention Mechanism Issues</h2>

<h3 id="abnormal-attention-weights">❌ Abnormal Attention Weights</h3>

<p><strong>Solutions</strong>:</p>

<pre><code class="language-python"># 1. Attention visualization
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_attention(attention_weights, save_path=None):
    """Visualize attention weights"""
    # attention_weights: [batch_size, n_heads, seq_len, seq_len]
    
    batch_size, n_heads, seq_len, _ = attention_weights.shape
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    for head in range(min(8, n_heads)):
        attention_map = attention_weights[0, head].detach().cpu().numpy()
        
        sns.heatmap(
            attention_map,
            ax=axes[head],
            cmap='Blues',
            cbar=True,
            square=True
        )
        axes[head].set_title(f'Head {head + 1}')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

# 2. Attention pattern analysis
def analyze_attention_patterns(attention_weights):
    """Analyze attention patterns"""
    # Calculate attention entropy
    entropy = -torch.sum(attention_weights * torch.log(attention_weights + 1e-8), dim=-1)
    
    # Calculate attention concentration
    max_attention = torch.max(attention_weights, dim=-1)[0]
    
    # Statistics
    stats = {
        'entropy_mean': entropy.mean().item(),
        'entropy_std': entropy.std().item(),
        'max_attention_mean': max_attention.mean().item(),
        'max_attention_std': max_attention.std().item(),
    }
    
    print("Attention Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value:.4f}")
    
    return stats

# 3. Attention regularization
class RegularizedAttention(nn.Module):
    def __init__(self, attention_module, entropy_weight=0.01):
        super().__init__()
        self.attention = attention_module
        self.entropy_weight = entropy_weight
    
    def forward(self, x):
        output, attention_weights = self.attention(x)
        
        # Calculate entropy regularization loss
        entropy = -torch.sum(attention_weights * torch.log(attention_weights + 1e-8), dim=-1)
        entropy_loss = -entropy.mean()  # Negative entropy, encourage diversity
        
        return output, attention_weights, entropy_loss * self.entropy_weight
</code></pre>

<h2 id="data-processing-issues">Data Processing Issues</h2>

<h3 id="data-loading-error">❌ Data Loading Error</h3>

<p><strong>Solutions</strong>:</p>

<pre><code class="language-python"># 1. Data validation
def validate_data(data_path):
    """Validate data file"""
    try:
        data = torch.load(data_path)
        
        # Check data type
        if not isinstance(data, (torch.Tensor, dict, list)):
            raise ValueError(f"Unsupported data type: {type(data)}")
        
        # Check data shape
        if isinstance(data, torch.Tensor):
            print(f"Data shape: {data.shape}")
            print(f"Data type: {data.dtype}")
            print(f"Data range: [{data.min():.4f}, {data.max():.4f}]")
            
            # Check anomalies
            if torch.isnan(data).any():
                print("Warning: Data contains NaN values")
            if torch.isinf(data).any():
                print("Warning: Data contains infinite values")
        
        return True
        
    except Exception as e:
        print(f"Data validation failed: {e}")
        return False

# 2. Data preprocessing pipeline
class DataPreprocessor:
    def __init__(self, config):
        self.config = config
        self.scaler = None
    
    def fit_transform(self, data):
        """Fit and transform data"""
        # Normalization
        if self.config.get('normalize', True):
            self.scaler = StandardScaler()
            data = self.scaler.fit_transform(data)
        
        # Outlier handling
        if self.config.get('clip_outliers', True):
            data = self.clip_outliers(data)
        
        # Fill missing values
        if self.config.get('fill_missing', True):
            data = self.fill_missing_values(data)
        
        return torch.tensor(data, dtype=torch.float32)
    
    def clip_outliers(self, data, std_threshold=3):
        """Clip outliers"""
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        
        lower_bound = mean - std_threshold * std
        upper_bound = mean + std_threshold * std
        
        return np.clip(data, lower_bound, upper_bound)
    
    def fill_missing_values(self, data, method='mean'):
        """Fill missing values"""
        if method == 'mean':
            mask = np.isnan(data)
            data[mask] = np.nanmean(data, axis=0)[mask[0]]
        elif method == 'forward_fill':
            data = pd.DataFrame(data).fillna(method='ffill').values
        
        return data
</code></pre>

<h2 id="debugging-tools">Debugging Tools</h2>

<h3>🔧 Debugging Toolkit</h3>

<pre><code class="language-python"># 1. Model debugger
class ModelDebugger:
    def __init__(self, model):
        self.model = model
        self.hooks = []
        self.activations = {}
        self.gradients = {}
    
    def register_hooks(self):
        """Register debugging hooks"""
        def forward_hook(name):
            def hook(module, input, output):
                self.activations[name] = {
                    'input_shape': input[0].shape if input else None,
                    'output_shape': output.shape if hasattr(output, 'shape') else None,
                    'output_mean': output.mean().item() if hasattr(output, 'mean') else None,
                    'output_std': output.std().item() if hasattr(output, 'std') else None,
                }
            return hook
        
        def backward_hook(name):
            def hook(module, grad_input, grad_output):
                if grad_output[0] is not None:
                    self.gradients[name] = {
                        'grad_norm': grad_output[0].norm().item(),
                        'grad_mean': grad_output[0].mean().item(),
                        'grad_std': grad_output[0].std().item(),
                    }
            return hook
        
        for name, module in self.model.named_modules():
            if len(list(module.children())) == 0:  # Leaf modules
                self.hooks.append(module.register_forward_hook(forward_hook(name)))
                self.hooks.append(module.register_backward_hook(backward_hook(name)))
    
    def print_debug_info(self):
        """Print debugging information"""
        print("=== Activation Statistics ===")
        for name, stats in self.activations.items():
            print(f"{name}: {stats}")
        
        print("\n=== Gradient Statistics ===")
        for name, stats in self.gradients.items():
            print(f"{name}: {stats}")
    
    def remove_hooks(self):
        """Remove hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

# 2. Training monitor
class TrainingMonitor:
    def __init__(self, log_dir='./logs'):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        self.metrics = defaultdict(list)
        self.start_time = time.time()
    
    def log_metrics(self, epoch, **kwargs):
        """Log metrics"""
        self.metrics['epoch'].append(epoch)
        self.metrics['timestamp'].append(time.time() - self.start_time)
        
        for key, value in kwargs.items():
            self.metrics[key].append(value)
    
    def plot_metrics(self, metrics_to_plot=None):
        """Plot metric curves"""
        if metrics_to_plot is None:
            metrics_to_plot = ['train_loss', 'val_loss']
        
        fig, axes = plt.subplots(len(metrics_to_plot), 1, figsize=(10, 6 * len(metrics_to_plot)))
        if len(metrics_to_plot) == 1:
            axes = [axes]
        
        for i, metric in enumerate(metrics_to_plot):
            if metric in self.metrics:
                axes[i].plot(self.metrics['epoch'], self.metrics[metric])
                axes[i].set_title(f'{metric.title()}')
                axes[i].set_xlabel('Epoch')
                axes[i].set_ylabel(metric)
                axes[i].grid(True)
        
        plt.tight_layout()
        plt.savefig(self.log_dir / 'training_metrics.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_logs(self):
        """Save logs"""
        log_file = self.log_dir / 'training_log.json'
        with open(log_file, 'w') as f:
            json.dump(dict(self.metrics), f, indent=2)
</code></pre>

<h2 id="common-error-codes">Common Error Codes</h2>

<h3>📋 Error Code Reference</h3>

<table>
<thead>
<tr>
<th>Error Code</th>
<th>Error Description</th>
<th>Possible Cause</th>
<th>Solution</th>
</tr>
</thead>
<tbody>
<tr>
<td>VIV-001</td>
<td>Configuration file parsing failed</td>
<td>YAML format error</td>
<td>Check configuration file syntax</td>
</tr>
<tr>
<td>VIV-002</td>
<td>Model initialization failed</td>
<td>Incompatible parameters</td>
<td>Validate model configuration parameters</td>
</tr>
<tr>
<td>VIV-003</td>
<td>Data loading failed</td>
<td>Wrong file path or unsupported format</td>
<td>Check data file path and format</td>
</tr>
<tr>
<td>VIV-004</td>
<td>CUDA out of memory</td>
<td>Batch size too large or model too big</td>
<td>Reduce batch size or use gradient accumulation</td>
</tr>
<tr>
<td>VIV-005</td>
<td>Gradient explosion</td>
<td>Learning rate too large or unstable model</td>
<td>Lower learning rate or add gradient clipping</td>
</tr>
<tr>
<td>VIV-006</td>
<td>Attention computation error</td>
<td>Sequence length mismatch</td>
<td>Check input sequence dimensions</td>
</tr>
<tr>
<td>VIV-007</td>
<td>Loss function returns NaN</td>
<td>Numerical instability</td>
<td>Check data preprocessing and model output</td>
</tr>
<tr>
<td>VIV-008</td>
<td>Model save failed</td>
<td>Insufficient disk space or permission issues</td>
<td>Check storage space and file permissions</td>
</tr>
</tbody>
</table>

<h3>🚨 Emergency Recovery</h3>

<pre><code class="language-python"># Emergency recovery script
def emergency_recovery(checkpoint_dir, config_path):
    """Emergency training recovery"""
    try:
        # 1. Load latest checkpoint
        checkpoints = list(Path(checkpoint_dir).glob('*.pth'))
        if not checkpoints:
            print("No checkpoint files found")
            return None
        
        latest_checkpoint = max(checkpoints, key=lambda x: x.stat().st_mtime)
        print(f"Loading checkpoint: {latest_checkpoint}")
        
        # 2. Load model state
        checkpoint = torch.load(latest_checkpoint, map_location='cpu')
        
        # 3. Rebuild model
        config = load_config(config_path)
        model = VIVTransformer(config)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # 4. Validate model
        model.eval()
        test_input = torch.randn(1, 100, config.input_dim)
        with torch.no_grad():
            output = model(test_input)
        
        print(f"Model recovery successful, output shape: {output.shape}")
        return model, checkpoint
        
    except Exception as e:
        print(f"Recovery failed: {e}")
        return None

# Usage example
model, checkpoint = emergency_recovery('./checkpoints', './config/default_config.yaml')
if model is not None:
    print(f"Resumed training from epoch {checkpoint['epoch']}")
</code></pre>

<h2>📞 Getting Help</h2>

<p>If the above solutions cannot resolve your issue, please get help through the following channels:</p>

<ol>
<li><strong>GitHub Issues</strong>: Submit detailed issue reports in the project repository</li>
<li><strong>Discussion Forum</strong>: Participate in community discussions and share experiences</li>
<li><strong>Documentation</strong>: Refer to complete API documentation and user guides</li>
<li><strong>Example Code</strong>: Check example code in the project</li>
</ol>

<p>When submitting issues, please include the following information:</p>
<ul>
<li>Complete error stack trace</li>
<li>Configuration file used</li>
<li>Python and dependency package versions</li>
<li>Hardware environment information</li>
<li>Minimal code example to reproduce the issue</li>
</ul>
</div>
