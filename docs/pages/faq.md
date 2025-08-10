---
layout: doc
title: FAQ
description: Frequently Asked Questions
nav_order: 17
parent: Getting Help
permalink: /pages/faq/
---

<div class="lang-content" data-lang-zh style="display: none;">
<h1 id="常见问题-faq">常见问题 (FAQ)</h1>

<p>本页面收集了VIVTransformer项目使用过程中的常见问题和解决方案。如果您的问题不在此列表中，请查看其他文档或提交Issue。</p>

<h2 id="目录">📋 目录</h2>

<ul>
<li><a href="#安装和环境问题">安装和环境问题</a></li>
<li><a href="#配置和运行问题">配置和运行问题</a></li>
<li><a href="#性能和资源问题">性能和资源问题</a></li>
<li><a href="#结果和分析问题">结果和分析问题</a></li>
<li><a href="#开发和扩展问题">开发和扩展问题</a></li>
<li><a href="#错误诊断">错误诊断</a></li>
</ul>

<h2 id="安装和环境问题">安装和环境问题</h2>

<h3 id="q1-安装依赖时出现版本冲突">❓ Q1: 安装依赖时出现版本冲突</h3>

<p><strong>问题描述</strong>：</p>
<pre><code class="language-bash">ERROR: pip's dependency resolver does not currently consider all the packages that are installed.
</code></pre>

<p><strong>解决方案</strong>：</p>
<pre><code class="language-bash"># 方案1：创建新的虚拟环境
conda create -n vivtransformer python=3.9
conda activate vivtransformer
pip install -r modify_multi_attention/requirements.txt

# 方案2：强制重新安装
pip install -r modify_multi_attention/requirements.txt --force-reinstall

# 方案3：逐个安装依赖
pip install torch numpy pyyaml matplotlib fightingcv_attention black flake8 tensorboard
</code></pre>

<h3 id="q2-cuda版本不匹配">❓ Q2: CUDA版本不匹配</h3>

<p><strong>问题描述</strong>：</p>
<pre><code>RuntimeError: CUDA runtime error: no kernel image is available for execution on the device
</code></pre>

<p><strong>解决方案</strong>：</p>
<pre><code class="language-bash"># 检查CUDA版本
nvcc --version
nvidia-smi

# 安装对应版本的PyTorch
# CUDA 11.6
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116

# CUDA 11.7
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu117

# CPU版本（如果没有GPU）
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu
</code></pre>

<h3 id="q3-fightingcv-attention安装失败">❓ Q3: fightingcv_attention安装失败</h3>

<p><strong>问题描述</strong>：</p>
<pre><code>ERROR: Could not find a version that satisfies the requirement fightingcv_attention
</code></pre>

<p><strong>解决方案</strong>：</p>
<pre><code class="language-bash"># 方案1：从GitHub安装
pip install git+https://github.com/xmu-xiaoma666/External-Attention-pytorch.git

# 方案2：手动下载安装
git clone https://github.com/xmu-xiaoma666/External-Attention-pytorch.git
cd External-Attention-pytorch
pip install -e .

# 方案3：使用替代源
pip install fightingcv_attention -i https://pypi.tuna.tsinghua.edu.cn/simple/
</code></pre>

<h3 id="q4-python版本过低">❓ Q4: Python版本过低</h3>

<p><strong>问题描述</strong>：</p>
<pre><code>SyntaxError: invalid syntax (使用了f-string等新特性)
</code></pre>

<p><strong>解决方案</strong>：</p>
<pre><code class="language-bash"># 检查Python版本
python --version

# 升级Python（推荐使用conda）
conda install python=3.9

# 或者使用pyenv
pyenv install 3.9.16
pyenv global 3.9.16
</code></pre>

<h2 id="配置和运行问题">配置和运行问题</h2>

<h3 id="q5-找不到配置文件">❓ Q5: 找不到配置文件</h3>

<p><strong>问题描述</strong>：</p>
<pre><code>FileNotFoundError: [Errno 2] No such file or directory: 'config.yaml'
</code></pre>

<p><strong>解决方案</strong>：</p>
<pre><code class="language-bash"># 检查配置文件路径
ls modify_multi_attention/configs/

# 使用绝对路径
python -m modify_multi_attention.main --config /absolute/path/to/config.yaml

# 从项目根目录运行
cd VIVTransformer
python -m modify_multi_attention.main
</code></pre>

<h3 id="q6-数据文件路径错误">❓ Q6: 数据文件路径错误</h3>

<p><strong>问题描述</strong>：</p>
<pre><code>FileNotFoundError: Data file not found at specified path
</code></pre>

<p><strong>解决方案</strong>：</p>
<pre><code class="language-yaml"># 修改config.yaml中的数据路径
data:
  path: "/correct/path/to/your/data.pt"  # 使用绝对路径
  # 或者
  path: "./relative/path/to/data.pt"     # 使用相对路径
</code></pre>

<pre><code class="language-bash"># 检查数据文件是否存在
ls -la /path/to/your/data.pt

# 创建软链接（如果数据在其他位置）
ln -s /actual/data/path/data.pt /config/data/path/data.pt
</code></pre>

<h3 id="q7-yaml配置文件语法错误">❓ Q7: YAML配置文件语法错误</h3>

<p><strong>问题描述</strong>：</p>
<pre><code>yaml.scanner.ScannerError: mapping values are not allowed here
</code></pre>

<p><strong>解决方案</strong>：</p>
<pre><code class="language-bash"># 验证YAML语法
python -c "import yaml; yaml.safe_load(open('modify_multi_attention/configs/config.yaml'))"

# 常见错误修复：
# 1. 缩进问题（使用空格，不要使用Tab）
# 2. 冒号后面要有空格
# 3. 字符串包含特殊字符时要加引号
</code></pre>

<p><strong>正确的YAML格式</strong>：</p>
<pre><code class="language-yaml">global:
  seed: 42                    # 冒号后有空格
  device: "cuda:0"            # 字符串加引号
data:
  batch_size: 128             # 正确缩进
  use_augmentation: true      # 布尔值
</code></pre>

<h3 id="q8-注意力机制名称错误">❓ Q8: 注意力机制名称错误</h3>

<p><strong>问题描述</strong>：</p>
<pre><code>KeyError: 'unknown_attention' not found in attention registry
</code></pre>

<p><strong>解决方案</strong>：</p>
<pre><code class="language-python"># 查看支持的注意力机制
python -c "
from modify_multi_attention.mymodels.model_factory import ModelFactory
print('支持的注意力机制:', ModelFactory.list_available_attentions())
"

# 或者查看配置文件中的完整列表
grep -A 50 "attention_test:" modify_multi_attention/configs/config.yaml
</code></pre>

<h2 id="性能和资源问题">性能和资源问题</h2>

<h3 id="q9-cuda内存不足-oom">❓ Q9: CUDA内存不足 (OOM)</h3>

<p><strong>问题描述</strong>：</p>
<pre><code>RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB
</code></pre>

<p><strong>解决方案</strong>：</p>
<pre><code class="language-yaml"># 方案1：减少批大小
data:
  batch_size: 32  # 从128减少到32

# 方案2：限制GPU内存使用
global:
  max_memory_fraction: 0.6  # 只使用60%的GPU内存

# 方案3：使用CPU模式
global:
  device: cpu
</code></pre>

<pre><code class="language-python"># 方案4：启用梯度累积
training:
  gradient_accumulation_steps: 4
  effective_batch_size: 128  # 实际批大小 = batch_size * accumulation_steps
</code></pre>

<h3 id="q10-训练速度太慢">❓ Q10: 训练速度太慢</h3>

<p><strong>问题描述</strong>：每个epoch需要很长时间完成</p>

<p><strong>解决方案</strong>：</p>
<pre><code class="language-yaml"># 方案1：使用高效注意力机制
model:
  attention_type: muse  # 或 eca, ufo

# 方案2：减少模型复杂度
model:
  d_model: 128      # 从256减少到128
  num_heads: 4      # 从8减少到4
  num_layers: 4     # 从6减少到4

# 方案3：增大批大小
data:
  batch_size: 256   # 如果内存允许

# 方案4：启用混合精度训练
training:
  use_amp: true
</code></pre>

<h3 id="q11-多gpu训练问题">❓ Q11: 多GPU训练问题</h3>

<p><strong>问题描述</strong>：</p>
<pre><code>RuntimeError: Expected all tensors to be on the same device
</code></pre>

<p><strong>解决方案</strong>：</p>
<pre><code class="language-yaml"># 启用数据并行
training:
  use_data_parallel: true
  gpu_ids: [0, 1, 2, 3]  # 指定使用的GPU
</code></pre>

<pre><code class="language-python"># 检查GPU可用性
import torch
print(f"可用GPU数量: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
</code></pre>

<h2 id="结果和分析问题">结果和分析问题</h2>

<h3 id="q12-结果文件找不到">❓ Q12: 结果文件找不到</h3>

<p><strong>问题描述</strong>：</p>
<pre><code>FileNotFoundError: Results file not found
</code></pre>

<p><strong>解决方案</strong>：</p>
<pre><code class="language-bash"># 检查输出目录
ls -la outputs/
ls -la results/

# 查看配置中的输出路径
grep -r "output" modify_multi_attention/configs/config.yaml

# 手动指定输出目录
python -m modify_multi_attention.main --output_dir ./my_results
</code></pre>

<h3 id="q13-可视化图表不显示">❓ Q13: 可视化图表不显示</h3>

<p><strong>问题描述</strong>：运行完成但没有生成图表</p>

<p><strong>解决方案</strong>：</p>
<pre><code class="language-yaml"># 启用可视化
visualization:
  enabled: true
  save_plots: true
  show_plots: false  # 服务器环境设为false
</code></pre>

<pre><code class="language-python"># 检查matplotlib后端
import matplotlib
print(f"当前后端: {matplotlib.get_backend()}")

# 设置非交互式后端
import matplotlib
matplotlib.use('Agg')
</code></pre>

<h3 id="q14-注意力权重分析异常">❓ Q14: 注意力权重分析异常</h3>

<p><strong>问题描述</strong>：注意力权重全为0或异常值</p>

<p><strong>解决方案</strong>：</p>
<pre><code class="language-python"># 检查注意力权重
def debug_attention_weights(model, data):
    model.eval()
    with torch.no_grad():
        # 获取注意力权重
        output, attention_weights = model(data, return_attention=True)
        
        print(f"注意力权重形状: {attention_weights.shape}")
        print(f"权重范围: [{attention_weights.min():.6f}, {attention_weights.max():.6f}]")
        print(f"权重和: {attention_weights.sum(dim=-1).mean():.6f}")
        
        # 检查是否有异常值
        if torch.isnan(attention_weights).any():
            print("警告: 注意力权重包含NaN")
        if torch.isinf(attention_weights).any():
            print("警告: 注意力权重包含无穷值")
</code></pre>

<h2 id="开发和扩展问题">开发和扩展问题</h2>

<h3 id="q15-添加新的注意力机制">❓ Q15: 添加新的注意力机制</h3>

<p><strong>问题描述</strong>：如何集成自定义的注意力机制</p>

<p><strong>解决方案</strong>：</p>
<pre><code class="language-python"># 1. 创建新的注意力类
class MyCustomAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        # 实现您的注意力机制
    
    def forward(self, x):
        # 实现前向传播
        return x

# 2. 注册到工厂类
from modify_multi_attention.mymodels.model_factory import ModelFactory

@ModelFactory.register_attention('my_custom')
class MyCustomAttentionWrapper:
    @staticmethod
    def create(d_model, num_heads, **kwargs):
        return MyCustomAttention(d_model, num_heads)
</code></pre>

<h3 id="q16-修改模型架构">❓ Q16: 修改模型架构</h3>

<p><strong>问题描述</strong>：如何修改Transformer的层数或维度</p>

<p><strong>解决方案</strong>：</p>
<pre><code class="language-yaml"># 在配置文件中修改
model:
  d_model: 512        # 模型维度
  num_heads: 8        # 注意力头数
  num_layers: 6       # Transformer层数
  d_ff: 2048         # 前馈网络维度
  dropout: 0.1       # Dropout率
</code></pre>

<pre><code class="language-python"># 或者在代码中直接修改
from modify_multi_attention.mymodels.vivtransformer import VIVTransformer

model = VIVTransformer(
    d_model=512,
    num_heads=8,
    num_layers=6,
    attention_type='muse'
)
</code></pre>

<h3 id="q17-自定义数据加载">❓ Q17: 自定义数据加载</h3>

<p><strong>问题描述</strong>：如何使用自己的数据集</p>

<p><strong>解决方案</strong>：</p>
<pre><code class="language-python"># 1. 创建自定义数据集类
from torch.utils.data import Dataset

class MyCustomDataset(Dataset):
    def __init__(self, data_path):
        # 加载您的数据
        self.data = self.load_data(data_path)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # 返回单个样本
        return self.data[idx]
    
    def load_data(self, path):
        # 实现数据加载逻辑
        pass

# 2. 在配置中指定数据集
data:
  dataset_class: "MyCustomDataset"
  dataset_args:
    data_path: "/path/to/your/data"
</code></pre>

<h2 id="错误诊断">错误诊断</h2>

<h3 id="q18-启用调试模式">❓ Q18: 启用调试模式</h3>

<p><strong>解决方案</strong>：</p>
<pre><code class="language-yaml"># 在配置文件中启用调试
global:
  debug: true
  log_level: DEBUG
  
logging:
  console_level: DEBUG
  file_level: DEBUG
  save_logs: true
</code></pre>

<pre><code class="language-bash"># 或者使用命令行参数
python -m modify_multi_attention.main --debug --verbose
</code></pre>

<h3 id="q19-性能分析">❓ Q19: 性能分析</h3>

<p><strong>解决方案</strong>：</p>
<pre><code class="language-python"># 使用PyTorch Profiler
import torch.profiler

with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
    schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
    on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/profiler'),
    record_shapes=True,
    profile_memory=True,
    with_stack=True
) as prof:
    for step, batch in enumerate(dataloader):
        # 训练步骤
        output = model(batch)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        prof.step()
</code></pre>

<h3 id="q20-内存泄漏检测">❓ Q20: 内存泄漏检测</h3>

<p><strong>解决方案</strong>：</p>
<pre><code class="language-python"># 监控GPU内存使用
import torch

def monitor_memory():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        cached = torch.cuda.memory_reserved() / 1024**3
        print(f"GPU内存 - 已分配: {allocated:.2f}GB, 已缓存: {cached:.2f}GB")

# 在训练循环中定期调用
for epoch in range(num_epochs):
    for batch in dataloader:
        # 训练代码
        pass
    
    # 每个epoch后检查内存
    monitor_memory()
    torch.cuda.empty_cache()  # 清理缓存
</code></pre>

<h2>💡 最佳实践建议</h2>

<h3>🔧 开发环境设置</h3>

<ol>
<li><strong>使用虚拟环境</strong>：始终在独立的conda或venv环境中工作</li>
<li><strong>版本固定</strong>：在requirements.txt中固定依赖版本</li>
<li><strong>代码格式化</strong>：使用black和flake8保持代码风格一致</li>
<li><strong>版本控制</strong>：使用git跟踪代码变更</li>
</ol>

<h3>🚀 性能优化</h3>

<ol>
<li><strong>批大小调优</strong>：根据GPU内存调整批大小</li>
<li><strong>混合精度</strong>：在支持的硬件上启用AMP</li>
<li><strong>数据预处理</strong>：使用多进程数据加载</li>
<li><strong>模型选择</strong>：根据任务选择合适的注意力机制</li>
</ol>

<h3>🐛 调试技巧</h3>

<ol>
<li><strong>逐步验证</strong>：从简单配置开始，逐步增加复杂度</li>
<li><strong>日志记录</strong>：启用详细日志记录</li>
<li><strong>可视化</strong>：使用TensorBoard监控训练过程</li>
<li><strong>单元测试</strong>：为关键组件编写测试</li>
</ol>

<h2>📞 获取更多帮助</h2>

<p>如果您的问题仍未解决，请通过以下方式获取帮助：</p>

<ul>
<li><strong>GitHub Issues</strong>: <a href="https://github.com/your-repo/issues">提交详细的问题报告</a></li>
<li><strong>讨论区</strong>: <a href="https://github.com/your-repo/discussions">参与社区讨论</a></li>
<li><strong>文档</strong>: <a href="/">查阅完整文档</a></li>
<li><strong>示例代码</strong>: <a href="{{ site.baseurl }}/pages/examples/">参考示例代码</a></li>
</ul>

<p>提交问题时，请包含：</p>
<ul>
<li>完整的错误信息和堆栈跟踪</li>
<li>使用的配置文件</li>
<li>Python和依赖包版本信息</li>
<li>硬件环境（GPU型号、内存等）</li>
<li>重现问题的最小代码示例</li>
</ul>
</div>

<div class="lang-content" data-lang-en>
<h1 id="faq">Frequently Asked Questions (FAQ)</h1>

<p>This page collects common questions and solutions encountered when using the VIVTransformer project. If your question is not listed here, please check other documentation or submit an Issue.</p>

<h2 id="table-of-contents">📋 Table of Contents</h2>

<ul>
<li><a href="#installation-and-environment-issues">Installation and Environment Issues</a></li>
<li><a href="#configuration-and-runtime-issues">Configuration and Runtime Issues</a></li>
<li><a href="#performance-and-resource-issues">Performance and Resource Issues</a></li>
<li><a href="#results-and-analysis-issues">Results and Analysis Issues</a></li>
<li><a href="#development-and-extension-issues">Development and Extension Issues</a></li>
<li><a href="#error-diagnosis">Error Diagnosis</a></li>
</ul>

<h2 id="installation-and-environment-issues">Installation and Environment Issues</h2>

<h3 id="q1-version-conflicts-when-installing-dependencies">❓ Q1: Version conflicts when installing dependencies</h3>

<p><strong>Problem Description</strong>:</p>
<pre><code class="language-bash">ERROR: pip's dependency resolver does not currently consider all the packages that are installed.
</code></pre>

<p><strong>Solutions</strong>:</p>
<pre><code class="language-bash"># Solution 1: Create a new virtual environment
conda create -n vivtransformer python=3.9
conda activate vivtransformer
pip install -r modify_multi_attention/requirements.txt

# Solution 2: Force reinstall
pip install -r modify_multi_attention/requirements.txt --force-reinstall

# Solution 3: Install dependencies individually
pip install torch numpy pyyaml matplotlib fightingcv_attention black flake8 tensorboard
</code></pre>

<h3 id="q2-cuda-version-mismatch">❓ Q2: CUDA version mismatch</h3>

<p><strong>Problem Description</strong>:</p>
<pre><code>RuntimeError: CUDA runtime error: no kernel image is available for execution on the device
</code></pre>

<p><strong>Solutions</strong>:</p>
<pre><code class="language-bash"># Check CUDA version
nvcc --version
nvidia-smi

# Install corresponding PyTorch version
# CUDA 11.6
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116

# CUDA 11.7
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu117

# CPU version (if no GPU)
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu
</code></pre>

<h3 id="q3-fightingcv-attention-installation-failed">❓ Q3: fightingcv_attention installation failed</h3>

<p><strong>Problem Description</strong>:</p>
<pre><code>ERROR: Could not find a version that satisfies the requirement fightingcv_attention
</code></pre>

<p><strong>Solutions</strong>:</p>
<pre><code class="language-bash"># Solution 1: Install from GitHub
pip install git+https://github.com/xmu-xiaoma666/External-Attention-pytorch.git

# Solution 2: Manual download and install
git clone https://github.com/xmu-xiaoma666/External-Attention-pytorch.git
cd External-Attention-pytorch
pip install -e .

# Solution 3: Use alternative source
pip install fightingcv_attention -i https://pypi.tuna.tsinghua.edu.cn/simple/
</code></pre>

<h3 id="q4-python-version-too-low">❓ Q4: Python version too low</h3>

<p><strong>Problem Description</strong>:</p>
<pre><code>SyntaxError: invalid syntax (using f-string and other new features)
</code></pre>

<p><strong>Solutions</strong>:</p>
<pre><code class="language-bash"># Check Python version
python --version

# Upgrade Python (recommended using conda)
conda install python=3.9

# Or use pyenv
pyenv install 3.9.16
pyenv global 3.9.16
</code></pre>

<h2 id="configuration-and-runtime-issues">Configuration and Runtime Issues</h2>

<h3 id="q5-configuration-file-not-found">❓ Q5: Configuration file not found</h3>

<p><strong>Problem Description</strong>:</p>
<pre><code>FileNotFoundError: [Errno 2] No such file or directory: 'config.yaml'
</code></pre>

<p><strong>Solutions</strong>:</p>
<pre><code class="language-bash"># Check configuration file path
ls modify_multi_attention/configs/

# Use absolute path
python -m modify_multi_attention.main --config /absolute/path/to/config.yaml

# Run from project root directory
cd VIVTransformer
python -m modify_multi_attention.main
</code></pre>

<h3 id="q6-data-file-path-error">❓ Q6: Data file path error</h3>

<p><strong>Problem Description</strong>:</p>
<pre><code>FileNotFoundError: Data file not found at specified path
</code></pre>

<p><strong>Solutions</strong>:</p>
<pre><code class="language-yaml"># Modify data path in config.yaml
data:
  path: "/correct/path/to/your/data.pt"  # Use absolute path
  # or
  path: "./relative/path/to/data.pt"     # Use relative path
</code></pre>

<pre><code class="language-bash"># Check if data file exists
ls -la /path/to/your/data.pt

# Create symbolic link (if data is in another location)
ln -s /actual/data/path/data.pt /config/data/path/data.pt
</code></pre>

<h3 id="q7-yaml-configuration-syntax-error">❓ Q7: YAML configuration syntax error</h3>

<p><strong>Problem Description</strong>:</p>
<pre><code>yaml.scanner.ScannerError: mapping values are not allowed here
</code></pre>

<p><strong>Solutions</strong>:</p>
<pre><code class="language-bash"># Validate YAML syntax
python -c "import yaml; yaml.safe_load(open('modify_multi_attention/configs/config.yaml'))"

# Common error fixes:
# 1. Indentation issues (use spaces, not tabs)
# 2. Space required after colon
# 3. Quote strings with special characters
</code></pre>

<p><strong>Correct YAML format</strong>:</p>
<pre><code class="language-yaml">global:
  seed: 42                    # Space after colon
  device: "cuda:0"            # Quote strings
data:
  batch_size: 128             # Correct indentation
  use_augmentation: true      # Boolean value
</code></pre>

<h3 id="q8-attention-mechanism-name-error">❓ Q8: Attention mechanism name error</h3>

<p><strong>Problem Description</strong>:</p>
<pre><code>KeyError: 'unknown_attention' not found in attention registry
</code></pre>

<p><strong>Solutions</strong>:</p>
<pre><code class="language-python"># Check supported attention mechanisms
python -c "
from modify_multi_attention.mymodels.model_factory import ModelFactory
print('Supported attention mechanisms:', ModelFactory.list_available_attentions())
"

# Or check complete list in config file
grep -A 50 "attention_test:" modify_multi_attention/configs/config.yaml
</code></pre>

<h2 id="performance-and-resource-issues">Performance and Resource Issues</h2>

<h3 id="q9-cuda-out-of-memory-oom">❓ Q9: CUDA out of memory (OOM)</h3>

<p><strong>Problem Description</strong>:</p>
<pre><code>RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB
</code></pre>

<p><strong>Solutions</strong>:</p>
<pre><code class="language-yaml"># Solution 1: Reduce batch size
data:
  batch_size: 32  # Reduce from 128 to 32

# Solution 2: Limit GPU memory usage
global:
  max_memory_fraction: 0.6  # Use only 60% of GPU memory

# Solution 3: Use CPU mode
global:
  device: cpu
</code></pre>

<pre><code class="language-python"># Solution 4: Enable gradient accumulation
training:
  gradient_accumulation_steps: 4
  effective_batch_size: 128  # Actual batch size = batch_size * accumulation_steps
</code></pre>

<h3 id="q10-training-speed-too-slow">❓ Q10: Training speed too slow</h3>

<p><strong>Problem Description</strong>: Each epoch takes too long to complete</p>

<p><strong>Solutions</strong>:</p>
<pre><code class="language-yaml"># Solution 1: Use efficient attention mechanisms
model:
  attention_type: muse  # or eca, ufo

# Solution 2: Reduce model complexity
model:
  d_model: 128      # Reduce from 256 to 128
  num_heads: 4      # Reduce from 8 to 4
  num_layers: 4     # Reduce from 6 to 4

# Solution 3: Increase batch size
data:
  batch_size: 256   # If memory allows

# Solution 4: Enable mixed precision training
training:
  use_amp: true
</code></pre>

<h3 id="q11-multi-gpu-training-issues">❓ Q11: Multi-GPU training issues</h3>

<p><strong>Problem Description</strong>:</p>
<pre><code>RuntimeError: Expected all tensors to be on the same device
</code></pre>

<p><strong>Solutions</strong>:</p>
<pre><code class="language-yaml"># Enable data parallelism
training:
  use_data_parallel: true
  gpu_ids: [0, 1, 2, 3]  # Specify GPUs to use
</code></pre>

<pre><code class="language-python"># Check GPU availability
import torch
print(f"Available GPUs: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
</code></pre>

<h2 id="results-and-analysis-issues">Results and Analysis Issues</h2>

<h3 id="q12-results-file-not-found">❓ Q12: Results file not found</h3>

<p><strong>Problem Description</strong>:</p>
<pre><code>FileNotFoundError: Results file not found
</code></pre>

<p><strong>Solutions</strong>:</p>
<pre><code class="language-bash"># Check output directories
ls -la outputs/
ls -la results/

# Check output path in configuration
grep -r "output" modify_multi_attention/configs/config.yaml

# Manually specify output directory
python -m modify_multi_attention.main --output_dir ./my_results
</code></pre>

<h3 id="q13-visualization-charts-not-displayed">❓ Q13: Visualization charts not displayed</h3>

<p><strong>Problem Description</strong>: Execution completes but no charts are generated</p>

<p><strong>Solutions</strong>:</p>
<pre><code class="language-yaml"># Enable visualization
visualization:
  enabled: true
  save_plots: true
  show_plots: false  # Set to false in server environment
</code></pre>

<pre><code class="language-python"># Check matplotlib backend
import matplotlib
print(f"Current backend: {matplotlib.get_backend()}")

# Set non-interactive backend
import matplotlib
matplotlib.use('Agg')
</code></pre>

<h3 id="q14-attention-weight-analysis-anomaly">❓ Q14: Attention weight analysis anomaly</h3>

<p><strong>Problem Description</strong>: Attention weights are all zeros or abnormal values</p>

<p><strong>Solutions</strong>:</p>
<pre><code class="language-python"># Check attention weights
def debug_attention_weights(model, data):
    model.eval()
    with torch.no_grad():
        # Get attention weights
        output, attention_weights = model(data, return_attention=True)
        
        print(f"Attention weight shape: {attention_weights.shape}")
        print(f"Weight range: [{attention_weights.min():.6f}, {attention_weights.max():.6f}]")
        print(f"Weight sum: {attention_weights.sum(dim=-1).mean():.6f}")
        
        # Check for anomalies
        if torch.isnan(attention_weights).any():
            print("Warning: Attention weights contain NaN")
        if torch.isinf(attention_weights).any():
            print("Warning: Attention weights contain infinite values")
</code></pre>

<h2 id="development-and-extension-issues">Development and Extension Issues</h2>

<h3 id="q15-adding-new-attention-mechanisms">❓ Q15: Adding new attention mechanisms</h3>

<p><strong>Problem Description</strong>: How to integrate custom attention mechanisms</p>

<p><strong>Solutions</strong>:</p>
<pre><code class="language-python"># 1. Create new attention class
class MyCustomAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        # Implement your attention mechanism
    
    def forward(self, x):
        # Implement forward pass
        return x

# 2. Register to factory class
from modify_multi_attention.mymodels.model_factory import ModelFactory

@ModelFactory.register_attention('my_custom')
class MyCustomAttentionWrapper:
    @staticmethod
    def create(d_model, num_heads, **kwargs):
        return MyCustomAttention(d_model, num_heads)
</code></pre>

<h3 id="q16-modifying-model-architecture">❓ Q16: Modifying model architecture</h3>

<p><strong>Problem Description</strong>: How to modify the number of Transformer layers or dimensions</p>

<p><strong>Solutions</strong>:</p>
<pre><code class="language-yaml"># Modify in configuration file
model:
  d_model: 512        # Model dimension
  num_heads: 8        # Number of attention heads
  num_layers: 6       # Number of Transformer layers
  d_ff: 2048         # Feed-forward network dimension
  dropout: 0.1       # Dropout rate
</code></pre>

<pre><code class="language-python"># Or modify directly in code
from modify_multi_attention.mymodels.vivtransformer import VIVTransformer

model = VIVTransformer(
    d_model=512,
    num_heads=8,
    num_layers=6,
    attention_type='muse'
)
</code></pre>

<h3 id="q17-custom-data-loading">❓ Q17: Custom data loading</h3>

<p><strong>Problem Description</strong>: How to use your own dataset</p>

<p><strong>Solutions</strong>:</p>
<pre><code class="language-python"># 1. Create custom dataset class
from torch.utils.data import Dataset

class MyCustomDataset(Dataset):
    def __init__(self, data_path):
        # Load your data
        self.data = self.load_data(data_path)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Return single sample
        return self.data[idx]
    
    def load_data(self, path):
        # Implement data loading logic
        pass

# 2. Specify dataset in configuration
data:
  dataset_class: "MyCustomDataset"
  dataset_args:
    data_path: "/path/to/your/data"
</code></pre>

<h2 id="error-diagnosis">Error Diagnosis</h2>

<h3 id="q18-enable-debug-mode">❓ Q18: Enable debug mode</h3>

<p><strong>Solutions</strong>:</p>
<pre><code class="language-yaml"># Enable debugging in configuration file
global:
  debug: true
  log_level: DEBUG
  
logging:
  console_level: DEBUG
  file_level: DEBUG
  save_logs: true
</code></pre>

<pre><code class="language-bash"># Or use command line arguments
python -m modify_multi_attention.main --debug --verbose
</code></pre>

<h3 id="q19-performance-profiling">❓ Q19: Performance profiling</h3>

<p><strong>Solutions</strong>:</p>
<pre><code class="language-python"># Use PyTorch Profiler
import torch.profiler

with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
    schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
    on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/profiler'),
    record_shapes=True,
    profile_memory=True,
    with_stack=True
) as prof:
    for step, batch in enumerate(dataloader):
        # Training step
        output = model(batch)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        prof.step()
</code></pre>

<h3 id="q20-memory-leak-detection">❓ Q20: Memory leak detection</h3>

<p><strong>Solutions</strong>:</p>
<pre><code class="language-python"># Monitor GPU memory usage
import torch

def monitor_memory():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        cached = torch.cuda.memory_reserved() / 1024**3
        print(f"GPU Memory - Allocated: {allocated:.2f}GB, Cached: {cached:.2f}GB")

# Call regularly in training loop
for epoch in range(num_epochs):
    for batch in dataloader:
        # Training code
        pass
    
    # Check memory after each epoch
    monitor_memory()
    torch.cuda.empty_cache()  # Clear cache
</code></pre>

<h2>💡 Best Practice Recommendations</h2>

<h3>🔧 Development Environment Setup</h3>

<ol>
<li><strong>Use Virtual Environments</strong>: Always work in isolated conda or venv environments</li>
<li><strong>Version Pinning</strong>: Pin dependency versions in requirements.txt</li>
<li><strong>Code Formatting</strong>: Use black and flake8 to maintain consistent code style</li>
<li><strong>Version Control</strong>: Use git to track code changes</li>
</ol>

<h3>🚀 Performance Optimization</h3>

<ol>
<li><strong>Batch Size Tuning</strong>: Adjust batch size according to GPU memory</li>
<li><strong>Mixed Precision</strong>: Enable AMP on supported hardware</li>
<li><strong>Data Preprocessing</strong>: Use multi-process data loading</li>
<li><strong>Model Selection</strong>: Choose appropriate attention mechanisms for your task</li>
</ol>

<h3>🐛 Debugging Tips</h3>

<ol>
<li><strong>Incremental Validation</strong>: Start with simple configurations, gradually increase complexity</li>
<li><strong>Logging</strong>: Enable detailed logging</li>
<li><strong>Visualization</strong>: Use TensorBoard to monitor training process</li>
<li><strong>Unit Testing</strong>: Write tests for critical components</li>
</ol>

<h2>📞 Getting More Help</h2>

<p>If your issue is still unresolved, please get help through the following channels:</p>

<ul>
<li><strong>GitHub Issues</strong>: <a href="https://github.com/your-repo/issues">Submit detailed issue reports</a></li>
<li><strong>Discussion Forum</strong>: <a href="https://github.com/your-repo/discussions">Participate in community discussions</a></li>
<li><strong>Documentation</strong>: <a href="/">Refer to complete documentation</a></li>
<li><strong>Example Code</strong>: <a href="{{ site.baseurl }}/pages/examples/">Check example code</a></li>
</ul>

<p>When submitting issues, please include:</p>
<ul>
<li>Complete error messages and stack traces</li>
<li>Configuration files used</li>
<li>Python and dependency package version information</li>
<li>Hardware environment (GPU model, memory, etc.)</li>
<li>Minimal code example to reproduce the issue</li>
</ul>
</div>
