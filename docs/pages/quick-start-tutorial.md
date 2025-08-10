---
layout: doc
title: Quick Start Tutorial
nav_order: 2
parent: Getting Started
permalink: /pages/quick-start-tutorial/
---

<div data-lang-zh style="display: none;">
<h1>快速开始教程</h1>
<p class="fs-6 fw-300">本教程将指导您在5分钟内完成VIVTransformer项目的安装、配置和首次运行。</p>

<h2>📋 目录</h2>
<ul>
<li><a href="#环境准备">环境准备</a></li>
<li><a href="#项目安装">项目安装</a></li>
<li><a href="#基础配置">基础配置</a></li>
<li><a href="#首次运行">首次运行</a></li>
<li><a href="#结果查看">结果查看</a></li>
<li><a href="#常见问题">常见问题</a></li>
<li><a href="#下一步">下一步</a></li>
</ul>

<h2 id="环境准备">🚀 环境准备</h2>

<h3>🖥️ 系统要求</h3>
<table>
<tr><th>组件</th><th>最低要求</th><th>推荐配置</th></tr>
<tr><td><strong>操作系统</strong></td><td>Windows 10/Linux/macOS</td><td>Windows 11/Ubuntu 20.04+</td></tr>
<tr><td><strong>Python</strong></td><td>3.8+</td><td>3.9+</td></tr>
<tr><td><strong>内存</strong></td><td>8GB</td><td>16GB+</td></tr>
<tr><td><strong>GPU</strong></td><td>可选</td><td>NVIDIA RTX 3060+</td></tr>
<tr><td><strong>存储</strong></td><td>5GB</td><td>10GB+</td></tr>
</table>

<h3>🐍 Python环境检查</h3>
<pre><code class="language-bash"># 检查Python版本
python --version
# 应该显示 Python 3.8.x 或更高版本

# 检查pip版本
pip --version
</code></pre>

<h2 id="项目安装">📦 项目安装</h2>

<h3>方法1：从GitHub克隆</h3>
<pre><code class="language-bash"># 克隆项目
git clone https://github.com/Pitohuie-AIversion/Sparse_to_Dense_Transformer.git
cd Sparse_to_Dense_Transformer

# 安装依赖
pip install -r requirements.txt
</code></pre>

<h3>方法2：下载ZIP包</h3>
<ol>
<li>访问 <a href="https://github.com/Pitohuie-AIversion/Sparse_to_Dense_Transformer">GitHub仓库</a></li>
<li>点击 "Code" → "Download ZIP"</li>
<li>解压到本地目录</li>
<li>在项目目录中运行：<code>pip install -r requirements.txt</code></li>
</ol>

<h2 id="基础配置">⚙️ 基础配置</h2>

<h3>创建配置文件</h3>
<pre><code class="language-python"># config/quick_start.yaml
model:
  d_model: 512
  num_layers: 6
  num_heads: 8
  dropout: 0.1

training:
  batch_size: 32
  learning_rate: 0.001
  epochs: 10

data:
  train_path: "data/train.pt"
  val_path: "data/val.pt"
</code></pre>

<h2 id="首次运行">🎯 首次运行</h2>

<h3>准备数据</h3>
<pre><code class="language-bash"># 生成示例数据
python scripts/generate_sample_data.py
</code></pre>

<h3>运行单个实验</h3>
<pre><code class="language-bash"># 运行快速开始实验
python main.py --config config/quick_start.yaml --mode train
</code></pre>

<h2 id="结果查看">📊 结果查看</h2>

<h3>查看训练日志</h3>
<pre><code class="language-bash"># 查看最新的训练日志
tail -f logs/training.log
</code></pre>

<h3>查看结果文件</h3>
<pre><code class="language-bash"># 结果保存在 results/ 目录下
ls results/
</code></pre>

<h2 id="常见问题">❓ 常见问题</h2>

<h3>Q1: 安装依赖时出现错误</h3>
<p><strong>解决方案</strong>：</p>
<pre><code class="language-bash"># 升级pip
pip install --upgrade pip

# 使用国内镜像源
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/
</code></pre>

<h3>Q2: CUDA相关错误</h3>
<p><strong>解决方案</strong>：</p>
<pre><code class="language-bash"># 检查CUDA版本
nvcc --version

# 安装对应的PyTorch版本
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
</code></pre>

<h2 id="下一步">🎓 下一步</h2>

<p>恭喜！您已经成功运行了第一个VIVTransformer实验。接下来可以：</p>

<ul>
<li><a href="/pages/training-guide/">深入学习训练指南</a></li>
<li><a href="/pages/api-reference/">查看API参考文档</a></li>
<li><a href="/pages/examples/">探索更多示例</a></li>
<li><a href="/pages/faq/">查看常见问题</a></li>
</ul>
</div>

<div data-lang-en>
<h1>Quick Start Tutorial</h1>
<p class="fs-6 fw-300">This tutorial will guide you through installing, configuring, and running the VIVTransformer project for the first time in 5 minutes.</p>

<h2>📋 Table of Contents</h2>
<ul>
<li><a href="#environment-setup">Environment Setup</a></li>
<li><a href="#project-installation">Project Installation</a></li>
<li><a href="#basic-configuration">Basic Configuration</a></li>
<li><a href="#first-run">First Run</a></li>
<li><a href="#view-results">View Results</a></li>
<li><a href="#common-issues">Common Issues</a></li>
<li><a href="#next-steps">Next Steps</a></li>
</ul>

<h2 id="environment-setup">🚀 Environment Setup</h2>

<h3>🖥️ System Requirements</h3>
<table>
<tr><th>Component</th><th>Minimum</th><th>Recommended</th></tr>
<tr><td><strong>Operating System</strong></td><td>Windows 10/Linux/macOS</td><td>Windows 11/Ubuntu 20.04+</td></tr>
<tr><td><strong>Python</strong></td><td>3.8+</td><td>3.9+</td></tr>
<tr><td><strong>Memory</strong></td><td>8GB</td><td>16GB+</td></tr>
<tr><td><strong>GPU</strong></td><td>Optional</td><td>NVIDIA RTX 3060+</td></tr>
<tr><td><strong>Storage</strong></td><td>5GB</td><td>10GB+</td></tr>
</table>

<h3>🐍 Python Environment Check</h3>
<pre><code class="language-bash"># Check Python version
python --version
# Should display Python 3.8.x or higher

# Check pip version
pip --version
</code></pre>

<h2 id="project-installation">📦 Project Installation</h2>

<h3>Method 1: Clone from GitHub</h3>
<pre><code class="language-bash"># Clone the project
git clone https://github.com/Pitohuie-AIversion/Sparse_to_Dense_Transformer.git
cd Sparse_to_Dense_Transformer

# Install dependencies
pip install -r requirements.txt
</code></pre>

<h3>Method 2: Download ZIP</h3>
<ol>
<li>Visit <a href="https://github.com/Pitohuie-AIversion/Sparse_to_Dense_Transformer">GitHub Repository</a></li>
<li>Click "Code" → "Download ZIP"</li>
<li>Extract to local directory</li>
<li>Run in project directory: <code>pip install -r requirements.txt</code></li>
</ol>

<h2 id="basic-configuration">⚙️ Basic Configuration</h2>

<h3>Create Configuration File</h3>
<pre><code class="language-python"># config/quick_start.yaml
model:
  d_model: 512
  num_layers: 6
  num_heads: 8
  dropout: 0.1

training:
  batch_size: 32
  learning_rate: 0.001
  epochs: 10

data:
  train_path: "data/train.pt"
  val_path: "data/val.pt"
</code></pre>

<h2 id="first-run">🎯 First Run</h2>

<h3>Prepare Data</h3>
<pre><code class="language-bash"># Generate sample data
python scripts/generate_sample_data.py
</code></pre>

<h3>Run Single Experiment</h3>
<pre><code class="language-bash"># Run quick start experiment
python main.py --config config/quick_start.yaml --mode train
</code></pre>

<h2 id="view-results">📊 View Results</h2>

<h3>View Training Logs</h3>
<pre><code class="language-bash"># View latest training logs
tail -f logs/training.log
</code></pre>

<h3>View Result Files</h3>
<pre><code class="language-bash"># Results are saved in results/ directory
ls results/
</code></pre>

<h2 id="common-issues">❓ Common Issues</h2>

<h3>Q1: Error installing dependencies</h3>
<p><strong>Solution</strong>:</p>
<pre><code class="language-bash"># Upgrade pip
pip install --upgrade pip

# Use domestic mirror source
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/
</code></pre>

<h3>Q2: CUDA related errors</h3>
<p><strong>Solution</strong>:</p>
<pre><code class="language-bash"># Check CUDA version
nvcc --version

# Install corresponding PyTorch version
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
</code></pre>

<h2 id="next-steps">🎓 Next Steps</h2>

<p>Congratulations! You have successfully run your first VIVTransformer experiment. Next, you can:</p>

<ul>
<li><a href="/pages/training-guide/">Learn more about training guide</a></li>
<li><a href="/pages/api-reference/">Check API reference documentation</a></li>
<li><a href="/pages/examples/">Explore more examples</a></li>
<li><a href="/pages/faq/">View frequently asked questions</a></li>
</ul>
</div>
