---
layout: default
title: "Quick Start Tutorial"
nav_order: 2
has_children: false
description: "Complete VIVTransformer installation, configuration and first run in 5 minutes"
---

<h1>Quick Start Tutorial</h1>
<p class="fs-6 fw-300">This tutorial will guide you through the installation, configuration, and first run of the VIVTransformer project in 5 minutes.</p>

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
<tr><th>Component</th><th>Minimum Requirement</th><th>Recommended</th></tr>
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
<li>Visit the <a href="https://github.com/Pitohuie-AIversion/Sparse_to_Dense_Transformer">GitHub repository</a></li>
<li>Click "Code" → "Download ZIP"</li>
<li>Extract to local directory</li>
<li>Run in project directory: <code>pip install -r requirements.txt</code></li>
</ol>

<h2 id="basic-configuration">⚙️ Basic Configuration</h2>

<h3>Create Configuration File</h3>
<pre><code class="language-yaml"># config/quick_start.yaml
model:
  name: "VIVTransformer"
  attention_type: "multi_head"
  loss_function: "svd_enhanced"

training:
  batch_size: 32
  learning_rate: 0.001
  max_epochs: 10

data:
  input_size: [64, 64]
  sequence_length: 100
</code></pre>

<h2 id="first-run">🎯 First Run</h2>

<h3>Prepare Data</h3>
<pre><code class="language-bash"># Generate sample data
python scripts/generate_sample_data.py
</code></pre>

<h3>Run Single Experiment</h3>
<pre><code class="language-bash"># Run quick start experiment
python main.py --config config/quick_start.yaml
</code></pre>

<h2 id="view-results">📊 View Results</h2>

<h3>View Training Logs</h3>
<pre><code class="language-bash"># View latest training logs
cat logs/training_*.log
</code></pre>

<h3>View Result Files</h3>
<pre><code class="language-bash"># Results are saved in results/ directory
ls results/
</code></pre>

<h2 id="common-issues">❓ Common Issues</h2>

<h3>Q1: Dependency Installation Error</h3>
<p><strong>Solution</strong>:</p>
<pre><code class="language-bash"># Upgrade pip
pip install --upgrade pip

# Use domestic mirror (for China users)
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/
</code></pre>

<h3>Q2: CUDA Related Errors</h3>
<p><strong>Solution</strong>:</p>
<pre><code class="language-bash"># Check CUDA version
nvidia-smi

# Install corresponding PyTorch version
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
</code></pre>

<h2 id="next-steps">🎓 Next Steps</h2>

<p>Congratulations! You have successfully run your first VIVTransformer experiment. Next, you can:</p>
<ul>
<li><a href="{{ site.baseurl }}/pages/training-guide/">Learn training guide in depth</a></li>
<li><a href="{{ site.baseurl }}/pages/api-reference/">Check API reference documentation</a></li>
<li><a href="{{ site.baseurl }}/pages/examples/">Explore more examples</a></li>
<li><a href="{{ site.baseurl }}/pages/faq/">View frequently asked questions</a></li>
</ul>
