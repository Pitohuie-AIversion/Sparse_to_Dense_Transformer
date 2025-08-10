---
layout: doc
title: 交互式演示
nav_order: 5
permalink: /pages/interactive-demo/
description: "VIVTransformer交互式演示和可视化展示"
---

# 交互式演示 🎮

体验VIVTransformer的强大功能，通过交互式界面探索涡激振动分析的魅力。

## 📋 目录

- [在线演示平台](#在线演示平台)
- [可视化展示](#可视化展示)
- [实时监控界面](#实时监控界面)
- [参数调节工具](#参数调节工具)
- [对比分析工具](#对比分析工具)
- [教学演示模式](#教学演示模式)

## 🌐 在线演示平台

### 🚀 快速体验

<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 20px; border-radius: 10px; margin: 20px 0; color: white;">
  <h3>🎯 一键体验 VIVTransformer</h3>
  <p>无需安装，直接在浏览器中体验完整功能</p>
  <div style="display: flex; gap: 10px; margin-top: 15px;">
    <button style="background: #4CAF50; color: white; border: none; padding: 10px 20px; border-radius: 5px; cursor: pointer;">🚀 启动演示</button>
    <button style="background: #2196F3; color: white; border: none; padding: 10px 20px; border-radius: 5px; cursor: pointer;">📊 查看示例</button>
    <button style="background: #FF9800; color: white; border: none; padding: 10px 20px; border-radius: 5px; cursor: pointer;">📚 教程模式</button>
  </div>
</div>

### 🎮 演示功能模块

#### 1. 🌊 流场可视化模块

```html
<!-- 流场可视化界面 -->
<div class="flow-visualization-panel">
  <div class="control-panel">
    <h4>🎛️ 控制面板</h4>
    
    <!-- 雷诺数控制 -->
    <div class="parameter-control">
      <label>雷诺数 (Re):</label>
      <input type="range" id="reynolds" min="1000" max="10000" value="4000" 
             oninput="updateReynolds(this.value)">
      <span id="reynolds-value">4000</span>
    </div>
    
    <!-- 时间控制 -->
    <div class="parameter-control">
      <label>时间步:</label>
      <input type="range" id="timestep" min="0" max="1000" value="0" 
             oninput="updateTimestep(this.value)">
      <span id="time-value">0.00s</span>
    </div>
    
    <!-- 可视化选项 -->
    <div class="visualization-options">
      <h5>显示选项:</h5>
      <label><input type="checkbox" id="velocity" checked> 速度场</label>
      <label><input type="checkbox" id="pressure"> 压力场</label>
      <label><input type="checkbox" id="vorticity"> 涡量场</label>
      <label><input type="checkbox" id="streamlines"> 流线</label>
    </div>
    
    <!-- 注意力机制选择 -->
    <div class="attention-selector">
      <label>注意力机制:</label>
      <select id="attention-type" onchange="updateAttention(this.value)">
        <option value="emsa">EMSA (推荐)</option>
        <option value="muse">MUSE</option>
        <option value="ufo">UFO</option>
        <option value="self">Self-Attention</option>
        <option value="sparse">Sparse</option>
      </select>
    </div>
  </div>
  
  <div class="visualization-area">
    <canvas id="flow-canvas" width="800" height="400"></canvas>
    <div class="colorbar">
      <div class="colorbar-gradient"></div>
      <div class="colorbar-labels">
        <span>0</span><span>0.5</span><span>1.0</span><span>1.5</span><span>2.0</span>
      </div>
    </div>
  </div>
</div>
```

#### 2. 📊 实时性能监控

```html
<!-- 性能监控面板 -->
<div class="performance-monitor">
  <div class="metrics-grid">
    <div class="metric-card">
      <h4>🎯 重构精度</h4>
      <div class="metric-value" id="mse-value">0.0234</div>
      <div class="metric-unit">MSE Loss</div>
      <div class="metric-trend positive">↗ +2.3%</div>
    </div>
    
    <div class="metric-card">
      <h4>⚡ 推理速度</h4>
      <div class="metric-value" id="fps-value">67</div>
      <div class="metric-unit">FPS</div>
      <div class="metric-trend positive">↗ +15%</div>
    </div>
    
    <div class="metric-card">
      <h4>💾 内存使用</h4>
      <div class="metric-value" id="memory-value">2.8</div>
      <div class="metric-unit">GB</div>
      <div class="metric-trend negative">↘ -12%</div>
    </div>
    
    <div class="metric-card">
      <h4>🔥 GPU利用率</h4>
      <div class="metric-value" id="gpu-value">85</div>
      <div class="metric-unit">%</div>
      <div class="metric-trend stable">→ 稳定</div>
    </div>
  </div>
  
  <!-- 实时图表 -->
  <div class="charts-container">
    <div class="chart-panel">
      <h4>📈 损失函数收敛</h4>
      <canvas id="loss-chart" width="400" height="200"></canvas>
    </div>
    
    <div class="chart-panel">
      <h4>🎯 注意力权重分布</h4>
      <canvas id="attention-chart" width="400" height="200"></canvas>
    </div>
  </div>
</div>
```

## 🎨 可视化展示

### 🌈 多维度可视化

#### 1. 流场动态演示

<div style="border: 2px solid #ddd; border-radius: 10px; padding: 20px; margin: 20px 0; background: #f9f9f9;">
  <h4>🌊 圆柱绕流可视化</h4>
  
  <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin: 15px 0;">
    <div style="text-align: center;">
      <h5>🎯 原始流场</h5>
      <div style="width: 100%; height: 200px; background: linear-gradient(90deg, #1e3c72 0%, #2a5298 50%, #1e3c72 100%); border-radius: 5px; position: relative; overflow: hidden;">
        <div style="position: absolute; top: 50%; left: 20%; width: 40px; height: 40px; background: #fff; border-radius: 50%; transform: translateY(-50%);"></div>
        <div style="position: absolute; top: 30%; right: 10%; width: 60px; height: 20px; background: rgba(255,255,255,0.3); border-radius: 10px; animation: float 2s ease-in-out infinite;"></div>
        <div style="position: absolute; bottom: 30%; right: 15%; width: 50px; height: 15px; background: rgba(255,255,255,0.2); border-radius: 8px; animation: float 2s ease-in-out infinite reverse;"></div>
      </div>
      <p style="font-size: 12px; color: #666;">CFD仿真结果</p>
    </div>
    
    <div style="text-align: center;">
      <h5>🤖 AI重构结果</h5>
      <div style="width: 100%; height: 200px; background: linear-gradient(90deg, #667eea 0%, #764ba2 50%, #667eea 100%); border-radius: 5px; position: relative; overflow: hidden;">
        <div style="position: absolute; top: 50%; left: 20%; width: 40px; height: 40px; background: #fff; border-radius: 50%; transform: translateY(-50%);"></div>
        <div style="position: absolute; top: 30%; right: 10%; width: 58px; height: 19px; background: rgba(255,255,255,0.3); border-radius: 10px; animation: float 2s ease-in-out infinite;"></div>
        <div style="position: absolute; bottom: 30%; right: 15%; width: 49px; height: 14px; background: rgba(255,255,255,0.2); border-radius: 8px; animation: float 2s ease-in-out infinite reverse;"></div>
      </div>
      <p style="font-size: 12px; color: #666;">VIVTransformer重构 (精度: 97.7%)</p>
    </div>
  </div>
  
  <div style="text-align: center; margin-top: 15px;">
    <button style="background: #4CAF50; color: white; border: none; padding: 8px 16px; border-radius: 5px; margin: 0 5px; cursor: pointer;">▶️ 播放动画</button>
    <button style="background: #2196F3; color: white; border: none; padding: 8px 16px; border-radius: 5px; margin: 0 5px; cursor: pointer;">📊 查看数据</button>
    <button style="background: #FF9800; color: white; border: none; padding: 8px 16px; border-radius: 5px; margin: 0 5px; cursor: pointer;">⚙️ 调整参数</button>
  </div>
</div>

<style>
@keyframes float {
  0%, 100% { transform: translateY(0px); }
  50% { transform: translateY(-10px); }
}
</style>

#### 2. 注意力热图可视化

```javascript
// 注意力权重可视化
class AttentionHeatmap {
    constructor(canvasId) {
        this.canvas = document.getElementById(canvasId);
        this.ctx = this.canvas.getContext('2d');
        this.width = this.canvas.width;
        this.height = this.canvas.height;
    }
    
    drawAttentionWeights(weights, heads = 8) {
        const cellSize = Math.min(this.width, this.height) / Math.sqrt(weights.length);
        
        // 清空画布
        this.ctx.clearRect(0, 0, this.width, this.height);
        
        // 绘制每个注意力头
        for (let head = 0; head < heads; head++) {
            const startX = (head % 4) * (this.width / 4);
            const startY = Math.floor(head / 4) * (this.height / 2);
            
            this.drawSingleHead(weights[head], startX, startY, cellSize);
            
            // 添加标签
            this.ctx.fillStyle = '#333';
            this.ctx.font = '12px Arial';
            this.ctx.fillText(`Head ${head + 1}`, startX + 5, startY + 15);
        }
    }
    
    drawSingleHead(headWeights, startX, startY, cellSize) {
        const size = Math.sqrt(headWeights.length);
        
        for (let i = 0; i < size; i++) {
            for (let j = 0; j < size; j++) {
                const weight = headWeights[i * size + j];
                const intensity = Math.floor(weight * 255);
                
                // 使用热力图颜色
                this.ctx.fillStyle = this.getHeatmapColor(weight);
                this.ctx.fillRect(
                    startX + j * cellSize / size,
                    startY + i * cellSize / size,
                    cellSize / size,
                    cellSize / size
                );
            }
        }
    }
    
    getHeatmapColor(value) {
        // 蓝色到红色的渐变
        const r = Math.floor(255 * value);
        const b = Math.floor(255 * (1 - value));
        return `rgb(${r}, 0, ${b})`;
    }
}

// 使用示例
const heatmap = new AttentionHeatmap('attention-heatmap');

// 模拟注意力权重数据
function generateMockAttentionWeights() {
    const heads = 8;
    const seqLen = 64;
    const weights = [];
    
    for (let h = 0; h < heads; h++) {
        const headWeights = [];
        for (let i = 0; i < seqLen; i++) {
            for (let j = 0; j < seqLen; j++) {
                // 模拟注意力模式
                const distance = Math.abs(i - j);
                const weight = Math.exp(-distance / 10) + Math.random() * 0.1;
                headWeights.push(weight);
            }
        }
        weights.push(headWeights);
    }
    
    return weights;
}

// 定期更新注意力可视化
setInterval(() => {
    const weights = generateMockAttentionWeights();
    heatmap.drawAttentionWeights(weights);
}, 1000);
```

### 📊 交互式图表

#### 1. 损失函数对比工具

<div style="border: 1px solid #ddd; border-radius: 8px; padding: 15px; margin: 15px 0; background: #fafafa;">
  <h4>📈 损失函数实时对比</h4>
  
  <div style="display: flex; gap: 10px; margin-bottom: 15px; flex-wrap: wrap;">
    <label style="display: flex; align-items: center; gap: 5px;">
      <input type="checkbox" checked> 
      <span style="color: #e74c3c;">■</span> MSE Loss
    </label>
    <label style="display: flex; align-items: center; gap: 5px;">
      <input type="checkbox" checked> 
      <span style="color: #3498db;">■</span> SVD Loss
    </label>
    <label style="display: flex; align-items: center; gap: 5px;">
      <input type="checkbox"> 
      <span style="color: #2ecc71;">■</span> Physics Loss
    </label>
    <label style="display: flex; align-items: center; gap: 5px;">
      <input type="checkbox"> 
      <span style="color: #f39c12;">■</span> Total Loss
    </label>
  </div>
  
  <div style="width: 100%; height: 300px; background: white; border: 1px solid #ddd; border-radius: 5px; position: relative;">
    <!-- 这里会插入Chart.js图表 -->
    <div style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); text-align: center; color: #999;">
      <div style="font-size: 48px;">📊</div>
      <p>实时损失函数图表</p>
      <small>点击上方选项切换显示内容</small>
    </div>
  </div>
  
  <div style="display: flex; justify-content: space-between; margin-top: 10px; font-size: 12px; color: #666;">
    <span>Epoch 0</span>
    <span>当前: Epoch 45</span>
    <span>目标: Epoch 100</span>
  </div>
</div>

#### 2. 性能基准测试工具

```html
<!-- 基准测试界面 -->
<div class="benchmark-tool">
  <h4>⚡ 性能基准测试</h4>
  
  <div class="test-configuration">
    <div class="config-group">
      <label>测试数据集:</label>
      <select id="dataset-select">
        <option value="cylinder">圆柱绕流 (Re=4000)</option>
        <option value="airfoil">翼型绕流 (Re=6000)</option>
        <option value="bridge">桥梁断面 (Re=8000)</option>
        <option value="custom">自定义数据</option>
      </select>
    </div>
    
    <div class="config-group">
      <label>批大小:</label>
      <input type="range" id="batch-size" min="8" max="64" value="32" step="8">
      <span id="batch-size-value">32</span>
    </div>
    
    <div class="config-group">
      <label>测试轮数:</label>
      <input type="number" id="test-rounds" value="10" min="1" max="100">
    </div>
  </div>
  
  <div class="attention-comparison">
    <h5>选择对比的注意力机制:</h5>
    <div class="attention-grid">
      <label><input type="checkbox" value="emsa" checked> EMSA</label>
      <label><input type="checkbox" value="muse" checked> MUSE</label>
      <label><input type="checkbox" value="ufo" checked> UFO</label>
      <label><input type="checkbox" value="self"> Self-Attention</label>
      <label><input type="checkbox" value="sparse"> Sparse</label>
      <label><input type="checkbox" value="crossformer"> CrossFormer</label>
    </div>
  </div>
  
  <div class="test-controls">
    <button id="start-benchmark" class="btn-primary">🚀 开始测试</button>
    <button id="stop-benchmark" class="btn-secondary" disabled>⏹️ 停止测试</button>
    <button id="export-results" class="btn-info">📊 导出结果</button>
  </div>
  
  <div class="test-progress" style="display: none;">
    <div class="progress-bar">
      <div class="progress-fill" style="width: 0%;"></div>
    </div>
    <div class="progress-text">准备中...</div>
  </div>
  
  <div class="results-table" style="display: none;">
    <table>
      <thead>
        <tr>
          <th>注意力机制</th>
          <th>平均推理时间</th>
          <th>内存占用</th>
          <th>精度 (MSE)</th>
          <th>综合评分</th>
        </tr>
      </thead>
      <tbody id="results-tbody">
        <!-- 结果会动态插入这里 -->
      </tbody>
    </table>
  </div>
</div>
```

## 🎓 教学演示模式

### 📚 分步教程

#### 1. 新手引导模式

<div style="background: linear-gradient(135deg, #74b9ff 0%, #0984e3 100%); color: white; padding: 20px; border-radius: 10px; margin: 20px 0;">
  <h4>🎯 新手入门指南</h4>
  <p>跟随交互式教程，5分钟掌握VIVTransformer基础操作</p>
  
  <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin-top: 15px;">
    <div style="background: rgba(255,255,255,0.1); padding: 15px; border-radius: 8px;">
      <h5>📖 第一步: 基础概念</h5>
      <p style="font-size: 14px;">了解涡激振动和Transformer架构</p>
      <button style="background: rgba(255,255,255,0.2); color: white; border: 1px solid white; padding: 5px 10px; border-radius: 3px; cursor: pointer;">开始学习</button>
    </div>
    
    <div style="background: rgba(255,255,255,0.1); padding: 15px; border-radius: 8px;">
      <h5>🔧 第二步: 参数调节</h5>
      <p style="font-size: 14px;">学习如何调整模型参数</p>
      <button style="background: rgba(255,255,255,0.2); color: white; border: 1px solid white; padding: 5px 10px; border-radius: 3px; cursor: pointer;">实践操作</button>
    </div>
    
    <div style="background: rgba(255,255,255,0.1); padding: 15px; border-radius: 8px;">
      <h5>📊 第三步: 结果分析</h5>
      <p style="font-size: 14px;">解读可视化结果和性能指标</p>
      <button style="background: rgba(255,255,255,0.2); color: white; border: 1px solid white; padding: 5px 10px; border-radius: 3px; cursor: pointer;">查看示例</button>
    </div>
    
    <div style="background: rgba(255,255,255,0.1); padding: 15px; border-radius: 8px;">
      <h5>🚀 第四步: 高级功能</h5>
      <p style="font-size: 14px;">探索高级特性和优化技巧</p>
      <button style="background: rgba(255,255,255,0.2); color: white; border: 1px solid white; padding: 5px 10px; border-radius: 3px; cursor: pointer;">深入学习</button>
    </div>
  </div>
</div>

#### 2. 交互式代码演示

```html
<!-- 代码演示界面 -->
<div class="code-demo-container">
  <div class="demo-tabs">
    <button class="tab-button active" onclick="showTab('basic')">基础使用</button>
    <button class="tab-button" onclick="showTab('advanced')">高级配置</button>
    <button class="tab-button" onclick="showTab('custom')">自定义模型</button>
  </div>
  
  <div id="basic-tab" class="tab-content active">
    <h4>🚀 快速开始</h4>
    <div class="code-editor">
      <pre><code class="python">
# 导入VIVTransformer
from vivtransformer import VIVTransformer, DataLoader

# 创建模型实例
model = VIVTransformer(
    attention_type='emsa',  # 选择注意力机制
    d_model=256,           # 模型维度
    num_heads=8,           # 注意力头数
    num_layers=6           # 层数
)

# 加载数据
data_loader = DataLoader('cylinder_flow_data.h5')
train_data, val_data = data_loader.split(0.8)

# 训练模型
model.train(train_data, val_data, epochs=100)

# 进行预测
predictions = model.predict(test_data)
      </code></pre>
    </div>
    
    <div class="demo-controls">
      <button onclick="runCode('basic')">▶️ 运行代码</button>
      <button onclick="explainCode('basic')">💡 代码解释</button>
      <button onclick="modifyCode('basic')">✏️ 修改参数</button>
    </div>
    
    <div class="output-panel">
      <h5>📤 输出结果:</h5>
      <div class="console-output">
        <div class="output-line">🔄 正在初始化模型...</div>
        <div class="output-line">✅ 模型创建成功! 参数量: 2.3M</div>
        <div class="output-line">📊 数据加载完成: 训练集 8000 样本, 验证集 2000 样本</div>
        <div class="output-line">🚀 开始训练...</div>
      </div>
    </div>
  </div>
  
  <div id="advanced-tab" class="tab-content">
    <h4>⚙️ 高级配置</h4>
    <div class="code-editor">
      <pre><code class="python">
# 高级配置示例
from vivtransformer.config import Config
from vivtransformer.losses import SVDLoss, MultiLoss

# 创建配置
config = Config({
    'model': {
        'attention_type': 'emsa',
        'd_model': 512,
        'num_heads': 16,
        'dropout': 0.1
    },
    'training': {
        'batch_size': 32,
        'learning_rate': 1e-4,
        'weight_decay': 1e-5
    },
    'loss': {
        'type': 'multi_loss',
        'weights': {
            'reconstruction': 0.6,
            'svd_regularization': 0.2,
            'physics_constraint': 0.2
        }
    }
})

# 使用配置创建模型
model = VIVTransformer.from_config(config)
      </code></pre>
    </div>
  </div>
</div>
```

### 🎮 互动练习

#### 参数调优挑战

<div style="border: 2px dashed #3498db; padding: 20px; border-radius: 10px; margin: 20px 0; background: #f8f9fa;">
  <h4>🎯 挑战: 优化模型性能</h4>
  <p><strong>目标:</strong> 通过调整参数，使模型在测试数据上的MSE损失降到0.025以下</p>
  
  <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin: 15px 0;">
    <div>
      <h5>🎛️ 可调参数:</h5>
      <div style="margin: 10px 0;">
        <label>学习率: </label>
        <input type="range" min="0.0001" max="0.01" step="0.0001" value="0.001" id="lr-slider">
        <span id="lr-value">0.001</span>
      </div>
      <div style="margin: 10px 0;">
        <label>批大小: </label>
        <select id="batch-select">
          <option value="16">16</option>
          <option value="32" selected>32</option>
          <option value="64">64</option>
        </select>
      </div>
      <div style="margin: 10px 0;">
        <label>注意力头数: </label>
        <input type="range" min="4" max="16" step="4" value="8" id="heads-slider">
        <span id="heads-value">8</span>
      </div>
    </div>
    
    <div>
      <h5>📊 当前性能:</h5>
      <div style="background: white; padding: 15px; border-radius: 5px; border: 1px solid #ddd;">
        <div>MSE Loss: <span id="current-mse" style="font-weight: bold; color: #e74c3c;">0.0289</span></div>
        <div>训练时间: <span id="train-time">234s</span></div>
        <div>内存使用: <span id="memory-usage">3.2GB</span></div>
        <div style="margin-top: 10px;">
          <div style="background: #ecf0f1; height: 20px; border-radius: 10px; overflow: hidden;">
            <div style="background: #e74c3c; height: 100%; width: 85%; transition: width 0.3s;"></div>
          </div>
          <small>距离目标: 15.6%</small>
        </div>
      </div>
    </div>
  </div>
  
  <div style="text-align: center; margin-top: 15px;">
    <button style="background: #3498db; color: white; border: none; padding: 10px 20px; border-radius: 5px; margin: 0 5px; cursor: pointer;">🚀 开始训练</button>
    <button style="background: #2ecc71; color: white; border: none; padding: 10px 20px; border-radius: 5px; margin: 0 5px; cursor: pointer;">💾 保存配置</button>
    <button style="background: #f39c12; color: white; border: none; padding: 10px 20px; border-radius: 5px; margin: 0 5px; cursor: pointer;">🔄 重置参数</button>
  </div>
</div>

## 🔧 开发者工具

### 🛠️ API测试工具

```html
<!-- API测试界面 -->
<div class="api-tester">
  <h4>🔌 API接口测试</h4>
  
  <div class="api-endpoint">
    <label>接口地址:</label>
    <div class="endpoint-input">
      <select>
        <option value="POST">POST</option>
        <option value="GET">GET</option>
      </select>
      <input type="text" value="/api/v1/predict" placeholder="输入API端点">
      <button>📋 复制</button>
    </div>
  </div>
  
  <div class="request-body">
    <label>请求体 (JSON):</label>
    <textarea rows="10" placeholder="输入JSON格式的请求数据">
{
  "model_config": {
    "attention_type": "emsa",
    "d_model": 256,
    "num_heads": 8
  },
  "input_data": {
    "flow_field": [[...]], 
    "reynolds_number": 4000,
    "timesteps": 100
  },
  "output_format": "json"
}
    </textarea>
  </div>
  
  <div class="api-controls">
    <button class="btn-primary">🚀 发送请求</button>
    <button class="btn-secondary">📝 生成代码</button>
    <button class="btn-info">📚 查看文档</button>
  </div>
  
  <div class="response-panel">
    <h5>📤 响应结果:</h5>
    <div class="response-headers">
      <strong>状态码:</strong> <span class="status-200">200 OK</span><br>
      <strong>响应时间:</strong> 245ms<br>
      <strong>内容类型:</strong> application/json
    </div>
    <div class="response-body">
      <pre><code class="json">
{
  "status": "success",
  "prediction": {
    "velocity_field": [...],
    "pressure_field": [...],
    "metrics": {
      "mse_loss": 0.0234,
      "inference_time": 67.3
    }
  },
  "model_info": {
    "attention_type": "emsa",
    "parameters": 2340000
  }
}
      </code></pre>
    </div>
  </div>
</div>
```

### 📱 移动端适配

<div style="background: #f8f9fa; padding: 20px; border-radius: 10px; margin: 20px 0; border-left: 4px solid #17a2b8;">
  <h4>📱 移动设备支持</h4>
  <p>VIVTransformer演示平台完全支持移动设备访问，提供触控优化的交互体验。</p>
  
  <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 15px; margin-top: 15px;">
    <div style="text-align: center; padding: 10px;">
      <div style="font-size: 24px;">📱</div>
      <strong>手机端</strong>
      <p style="font-size: 12px; margin: 5px 0;">触控优化界面</p>
    </div>
    <div style="text-align: center; padding: 10px;">
      <div style="font-size: 24px;">📟</div>
      <strong>平板端</strong>
      <p style="font-size: 12px; margin: 5px 0;">大屏幕体验</p>
    </div>
    <div style="text-align: center; padding: 10px;">
      <div style="font-size: 24px;">💻</div>
      <strong>桌面端</strong>
      <p style="font-size: 12px; margin: 5px 0;">完整功能</p>
    </div>
    <div style="text-align: center; padding: 10px;">
      <div style="font-size: 24px;">🌐</div>
      <strong>Web端</strong>
      <p style="font-size: 12px; margin: 5px 0;">跨平台兼容</p>
    </div>
  </div>
</div>

---

## 🚀 立即体验

<div style="text-align: center; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; border-radius: 15px; margin: 30px 0;">
  <h3>🎯 准备好探索VIVTransformer了吗？</h3>
  <p style="font-size: 18px; margin: 20px 0;">选择您的体验方式，开始您的AI涡激振动分析之旅！</p>
  
  <div style="display: flex; justify-content: center; gap: 15px; flex-wrap: wrap; margin-top: 25px;">
    <button style="background: #4CAF50; color: white; border: none; padding: 15px 30px; border-radius: 8px; font-size: 16px; cursor: pointer; transition: transform 0.2s;" onmouseover="this.style.transform='scale(1.05)'" onmouseout="this.style.transform='scale(1)'">
      🚀 启动在线演示
    </button>
    <button style="background: #2196F3; color: white; border: none; padding: 15px 30px; border-radius: 8px; font-size: 16px; cursor: pointer; transition: transform 0.2s;" onmouseover="this.style.transform='scale(1.05)'" onmouseout="this.style.transform='scale(1)'">
      📚 查看教程
    </button>
    <button style="background: #FF9800; color: white; border: none; padding: 15px 30px; border-radius: 8px; font-size: 16px; cursor: pointer; transition: transform 0.2s;" onmouseover="this.style.transform='scale(1.05)'" onmouseout="this.style.transform='scale(1)'">
      💻 下载源码
    </button>
  </div>
  
  <p style="margin-top: 20px; font-size: 14px; opacity: 0.9;">
    💡 提示: 建议使用Chrome或Firefox浏览器以获得最佳体验
  </p>
</div>

---

*体验未来的流体力学分析技术，让AI为您的研究插上翅膀！*