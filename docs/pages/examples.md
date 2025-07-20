---
layout: default
title: 示例代码
nav_order: 16
parent: 核心文档
permalink: /pages/examples/
---

# 示例代码
{: .no_toc }

本页面提供 VIVTransformer 项目的完整代码示例，涵盖从基础使用到高级应用的各种场景。
{: .fs-6 .fw-300 }

## 目录
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## 基础示例

### 1. 快速开始示例

最简单的使用示例，展示如何快速上手 VIVTransformer。

```python
import torch
import numpy as np
from vivtransformer import VIVTransformer, VIVConfig
from vivtransformer.data import VIVDataset
from vivtransformer.utils import create_dataloader

# 1. 创建配置
config = VIVConfig(
    d_model=512,
    num_layers=6,
    num_heads=8,
    d_ff=2048,
    dropout=0.1,
    flow_input_dim=64,
    structure_input_dim=32,
    output_dim=9  # 3D displacement + velocity + force
)

# 2. 创建模型
model = VIVTransformer(config)
print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")

# 3. 准备示例数据
batch_size, seq_len = 4, 100
flow_data = torch.randn(batch_size, seq_len, config.flow_input_dim)
structure_data = torch.randn(batch_size, seq_len, config.structure_input_dim)

# 4. 前向传播
model.eval()
with torch.no_grad():
    outputs = model(flow_data, structure_data)
    
print("输出形状:")
for key, value in outputs.items():
    print(f"  {key}: {value.shape}")
```

### 2. 数据加载示例

展示如何加载和预处理 VIV 数据。

```python
import os
import pandas as pd
from torch.utils.data import DataLoader
from vivtransformer.data import VIVDataset, DataProcessor

# 1. 数据预处理
processor = DataProcessor({
    'normalize': True,
    'sequence_length': 100,
    'overlap': 0.5,
    'features': ['velocity', 'pressure', 'displacement']
})

# 2. 创建数据集
train_dataset = VIVDataset(
    data_dir='data/train',
    split='train',
    sequence_length=100,
    processor=processor
)

val_dataset = VIVDataset(
    data_dir='data/val',
    split='val',
    sequence_length=100,
    processor=processor
)

# 3. 创建数据加载器
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

# 4. 检查数据
for batch in train_loader:
    print("批次数据形状:")
    print(f"  流体数据: {batch['flow_data'].shape}")
    print(f"  结构数据: {batch['structure_data'].shape}")
    print(f"  目标数据: {batch['target'].shape}")
    break
```

---

## 训练示例

### 3. 完整训练流程

展示完整的模型训练过程。

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from vivtransformer import VIVTransformer, VIVConfig
from vivtransformer.training import Trainer, VIVLoss
from vivtransformer.utils import set_seed, save_checkpoint

# 1. 设置随机种子
set_seed(42)

# 2. 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 3. 模型配置
config = VIVConfig(
    d_model=512,
    num_layers=6,
    num_heads=8,
    d_ff=2048,
    dropout=0.1,
    max_seq_length=1000,
    flow_input_dim=64,
    structure_input_dim=32
)

# 4. 创建模型
model = VIVTransformer(config).to(device)

# 5. 损失函数和优化器
criterion = VIVLoss(
    mse_weight=1.0,
    physics_weight=0.1,
    consistency_weight=0.05
)

optimizer = optim.AdamW(
    model.parameters(),
    lr=1e-4,
    weight_decay=1e-5
)

scheduler = CosineAnnealingLR(
    optimizer,
    T_max=100,
    eta_min=1e-6
)

# 6. 训练配置
training_config = {
    'epochs': 100,
    'save_every': 10,
    'eval_every': 5,
    'early_stopping_patience': 15,
    'gradient_clip_norm': 1.0,
    'log_every': 100
}

# 7. 创建训练器
trainer = Trainer(
    model=model,
    criterion=criterion,
    optimizer=optimizer,
    scheduler=scheduler,
    train_dataloader=train_loader,
    val_dataloader=val_loader,
    device=device,
    config=training_config
)

# 8. 开始训练
history = trainer.train()

# 9. 保存最终模型
save_checkpoint({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'config': config,
    'history': history
}, 'checkpoints/final_model.pth')

print("训练完成！")
```

### 4. 自定义训练循环

展示如何实现自定义的训练循环。

```python
import torch
from tqdm import tqdm
from vivtransformer.metrics import compute_metrics
from vivtransformer.utils import AverageMeter

def custom_training_loop(model, train_loader, val_loader, 
                        criterion, optimizer, scheduler, 
                        epochs=100, device='cuda'):
    """
    自定义训练循环
    """
    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': [], 'metrics': []}
    
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss_meter = AverageMeter()
        
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]')
        for batch_idx, batch in enumerate(train_pbar):
            # 数据移动到设备
            flow_data = batch['flow_data'].to(device)
            structure_data = batch['structure_data'].to(device)
            targets = batch['target'].to(device)
            
            # 前向传播
            optimizer.zero_grad()
            outputs = model(flow_data, structure_data)
            
            # 计算损失
            loss_dict = criterion(outputs, targets)
            total_loss = loss_dict['total_loss']
            
            # 反向传播
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            # 更新统计
            train_loss_meter.update(total_loss.item())
            train_pbar.set_postfix({
                'loss': f'{train_loss_meter.avg:.4f}',
                'lr': f'{optimizer.param_groups[0]["lr"]:.2e}'
            })
        
        # 验证阶段
        model.eval()
        val_loss_meter = AverageMeter()
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{epochs} [Val]')
            for batch in val_pbar:
                flow_data = batch['flow_data'].to(device)
                structure_data = batch['structure_data'].to(device)
                targets = batch['target'].to(device)
                
                outputs = model(flow_data, structure_data)
                loss_dict = criterion(outputs, targets)
                
                val_loss_meter.update(loss_dict['total_loss'].item())
                
                # 收集预测结果
                all_predictions.append(outputs['displacement'].cpu())
                all_targets.append(targets.cpu())
                
                val_pbar.set_postfix({'val_loss': f'{val_loss_meter.avg:.4f}'})
        
        # 计算评估指标
        predictions = torch.cat(all_predictions, dim=0).numpy()
        targets = torch.cat(all_targets, dim=0).numpy()
        metrics = compute_metrics(predictions, targets)
        
        # 更新学习率
        scheduler.step()
        
        # 记录历史
        history['train_loss'].append(train_loss_meter.avg)
        history['val_loss'].append(val_loss_meter.avg)
        history['metrics'].append(metrics)
        
        # 保存最佳模型
        if val_loss_meter.avg < best_val_loss:
            best_val_loss = val_loss_meter.avg
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
                'metrics': metrics
            }, 'checkpoints/best_model.pth')
        
        print(f"Epoch {epoch+1}: Train Loss: {train_loss_meter.avg:.4f}, "
              f"Val Loss: {val_loss_meter.avg:.4f}, RMSE: {metrics['rmse']:.4f}")
    
    return history

# 使用自定义训练循环
history = custom_training_loop(
    model, train_loader, val_loader,
    criterion, optimizer, scheduler,
    epochs=100, device=device
)
```

---

## 推理示例

### 5. 模型推理

展示如何使用训练好的模型进行推理。

```python
import torch
import numpy as np
from vivtransformer import VIVTransformer
from vivtransformer.utils import load_checkpoint

def load_model_for_inference(checkpoint_path, device='cuda'):
    """
    加载模型用于推理
    """
    checkpoint = load_checkpoint(checkpoint_path)
    config = checkpoint['config']
    
    model = VIVTransformer(config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, config

def predict_viv_response(model, flow_data, structure_data, device='cuda'):
    """
    预测 VIV 响应
    
    Args:
        model: 训练好的模型
        flow_data: 流体数据 [seq_len, flow_dim]
        structure_data: 结构数据 [seq_len, struct_dim]
        
    Returns:
        预测结果字典
    """
    # 添加批次维度
    flow_data = torch.tensor(flow_data, dtype=torch.float32).unsqueeze(0).to(device)
    structure_data = torch.tensor(structure_data, dtype=torch.float32).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(flow_data, structure_data)
    
    # 移除批次维度并转换为 numpy
    results = {}
    for key, value in outputs.items():
        if isinstance(value, torch.Tensor):
            results[key] = value.squeeze(0).cpu().numpy()
    
    return results

# 示例使用
model, config = load_model_for_inference('checkpoints/best_model.pth')

# 准备测试数据
seq_len = 200
flow_data = np.random.randn(seq_len, config.flow_input_dim)
structure_data = np.random.randn(seq_len, config.structure_input_dim)

# 执行预测
predictions = predict_viv_response(model, flow_data, structure_data)

print("预测结果:")
for key, value in predictions.items():
    if key != 'attention_weights':
        print(f"  {key}: {value.shape}")
```

### 6. 批量推理

展示如何进行批量推理以提高效率。

```python
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

def batch_inference(model, dataloader, device='cuda', save_results=True):
    """
    批量推理
    
    Args:
        model: 训练好的模型
        dataloader: 数据加载器
        device: 计算设备
        save_results: 是否保存结果
        
    Returns:
        所有预测结果
    """
    model.eval()
    all_predictions = []
    all_targets = []
    all_metadata = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='批量推理'):
            flow_data = batch['flow_data'].to(device)
            structure_data = batch['structure_data'].to(device)
            
            # 执行推理
            outputs = model(flow_data, structure_data)
            
            # 收集结果
            all_predictions.append({
                'displacement': outputs['displacement'].cpu(),
                'velocity': outputs['velocity'].cpu(),
                'force': outputs['force'].cpu()
            })
            
            if 'target' in batch:
                all_targets.append(batch['target'])
            
            if 'metadata' in batch:
                all_metadata.append(batch['metadata'])
    
    # 合并结果
    predictions = {}
    for key in all_predictions[0].keys():
        predictions[key] = torch.cat([p[key] for p in all_predictions], dim=0)
    
    results = {'predictions': predictions}
    
    if all_targets:
        results['targets'] = torch.cat(all_targets, dim=0)
    
    if all_metadata:
        results['metadata'] = all_metadata
    
    # 保存结果
    if save_results:
        torch.save(results, 'results/batch_inference_results.pth')
        print("结果已保存到 results/batch_inference_results.pth")
    
    return results

# 创建测试数据加载器
test_dataset = VIVDataset('data/test', split='test')
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 执行批量推理
results = batch_inference(model, test_loader)

print(f"推理完成，共处理 {results['predictions']['displacement'].shape[0]} 个样本")
```

---

## 高级示例

### 7. 注意力可视化

展示如何可视化模型的注意力权重。

```python
import matplotlib.pyplot as plt
import seaborn as sns
from vivtransformer.visualization import plot_attention_weights

def visualize_attention(model, flow_data, structure_data, 
                       layer_idx=0, head_idx=0, save_path=None):
    """
    可视化注意力权重
    """
    model.eval()
    
    # 添加钩子函数获取注意力权重
    attention_weights = []
    
    def hook_fn(module, input, output):
        if hasattr(output, 'attention_weights'):
            attention_weights.append(output.attention_weights)
    
    # 注册钩子
    hooks = []
    for layer in model.transformer_layers:
        hook = layer.self_attention.register_forward_hook(hook_fn)
        hooks.append(hook)
    
    # 前向传播
    with torch.no_grad():
        outputs = model(flow_data.unsqueeze(0), structure_data.unsqueeze(0))
    
    # 移除钩子
    for hook in hooks:
        hook.remove()
    
    # 获取指定层和头的注意力权重
    if attention_weights:
        attn = attention_weights[layer_idx][0, head_idx].cpu().numpy()
        
        # 绘制注意力热图
        plt.figure(figsize=(12, 10))
        sns.heatmap(attn, cmap='Blues', cbar=True)
        plt.title(f'Attention Weights - Layer {layer_idx}, Head {head_idx}')
        plt.xlabel('Key Position')
        plt.ylabel('Query Position')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return attn
    
    return None

# 示例使用
seq_len = 100
flow_data = torch.randn(seq_len, config.flow_input_dim)
structure_data = torch.randn(seq_len, config.structure_input_dim)

attn_weights = visualize_attention(
    model, flow_data, structure_data,
    layer_idx=0, head_idx=0,
    save_path='visualizations/attention_layer0_head0.png'
)
```

### 8. 模型解释性分析

展示如何分析模型的预测行为。

```python
import torch
import numpy as np
from sklearn.decomposition import PCA
from vivtransformer.analysis import FeatureImportanceAnalyzer

def analyze_feature_importance(model, dataloader, device='cuda'):
    """
    分析特征重要性
    """
    analyzer = FeatureImportanceAnalyzer(model)
    
    # 收集特征表示
    all_features = []
    all_targets = []
    
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            flow_data = batch['flow_data'].to(device)
            structure_data = batch['structure_data'].to(device)
            targets = batch['target']
            
            # 获取中间特征表示
            features = analyzer.extract_features(flow_data, structure_data)
            
            all_features.append(features.cpu())
            all_targets.append(targets)
    
    features = torch.cat(all_features, dim=0).numpy()
    targets = torch.cat(all_targets, dim=0).numpy()
    
    # PCA 分析
    pca = PCA(n_components=50)
    features_pca = pca.fit_transform(features.reshape(features.shape[0], -1))
    
    # 计算特征重要性
    importance_scores = analyzer.compute_importance(features_pca, targets)
    
    return {
        'features': features,
        'features_pca': features_pca,
        'importance_scores': importance_scores,
        'explained_variance': pca.explained_variance_ratio_
    }

# 执行分析
analysis_results = analyze_feature_importance(model, val_loader)

print("特征重要性分析完成")
print(f"前10个主成分解释方差比: {analysis_results['explained_variance'][:10]}")
```

### 9. 超参数优化

展示如何使用 Optuna 进行超参数优化。

```python
import optuna
from optuna.integration import PyTorchLightningPruningCallback

def objective(trial):
    """
    Optuna 优化目标函数
    """
    # 建议超参数
    config = VIVConfig(
        d_model=trial.suggest_categorical('d_model', [256, 512, 768]),
        num_layers=trial.suggest_int('num_layers', 4, 8),
        num_heads=trial.suggest_categorical('num_heads', [4, 8, 12]),
        d_ff=trial.suggest_categorical('d_ff', [1024, 2048, 3072]),
        dropout=trial.suggest_float('dropout', 0.05, 0.3),
        learning_rate=trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True),
        flow_input_dim=64,
        structure_input_dim=32
    )
    
    # 创建模型
    model = VIVTransformer(config).to(device)
    
    # 创建优化器
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=trial.suggest_float('weight_decay', 1e-6, 1e-4, log=True)
    )
    
    # 训练模型（简化版）
    model.train()
    total_loss = 0
    num_batches = 0
    
    for epoch in range(10):  # 快速训练几个 epoch
        for batch in train_loader:
            if num_batches >= 100:  # 限制批次数量
                break
                
            flow_data = batch['flow_data'].to(device)
            structure_data = batch['structure_data'].to(device)
            targets = batch['target'].to(device)
            
            optimizer.zero_grad()
            outputs = model(flow_data, structure_data)
            loss = criterion(outputs, targets)['total_loss']
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        # 报告中间结果
        trial.report(total_loss / num_batches, epoch)
        
        # 检查是否应该剪枝
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
    
    return total_loss / num_batches

# 创建研究对象
study = optuna.create_study(
    direction='minimize',
    pruner=optuna.pruners.MedianPruner()
)

# 执行优化
study.optimize(objective, n_trials=50)

print("最佳超参数:")
for key, value in study.best_params.items():
    print(f"  {key}: {value}")

print(f"最佳验证损失: {study.best_value:.4f}")
```

---

## 部署示例

### 10. 模型导出和部署

展示如何导出模型用于生产环境。

```python
import torch
import torch.jit as jit
from vivtransformer.deployment import ModelExporter

def export_model_for_deployment(model, config, export_path):
    """
    导出模型用于部署
    """
    model.eval()
    
    # 创建示例输入
    example_flow = torch.randn(1, 100, config.flow_input_dim)
    example_structure = torch.randn(1, 100, config.structure_input_dim)
    
    # 导出为 TorchScript
    traced_model = jit.trace(model, (example_flow, example_structure))
    traced_model.save(f"{export_path}/model_traced.pt")
    
    # 导出为 ONNX
    torch.onnx.export(
        model,
        (example_flow, example_structure),
        f"{export_path}/model.onnx",
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['flow_data', 'structure_data'],
        output_names=['displacement', 'velocity', 'force'],
        dynamic_axes={
            'flow_data': {0: 'batch_size', 1: 'sequence_length'},
            'structure_data': {0: 'batch_size', 1: 'sequence_length'},
            'displacement': {0: 'batch_size', 1: 'sequence_length'},
            'velocity': {0: 'batch_size', 1: 'sequence_length'},
            'force': {0: 'batch_size', 1: 'sequence_length'}
        }
    )
    
    print(f"模型已导出到 {export_path}")

# 导出模型
export_model_for_deployment(model, config, 'deployment/models')
```

### 11. REST API 服务

展示如何创建 REST API 服务。

```python
from flask import Flask, request, jsonify
import torch
import numpy as np
from vivtransformer.deployment import ModelServer

app = Flask(__name__)

# 加载模型
model_server = ModelServer('deployment/models/model_traced.pt')

@app.route('/predict', methods=['POST'])
def predict():
    """
    VIV 预测 API 端点
    """
    try:
        # 获取输入数据
        data = request.json
        flow_data = np.array(data['flow_data'])
        structure_data = np.array(data['structure_data'])
        
        # 执行预测
        predictions = model_server.predict(flow_data, structure_data)
        
        # 返回结果
        return jsonify({
            'status': 'success',
            'predictions': {
                'displacement': predictions['displacement'].tolist(),
                'velocity': predictions['velocity'].tolist(),
                'force': predictions['force'].tolist()
            }
        })
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 400

@app.route('/health', methods=['GET'])
def health_check():
    """健康检查端点"""
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
```

---

## 工具和实用函数

### 12. 数据可视化工具

```python
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

def plot_viv_trajectory(displacement, time_steps, save_path=None):
    """
    绘制 VIV 轨迹
    
    Args:
        displacement: 位移数据 [seq_len, 3]
        time_steps: 时间步 [seq_len]
        save_path: 保存路径
    """
    fig = plt.figure(figsize=(15, 10))
    
    # 3D 轨迹图
    ax1 = fig.add_subplot(221, projection='3d')
    ax1.plot(displacement[:, 0], displacement[:, 1], displacement[:, 2])
    ax1.set_xlabel('X Displacement')
    ax1.set_ylabel('Y Displacement')
    ax1.set_zlabel('Z Displacement')
    ax1.set_title('3D VIV Trajectory')
    
    # X 方向时间序列
    ax2 = fig.add_subplot(222)
    ax2.plot(time_steps, displacement[:, 0])
    ax2.set_xlabel('Time')
    ax2.set_ylabel('X Displacement')
    ax2.set_title('X-Direction Displacement')
    ax2.grid(True)
    
    # Y 方向时间序列
    ax3 = fig.add_subplot(223)
    ax3.plot(time_steps, displacement[:, 1])
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Y Displacement')
    ax3.set_title('Y-Direction Displacement')
    ax3.grid(True)
    
    # Z 方向时间序列
    ax4 = fig.add_subplot(224)
    ax4.plot(time_steps, displacement[:, 2])
    ax4.set_xlabel('Time')
    ax4.set_ylabel('Z Displacement')
    ax4.set_title('Z-Direction Displacement')
    ax4.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_comparison(predictions, targets, time_steps, save_path=None):
    """
    绘制预测与真实值对比
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    directions = ['X', 'Y', 'Z']
    colors = ['red', 'blue', 'green']
    
    for i, (direction, color) in enumerate(zip(directions, colors)):
        axes[i].plot(time_steps, targets[:, i], 
                    label='True', color='black', linewidth=2)
        axes[i].plot(time_steps, predictions[:, i], 
                    label='Predicted', color=color, linewidth=1.5, alpha=0.8)
        axes[i].set_ylabel(f'{direction} Displacement')
        axes[i].set_title(f'{direction}-Direction Comparison')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    axes[-1].set_xlabel('Time')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

# 示例使用
seq_len = 200
time_steps = np.linspace(0, 10, seq_len)
displacement = np.random.randn(seq_len, 3) * 0.1

plot_viv_trajectory(displacement, time_steps, 'visualizations/viv_trajectory.png')
```

---

## 总结

本页面提供了 VIVTransformer 项目的全面代码示例，涵盖：

- **基础使用**: 快速开始和数据加载
- **模型训练**: 完整训练流程和自定义训练循环
- **模型推理**: 单样本和批量推理
- **高级功能**: 注意力可视化、模型解释性分析、超参数优化
- **生产部署**: 模型导出、REST API 服务
- **工具函数**: 数据可视化和分析工具

这些示例可以帮助您快速上手并深入理解 VIVTransformer 的各种功能和应用场景。

---

## 相关链接

- [API 参考](api-reference) - 详细的 API 文档
- [训练指南](training-guide) - 深入的训练说明
- [架构概览](architecture-overview) - 了解模型架构
- [故障排除](troubleshooting) - 常见问题解决

*需要帮助？查看 [FAQ](faq) 或 [故障排除](troubleshooting) 页面。*