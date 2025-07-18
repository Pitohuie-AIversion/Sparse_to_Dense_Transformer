---
layout: default
title: Deployment Guide
description: 生产环境部署的完整指南
permalink: /pages/deployment-guide/
---

# 部署指南 {#部署指南}

本指南详细介绍了如何在不同环境中部署 VIVTransformer 模型，包括开发、测试和生产环境的部署策略。

## 目录 {#目录}

- [部署概述](#部署概述)
- [环境准备](#环境准备)
- [模型导出](#模型导出)
- [容器化部署](#容器化部署)
- [云平台部署](#云平台部署)
- [边缘设备部署](#边缘设备部署)
- [性能优化](#性能优化)
- [监控与维护](#监控与维护)
- [故障排除](#故障排除)

## 部署概述 {#部署概述}

### 🎯 部署目标 {#部署目标}

VIVTransformer 的部署旨在实现：

1. **高可用性**: 确保服务稳定运行
2. **高性能**: 优化推理速度和吞吐量
3. **可扩展性**: 支持负载动态调整
4. **易维护性**: 简化运维管理
5. **安全性**: 保护模型和数据安全

### 📊 部署架构 {#部署架构}

```python
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum

class DeploymentType(Enum):
    """部署类型"""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"
    EDGE = "edge"

class InferenceBackend(Enum):
    """推理后端"""
    PYTORCH = "pytorch"
    ONNX = "onnx"
    TENSORRT = "tensorrt"
    OPENVINO = "openvino"
    TFLITE = "tflite"

@dataclass
class DeploymentConfig:
    """部署配置"""
    deployment_type: DeploymentType
    backend: InferenceBackend
    model_path: str
    batch_size: int = 1
    max_sequence_length: int = 512
    device: str = "cpu"
    num_workers: int = 1
    enable_optimization: bool = True
    enable_monitoring: bool = True
    
    # 性能配置
    use_mixed_precision: bool = False
    enable_dynamic_batching: bool = False
    max_batch_delay_ms: int = 100
    
    # 安全配置
    enable_auth: bool = False
    api_key: Optional[str] = None
    rate_limit: Optional[int] = None
    
    # 资源配置
    memory_limit: Optional[str] = None
    cpu_limit: Optional[str] = None
    gpu_memory_fraction: float = 0.8

class DeploymentManager:
    """部署管理器"""
    
    def __init__(self, config: DeploymentConfig):
        self.config = config
        self.model = None
        self.preprocessor = None
        self.postprocessor = None
    
    def validate_config(self) -> List[str]:
        """验证配置"""
        errors = []
        
        # 检查模型路径
        if not self.config.model_path:
            errors.append("模型路径不能为空")
        
        # 检查批次大小
        if self.config.batch_size <= 0:
            errors.append("批次大小必须大于0")
        
        # 检查序列长度
        if self.config.max_sequence_length <= 0:
            errors.append("最大序列长度必须大于0")
        
        # 检查设备配置
        if self.config.device not in ["cpu", "cuda", "mps"]:
            errors.append(f"不支持的设备类型: {self.config.device}")
        
        # 检查工作进程数
        if self.config.num_workers < 1:
            errors.append("工作进程数必须至少为1")
        
        return errors
    
    def get_deployment_recommendations(self) -> Dict[str, str]:
        """获取部署建议"""
        recommendations = {}
        
        if self.config.deployment_type == DeploymentType.PRODUCTION:
            recommendations.update({
                "监控": "启用全面监控和日志记录",
                "备份": "配置模型和配置文件备份",
                "负载均衡": "使用负载均衡器分发请求",
                "安全": "启用身份验证和访问控制",
                "缓存": "配置结果缓存以提高性能"
            })
        
        if self.config.device == "cuda":
            recommendations.update({
                "GPU内存": "监控GPU内存使用情况",
                "批处理": "启用动态批处理以提高吞吐量",
                "精度": "考虑使用混合精度推理"
            })
        
        if self.config.deployment_type == DeploymentType.EDGE:
            recommendations.update({
                "模型压缩": "使用模型量化和剪枝",
                "内存优化": "限制内存使用",
                "离线模式": "支持离线推理"
            })
        
        return recommendations
```

## 环境准备 {#环境准备}

### 🔧 系统要求 {#系统要求}

```bash
# 基础系统要求 {#基础系统要求}
# CPU: 4核心以上 {#cpu-4核心以上}
# 内存: 8GB以上 {#内存-8gb以上}
# 存储: 50GB可用空间 {#存储-50gb可用空间}
# 网络: 稳定的网络连接 {#网络-稳定的网络连接}

# Python环境 {#python环境}
python --version  # >= 3.8
pip --version     # 最新版本
```

### 📦 依赖安装 {#依赖安装}

```python
# requirements-deployment.txt {#requirements-deployment-txt}
torch>=1.12.0
torchvision>=0.13.0
numpy>=1.21.0
scipy>=1.7.0
transformers>=4.20.0
onnx>=1.12.0
onnxruntime>=1.12.0
fastapi>=0.68.0
uvicorn>=0.15.0
gunicorn>=20.1.0
prometheus-client>=0.14.0
psutil>=5.8.0
requests>=2.25.0
pydantic>=1.8.0
aiofiles>=0.7.0
pillow>=8.3.0
opencv-python>=4.5.0

# 可选依赖 {#可选依赖}
# tensorrt>=8.0.0  # NVIDIA TensorRT {#tensorrt-8-0-0-nvidia-tensorrt}
# openvino>=2022.1  # Intel OpenVINO {#openvino-2022-1-intel-openvino}
# tensorflow>=2.8.0  # TensorFlow Lite {#tensorflow-2-8-0-tensorflow-lite}
```

### 🐳 Docker环境 {#docker环境}

```dockerfile
# Dockerfile {#dockerfile}
FROM python:3.9-slim

# 设置工作目录 {#设置工作目录}
WORKDIR /app

# 安装系统依赖 {#安装系统依赖}
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件 {#复制依赖文件}
COPY requirements-deployment.txt .

# 安装Python依赖 {#安装python依赖}
RUN pip install --no-cache-dir -r requirements-deployment.txt

# 复制应用代码 {#复制应用代码}
COPY . .

# 创建非root用户 {#创建非root用户}
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# 暴露端口 {#暴露端口}
EXPOSE 8000

# 启动命令 {#启动命令}
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### ⚙️ 环境配置 {#环境配置}

```python
import os
from pathlib import Path
from typing import Optional

class EnvironmentSetup:
    """环境设置"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.model_dir = self.project_root / "models"
        self.config_dir = self.project_root / "configs"
        self.logs_dir = self.project_root / "logs"
    
    def setup_directories(self):
        """创建必要的目录"""
        directories = [
            self.model_dir,
            self.config_dir,
            self.logs_dir,
            self.logs_dir / "inference",
            self.logs_dir / "monitoring"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            print(f"✅ 创建目录: {directory}")
    
    def setup_environment_variables(self):
        """设置环境变量"""
        env_vars = {
            "VIVTRANSFORMER_MODEL_DIR": str(self.model_dir),
            "VIVTRANSFORMER_CONFIG_DIR": str(self.config_dir),
            "VIVTRANSFORMER_LOGS_DIR": str(self.logs_dir),
            "PYTHONPATH": str(self.project_root),
            "OMP_NUM_THREADS": "4",
            "MKL_NUM_THREADS": "4"
        }
        
        for key, value in env_vars.items():
            os.environ[key] = value
            print(f"✅ 设置环境变量: {key}={value}")
    
    def check_dependencies(self) -> Dict[str, bool]:
        """检查依赖"""
        dependencies = {}
        
        # 检查Python包
        packages = [
            "torch", "numpy", "transformers", 
            "fastapi", "uvicorn", "onnx"
        ]
        
        for package in packages:
            try:
                __import__(package)
                dependencies[package] = True
            except ImportError:
                dependencies[package] = False
        
        # 检查CUDA
        try:
            import torch
            dependencies["cuda"] = torch.cuda.is_available()
        except:
            dependencies["cuda"] = False
        
        return dependencies
    
    def generate_setup_script(self) -> str:
        """生成设置脚本"""
        script = '''#!/bin/bash
# VIVTransformer 部署环境设置脚本 {#vivtransformer-部署环境设置脚本}

set -e

echo "🚀 开始设置 VIVTransformer 部署环境..."

# 检查Python版本 {#检查python版本}
echo "📋 检查Python版本..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python版本: $python_version"

# 创建虚拟环境 {#创建虚拟环境}
echo "🔧 创建虚拟环境..."
python3 -m venv venv
source venv/bin/activate

# 升级pip {#升级pip}
echo "📦 升级pip..."
pip install --upgrade pip

# 安装依赖 {#安装依赖}
echo "📚 安装依赖包..."
pip install -r requirements-deployment.txt

# 设置环境变量 {#设置环境变量}
echo "⚙️ 设置环境变量..."
export VIVTRANSFORMER_MODEL_DIR="$(pwd)/models"
export VIVTRANSFORMER_CONFIG_DIR="$(pwd)/configs"
export VIVTRANSFORMER_LOGS_DIR="$(pwd)/logs"

# 创建目录 {#创建目录}
echo "📁 创建必要目录..."
mkdir -p models configs logs/inference logs/monitoring

# 检查安装 {#检查安装}
echo "✅ 验证安装..."
python -c "import torch; print(f'PyTorch版本: {torch.__version__}')"
python -c "import transformers; print(f'Transformers版本: {transformers.__version__}')"

echo "🎉 环境设置完成！"
echo "💡 使用 'source venv/bin/activate' 激活环境"
'''
        return script
    
    def run_setup(self):
        """运行完整设置"""
        print("🚀 开始 VIVTransformer 部署环境设置...")
        
        # 创建目录
        self.setup_directories()
        
        # 设置环境变量
        self.setup_environment_variables()
        
        # 检查依赖
        deps = self.check_dependencies()
        print("\n📋 依赖检查结果:")
        for dep, available in deps.items():
            status = "✅" if available else "❌"
            print(f"  {status} {dep}")
        
        # 生成设置脚本
        setup_script = self.generate_setup_script()
        script_path = self.project_root / "setup_deployment.sh"
        with open(script_path, "w") as f:
            f.write(setup_script)
        os.chmod(script_path, 0o755)
        print(f"\n📝 生成设置脚本: {script_path}")
        
        print("\n🎉 环境设置完成！")

# 使用示例 {#使用示例}
if __name__ == "__main__":
    setup = EnvironmentSetup()
    setup.run_setup()
```

## 模型导出 {#模型导出}

### 🔄 格式转换 {#格式转换}

```python
import torch
import onnx
from pathlib import Path
from typing import Dict, List, Tuple, Optional

class ModelExporter:
    """模型导出器"""
    
    def __init__(self, model_path: str, output_dir: str):
        self.model_path = Path(model_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def export_to_onnx(self, 
                      input_shape: Tuple[int, ...] = (1, 512),
                      opset_version: int = 11,
                      dynamic_axes: Optional[Dict] = None) -> str:
        """导出为ONNX格式"""
        print("🔄 开始导出ONNX模型...")
        
        # 加载模型
        model = torch.load(self.model_path, map_location='cpu')
        model.eval()
        
        # 创建示例输入
        dummy_input = torch.randint(0, 1000, input_shape, dtype=torch.long)
        
        # 设置动态轴
        if dynamic_axes is None:
            dynamic_axes = {
                'input': {0: 'batch_size', 1: 'sequence_length'},
                'output': {0: 'batch_size'}
            }
        
        # 导出路径
        onnx_path = self.output_dir / "model.onnx"
        
        # 导出模型
        torch.onnx.export(
            model,
            dummy_input,
            str(onnx_path),
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes=dynamic_axes,
            verbose=False
        )
        
        # 验证ONNX模型
        onnx_model = onnx.load(str(onnx_path))
        onnx.checker.check_model(onnx_model)
        
        print(f"✅ ONNX模型导出成功: {onnx_path}")
        return str(onnx_path)
    
    def export_to_torchscript(self, 
                              input_shape: Tuple[int, ...] = (1, 512)) -> str:
        """导出为TorchScript格式"""
        print("🔄 开始导出TorchScript模型...")
        
        # 加载模型
        model = torch.load(self.model_path, map_location='cpu')
        model.eval()
        
        # 创建示例输入
        dummy_input = torch.randint(0, 1000, input_shape, dtype=torch.long)
        
        # 追踪模型
        traced_model = torch.jit.trace(model, dummy_input)
        
        # 导出路径
        script_path = self.output_dir / "model.pt"
        
        # 保存模型
        traced_model.save(str(script_path))
        
        print(f"✅ TorchScript模型导出成功: {script_path}")
        return str(script_path)
    
    def optimize_onnx_model(self, onnx_path: str) -> str:
        """优化ONNX模型"""
        try:
            from onnxruntime.tools import optimizer
            
            print("🔧 开始优化ONNX模型...")
            
            # 优化配置
            opt_model_path = self.output_dir / "model_optimized.onnx"
            
            # 执行优化
            optimizer.optimize_model(
                onnx_path,
                str(opt_model_path),
                file_type='onnx',
                optimization_level=99
            )
            
            print(f"✅ ONNX模型优化完成: {opt_model_path}")
            return str(opt_model_path)
            
        except ImportError:
            print("⚠️ 未安装onnxruntime-tools，跳过优化")
            return onnx_path
    
    def quantize_model(self, model_path: str, backend: str = "onnx") -> str:
        """模型量化"""
        print(f"🔧 开始{backend}模型量化...")
        
        if backend == "onnx":
            return self._quantize_onnx(model_path)
        elif backend == "pytorch":
            return self._quantize_pytorch(model_path)
        else:
            raise ValueError(f"不支持的量化后端: {backend}")
    
    def _quantize_onnx(self, onnx_path: str) -> str:
        """ONNX模型量化"""
        try:
            from onnxruntime.quantization import quantize_dynamic, QuantType
            
            quantized_path = self.output_dir / "model_quantized.onnx"
            
            quantize_dynamic(
                onnx_path,
                str(quantized_path),
                weight_type=QuantType.QUInt8
            )
            
            print(f"✅ ONNX量化完成: {quantized_path}")
            return str(quantized_path)
            
        except ImportError:
            print("⚠️ 未安装量化工具，跳过量化")
            return onnx_path
    
    def _quantize_pytorch(self, model_path: str) -> str:
        """PyTorch模型量化"""
        # 加载模型
        model = torch.load(model_path, map_location='cpu')
        model.eval()
        
        # 动态量化
        quantized_model = torch.quantization.quantize_dynamic(
            model,
            {torch.nn.Linear},
            dtype=torch.qint8
        )
        
        # 保存量化模型
        quantized_path = self.output_dir / "model_quantized.pt"
        torch.save(quantized_model, quantized_path)
        
        print(f"✅ PyTorch量化完成: {quantized_path}")
        return str(quantized_path)
    
    def export_all_formats(self, input_shape: Tuple[int, ...] = (1, 512)) -> Dict[str, str]:
        """导出所有格式"""
        exported_models = {}
        
        try:
            # 导出ONNX
            onnx_path = self.export_to_onnx(input_shape)
            exported_models['onnx'] = onnx_path
            
            # 优化ONNX
            optimized_onnx = self.optimize_onnx_model(onnx_path)
            exported_models['onnx_optimized'] = optimized_onnx
            
            # 量化ONNX
            quantized_onnx = self.quantize_model(optimized_onnx, "onnx")
            exported_models['onnx_quantized'] = quantized_onnx
            
        except Exception as e:
            print(f"❌ ONNX导出失败: {e}")
        
        try:
            # 导出TorchScript
            script_path = self.export_to_torchscript(input_shape)
            exported_models['torchscript'] = script_path
            
            # 量化TorchScript
            quantized_script = self.quantize_model(script_path, "pytorch")
            exported_models['torchscript_quantized'] = quantized_script
            
        except Exception as e:
            print(f"❌ TorchScript导出失败: {e}")
        
        return exported_models
    
    def benchmark_models(self, exported_models: Dict[str, str], 
                        input_shape: Tuple[int, ...] = (1, 512),
                        num_runs: int = 100) -> Dict[str, Dict[str, float]]:
        """基准测试导出的模型"""
        import time
        import numpy as np
        
        results = {}
        
        for model_type, model_path in exported_models.items():
            print(f"🔍 测试 {model_type} 模型性能...")
            
            try:
                # 加载模型
                if 'onnx' in model_type:
                    import onnxruntime as ort
                    session = ort.InferenceSession(model_path)
                    input_name = session.get_inputs()[0].name
                    
                    # 预热
                    dummy_input = np.random.randint(0, 1000, input_shape, dtype=np.int64)
                    for _ in range(10):
                        session.run(None, {input_name: dummy_input})
                    
                    # 计时
                    times = []
                    for _ in range(num_runs):
                        start_time = time.time()
                        session.run(None, {input_name: dummy_input})
                        times.append(time.time() - start_time)
                
                elif 'torchscript' in model_type:
                    model = torch.jit.load(model_path)
                    model.eval()
                    
                    # 预热
                    dummy_input = torch.randint(0, 1000, input_shape, dtype=torch.long)
                    with torch.no_grad():
                        for _ in range(10):
                            model(dummy_input)
                    
                    # 计时
                    times = []
                    with torch.no_grad():
                        for _ in range(num_runs):
                            start_time = time.time()
                            model(dummy_input)
                            times.append(time.time() - start_time)
                
                # 计算统计信息
                times = np.array(times) * 1000  # 转换为毫秒
                results[model_type] = {
                    'mean_latency_ms': float(np.mean(times)),
                    'std_latency_ms': float(np.std(times)),
                    'min_latency_ms': float(np.min(times)),
                    'max_latency_ms': float(np.max(times)),
                    'throughput_qps': float(1000 / np.mean(times))
                }
                
                print(f"  平均延迟: {results[model_type]['mean_latency_ms']:.2f}ms")
                print(f"  吞吐量: {results[model_type]['throughput_qps']:.2f} QPS")
                
            except Exception as e:
                print(f"❌ {model_type} 测试失败: {e}")
                results[model_type] = {'error': str(e)}
        
        return results
```

### 📊 导出脚本 {#导出脚本}

```python
#!/usr/bin/env python3
# export_models.py {#export-models-py}

import argparse
import json
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="VIVTransformer模型导出工具")
    parser.add_argument("--model-path", required=True, help="模型文件路径")
    parser.add_argument("--output-dir", required=True, help="输出目录")
    parser.add_argument("--batch-size", type=int, default=1, help="批次大小")
    parser.add_argument("--seq-length", type=int, default=512, help="序列长度")
    parser.add_argument("--formats", nargs="+", 
                       choices=["onnx", "torchscript", "all"],
                       default=["all"], help="导出格式")
    parser.add_argument("--optimize", action="store_true", help="启用优化")
    parser.add_argument("--quantize", action="store_true", help="启用量化")
    parser.add_argument("--benchmark", action="store_true", help="运行基准测试")
    
    args = parser.parse_args()
    
    # 创建导出器
    exporter = ModelExporter(args.model_path, args.output_dir)
    
    # 输入形状
    input_shape = (args.batch_size, args.seq_length)
    
    # 导出模型
    if "all" in args.formats:
        exported_models = exporter.export_all_formats(input_shape)
    else:
        exported_models = {}
        for fmt in args.formats:
            if fmt == "onnx":
                path = exporter.export_to_onnx(input_shape)
                exported_models['onnx'] = path
                if args.optimize:
                    path = exporter.optimize_onnx_model(path)
                    exported_models['onnx_optimized'] = path
                if args.quantize:
                    path = exporter.quantize_model(path, "onnx")
                    exported_models['onnx_quantized'] = path
            elif fmt == "torchscript":
                path = exporter.export_to_torchscript(input_shape)
                exported_models['torchscript'] = path
                if args.quantize:
                    path = exporter.quantize_model(path, "pytorch")
                    exported_models['torchscript_quantized'] = path
    
    # 保存导出信息
    export_info = {
        "input_shape": input_shape,
        "exported_models": exported_models,
        "optimization_enabled": args.optimize,
        "quantization_enabled": args.quantize
    }
    
    # 运行基准测试
    if args.benchmark:
        print("\n🔍 开始基准测试...")
        benchmark_results = exporter.benchmark_models(exported_models, input_shape)
        export_info["benchmark_results"] = benchmark_results
    
    # 保存结果
    info_path = Path(args.output_dir) / "export_info.json"
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(export_info, f, indent=2, ensure_ascii=False)
    
    print(f"\n📝 导出信息保存至: {info_path}")
    print("🎉 模型导出完成！")

if __name__ == "__main__":
    main()
```

## 容器化部署 {#容器化部署}

### 🐳 Docker部署 {#docker部署}

```dockerfile
# 多阶段构建Dockerfile {#多阶段构建dockerfile}
# 阶段1: 构建阶段 {#阶段1-构建阶段}
FROM python:3.9-slim as builder

WORKDIR /build

# 安装构建依赖 {#安装构建依赖}
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件 {#复制依赖文件}
COPY requirements-deployment.txt .

# 安装Python依赖到临时目录 {#安装python依赖到临时目录}
RUN pip install --user --no-cache-dir -r requirements-deployment.txt

# 阶段2: 运行阶段 {#阶段2-运行阶段}
FROM python:3.9-slim

# 设置标签 {#设置标签}
LABEL maintainer="VIVTransformer Team"
LABEL version="1.0.0"
LABEL description="VIVTransformer Inference Server"

# 安装运行时依赖 {#安装运行时依赖}
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 从构建阶段复制Python包 {#从构建阶段复制python包}
COPY --from=builder /root/.local /root/.local

# 设置PATH {#设置path}
ENV PATH=/root/.local/bin:$PATH

# 创建应用目录 {#创建应用目录}
WORKDIR /app

# 创建非root用户 {#创建非root用户}
RUN useradd -m -u 1000 appuser

# 复制应用代码 {#复制应用代码}
COPY --chown=appuser:appuser . .

# 创建必要目录 {#创建必要目录}
RUN mkdir -p models logs && chown -R appuser:appuser /app

# 切换到非root用户 {#切换到非root用户}
USER appuser

# 设置环境变量 {#设置环境变量}
ENV PYTHONPATH=/app
ENV VIVTRANSFORMER_MODEL_DIR=/app/models
ENV VIVTRANSFORMER_CONFIG_DIR=/app/configs
ENV VIVTRANSFORMER_LOGS_DIR=/app/logs

# 健康检查 {#健康检查}
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# 暴露端口 {#暴露端口}
EXPOSE 8000

# 启动命令 {#启动命令}
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
```

### 🚀 Docker Compose {#docker-compose}

```yaml
# docker-compose.yml {#docker-compose-yml}
version: '3.8'

services:
  vivtransformer-api:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: vivtransformer-api
    ports:
      - "8000:8000"
    volumes:
      - ./models:/app/models:ro
      - ./configs:/app/configs:ro
      - ./logs:/app/logs
    environment:
      - DEPLOYMENT_TYPE=production
      - BACKEND=onnx
      - DEVICE=cpu
      - BATCH_SIZE=4
      - MAX_SEQUENCE_LENGTH=512
      - ENABLE_MONITORING=true
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s
    networks:
      - vivtransformer-network
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"

  # 负载均衡器
  nginx:
    image: nginx:alpine
    container_name: vivtransformer-nginx
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - ./ssl:/etc/nginx/ssl:ro
    depends_on:
      - vivtransformer-api
    restart: unless-stopped
    networks:
      - vivtransformer-network

  # 监控服务
  prometheus:
    image: prom/prometheus:latest
    container_name: vivtransformer-prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
    restart: unless-stopped
    networks:
      - vivtransformer-network

  # 可视化监控
  grafana:
    image: grafana/grafana:latest
    container_name: vivtransformer-grafana
    ports:
      - "3000:3000"
    volumes:
      - grafana_data:/var/lib/grafana
      - ./grafana/dashboards:/etc/grafana/provisioning/dashboards:ro
      - ./grafana/datasources:/etc/grafana/provisioning/datasources:ro
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin123
    restart: unless-stopped
    networks:
      - vivtransformer-network

  # Redis缓存
  redis:
    image: redis:alpine
    container_name: vivtransformer-redis
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    command: redis-server --appendonly yes
    restart: unless-stopped
    networks:
      - vivtransformer-network

volumes:
  prometheus_data:
  grafana_data:
  redis_data:

networks:
  vivtransformer-network:
    driver: bridge
```

### ⚙️ Nginx配置 {#nginx配置}

```nginx
# nginx.conf {#nginx-conf}
events {
    worker_connections 1024;
}

http {
    upstream vivtransformer_backend {
        least_conn;
        server vivtransformer-api:8000 max_fails=3 fail_timeout=30s;
        # 可以添加更多实例
        # server vivtransformer-api-2:8000 max_fails=3 fail_timeout=30s;
    }

    # 限流配置
    limit_req_zone $binary_remote_addr zone=api_limit:10m rate=10r/s;
    
    # 日志格式
    log_format main '$remote_addr - $remote_user [$time_local] "$request" '
                   '$status $body_bytes_sent "$http_referer" '
                   '"$http_user_agent" "$http_x_forwarded_for" '
                   'rt=$request_time uct="$upstream_connect_time" '
                   'uht="$upstream_header_time" urt="$upstream_response_time"';

    access_log /var/log/nginx/access.log main;
    error_log /var/log/nginx/error.log warn;

    # 基础配置
    sendfile on;
    tcp_nopush on;
    tcp_nodelay on;
    keepalive_timeout 65;
    types_hash_max_size 2048;
    client_max_body_size 10M;

    # Gzip压缩
    gzip on;
    gzip_vary on;
    gzip_min_length 1024;
    gzip_types text/plain text/css application/json application/javascript text/xml application/xml;

    server {
        listen 80;
        server_name _;

        # 健康检查
        location /health {
            proxy_pass http://vivtransformer_backend/health;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }

        # API路由
        location /api/ {
            # 应用限流
            limit_req zone=api_limit burst=20 nodelay;
            
            proxy_pass http://vivtransformer_backend;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            
            # 超时设置
            proxy_connect_timeout 30s;
            proxy_send_timeout 60s;
            proxy_read_timeout 60s;
            
            # 缓冲设置
            proxy_buffering on;
            proxy_buffer_size 4k;
            proxy_buffers 8 4k;
        }

        # 静态文件
        location /static/ {
            alias /app/static/;
            expires 1d;
            add_header Cache-Control "public, immutable";
        }

        # 默认路由
        location / {
            return 301 /api/docs;
        }
    }

    # HTTPS配置（可选）
    # server {
    #     listen 443 ssl http2;
    #     server_name your-domain.com;
    #     
    #     ssl_certificate /etc/nginx/ssl/cert.pem;
    #     ssl_certificate_key /etc/nginx/ssl/key.pem;
    #     
    #     # SSL配置
    #     ssl_protocols TLSv1.2 TLSv1.3;
    #     ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512:ECDHE-RSA-AES256-GCM-SHA384:DHE-RSA-AES256-GCM-SHA384;
    #     ssl_prefer_server_ciphers off;
    #     
    #     # 其他配置同HTTP
    # }
}
```

### 🔧 部署脚本 {#部署脚本}

```bash
#!/bin/bash
# deploy.sh - VIVTransformer Docker部署脚本 {#deploy-sh-vivtransformer-docker部署脚本}

set -e

# 配置变量 {#配置变量}
IMAGE_NAME="vivtransformer"
IMAGE_TAG="latest"
CONTAINER_NAME="vivtransformer-api"
PORT="8000"
MODEL_DIR="./models"
CONFIG_DIR="./configs"
LOGS_DIR="./logs"

# 颜色输出 {#颜色输出}
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

echo_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

echo_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查Docker {#检查docker}
check_docker() {
    if ! command -v docker &> /dev/null; then
        echo_error "Docker未安装，请先安装Docker"
        exit 1
    fi
    
    if ! docker info &> /dev/null; then
        echo_error "Docker服务未运行，请启动Docker服务"
        exit 1
    fi
    
    echo_info "Docker检查通过"
}

# 检查必要文件 {#检查必要文件}
check_files() {
    local required_files=("Dockerfile" "requirements-deployment.txt")
    
    for file in "${required_files[@]}"; do
        if [[ ! -f "$file" ]]; then
            echo_error "缺少必要文件: $file"
            exit 1
        fi
    done
    
    echo_info "文件检查通过"
}

# 创建必要目录 {#创建必要目录}
setup_directories() {
    local dirs=("$MODEL_DIR" "$CONFIG_DIR" "$LOGS_DIR")
    
    for dir in "${dirs[@]}"; do
        if [[ ! -d "$dir" ]]; then
            mkdir -p "$dir"
            echo_info "创建目录: $dir"
        fi
    done
}

# 构建镜像 {#构建镜像}
build_image() {
    echo_info "开始构建Docker镜像..."
    
    docker build -t "$IMAGE_NAME:$IMAGE_TAG" .
    
    if [[ $? -eq 0 ]]; then
        echo_info "镜像构建成功: $IMAGE_NAME:$IMAGE_TAG"
    else
        echo_error "镜像构建失败"
        exit 1
    fi
}

# 停止现有容器 {#停止现有容器}
stop_existing() {
    if docker ps -q -f name="$CONTAINER_NAME" | grep -q .; then
        echo_info "停止现有容器: $CONTAINER_NAME"
        docker stop "$CONTAINER_NAME"
        docker rm "$CONTAINER_NAME"
    fi
}

# 运行容器 {#运行容器}
run_container() {
    echo_info "启动容器: $CONTAINER_NAME"
    
    docker run -d \
        --name "$CONTAINER_NAME" \
        -p "$PORT:8000" \
        -v "$(pwd)/$MODEL_DIR:/app/models:ro" \
        -v "$(pwd)/$CONFIG_DIR:/app/configs:ro" \
        -v "$(pwd)/$LOGS_DIR:/app/logs" \
        -e DEPLOYMENT_TYPE=production \
        -e BACKEND=onnx \
        -e DEVICE=cpu \
        --restart unless-stopped \
        "$IMAGE_NAME:$IMAGE_TAG"
    
    if [[ $? -eq 0 ]]; then
        echo_info "容器启动成功"
    else
        echo_error "容器启动失败"
        exit 1
    fi
}

# 健康检查 {#健康检查}
health_check() {
    echo_info "等待服务启动..."
    sleep 10
    
    local max_attempts=30
    local attempt=1
    
    while [[ $attempt -le $max_attempts ]]; do
        if curl -f "http://localhost:$PORT/health" &> /dev/null; then
            echo_info "服务健康检查通过"
            return 0
        fi
        
        echo_warn "健康检查失败，重试 $attempt/$max_attempts"
        sleep 5
        ((attempt++))
    done
    
    echo_error "服务启动失败，请检查日志"
    docker logs "$CONTAINER_NAME"
    exit 1
}

# 显示部署信息 {#显示部署信息}
show_info() {
    echo_info "部署完成！"
    echo "容器名称: $CONTAINER_NAME"
    echo "访问地址: http://localhost:$PORT"
    echo "API文档: http://localhost:$PORT/docs"
    echo "健康检查: http://localhost:$PORT/health"
    echo ""
    echo "常用命令:"
    echo "  查看日志: docker logs $CONTAINER_NAME"
    echo "  停止服务: docker stop $CONTAINER_NAME"
    echo "  重启服务: docker restart $CONTAINER_NAME"
    echo "  进入容器: docker exec -it $CONTAINER_NAME bash"
}

# 主函数 {#主函数}
main() {
    echo_info "开始VIVTransformer Docker部署..."
    
    check_docker
    check_files
    setup_directories
    build_image
    stop_existing
    run_container
    health_check
    show_info
}

# 解析命令行参数 {#解析命令行参数}
while [[ $# -gt 0 ]]; do
    case $1 in
        --port)
            PORT="$2"
            shift 2
            ;;
        --tag)
            IMAGE_TAG="$2"
            shift 2
            ;;
        --name)
            CONTAINER_NAME="$2"
            shift 2
            ;;
        --help)
            echo "用法: $0 [选项]"
            echo "选项:"
            echo "  --port PORT     指定端口 (默认: 8000)"
            echo "  --tag TAG       指定镜像标签 (默认: latest)"
            echo "  --name NAME     指定容器名称 (默认: vivtransformer-api)"
            echo "  --help          显示帮助信息"
            exit 0
            ;;
        *)
            echo_error "未知选项: $1"
            exit 1
            ;;
    esac
done

# 执行主函数 {#执行主函数}
main
```

## 云平台部署 {#云平台部署}

### ☁️ AWS部署 {#aws部署}

```yaml
# aws-deployment.yml {#aws-deployment-yml}
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vivtransformer-deployment
  labels:
    app: vivtransformer
spec:
  replicas: 3
  selector:
    matchLabels:
      app: vivtransformer
  template:
    metadata:
      labels:
        app: vivtransformer
    spec:
      containers:
      - name: vivtransformer
        image: your-ecr-repo/vivtransformer:latest
        ports:
        - containerPort: 8000
        env:
        - name: DEPLOYMENT_TYPE
          value: "production"
        - name: BACKEND
          value: "onnx"
        - name: DEVICE
          value: "cpu"
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 60
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        volumeMounts:
        - name: model-storage
          mountPath: /app/models
          readOnly: true
      volumes:
      - name: model-storage
        persistentVolumeClaim:
          claimName: model-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: vivtransformer-service
spec:
  selector:
    app: vivtransformer
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: vivtransformer-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: vivtransformer-deployment
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

### 🔧 Terraform配置 {#terraform配置}

```hcl
# main.tf {#main-tf}
provider "aws" {
  region = var.aws_region
}

# VPC配置 {#vpc配置}
resource "aws_vpc" "vivtransformer_vpc" {
  cidr_block           = "10.0.0.0/16"
  enable_dns_hostnames = true
  enable_dns_support   = true
  
  tags = {
    Name = "vivtransformer-vpc"
  }
}

# 子网配置 {#子网配置}
resource "aws_subnet" "public_subnet" {
  count             = 2
  vpc_id            = aws_vpc.vivtransformer_vpc.id
  cidr_block        = "10.0.${count.index + 1}.0/24"
  availability_zone = data.aws_availability_zones.available.names[count.index]
  
  map_public_ip_on_launch = true
  
  tags = {
    Name = "vivtransformer-public-subnet-${count.index + 1}"
  }
}

# ECS集群 {#ecs集群}
resource "aws_ecs_cluster" "vivtransformer_cluster" {
  name = "vivtransformer-cluster"
  
  setting {
    name  = "containerInsights"
    value = "enabled"
  }
}

# ECS任务定义 {#ecs任务定义}
resource "aws_ecs_task_definition" "vivtransformer_task" {
  family                   = "vivtransformer"
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = "1024"
  memory                   = "2048"
  execution_role_arn       = aws_iam_role.ecs_execution_role.arn
  task_role_arn           = aws_iam_role.ecs_task_role.arn
  
  container_definitions = jsonencode([
    {
      name  = "vivtransformer"
      image = "${aws_ecr_repository.vivtransformer_repo.repository_url}:latest"
      
      portMappings = [
        {
          containerPort = 8000
          protocol      = "tcp"
        }
      ]
      
      environment = [
        {
          name  = "DEPLOYMENT_TYPE"
          value = "production"
        },
        {
          name  = "BACKEND"
          value = "onnx"
        }
      ]
      
      logConfiguration = {
        logDriver = "awslogs"
        options = {
          "awslogs-group"         = aws_cloudwatch_log_group.vivtransformer_logs.name
          "awslogs-region"        = var.aws_region
          "awslogs-stream-prefix" = "ecs"
        }
      }
      
      healthCheck = {
        command     = ["CMD-SHELL", "curl -f http://localhost:8000/health || exit 1"]
        interval    = 30
        timeout     = 5
        retries     = 3
        startPeriod = 60
      }
    }
  ])
}

# ECS服务 {#ecs服务}
resource "aws_ecs_service" "vivtransformer_service" {
  name            = "vivtransformer-service"
  cluster         = aws_ecs_cluster.vivtransformer_cluster.id
  task_definition = aws_ecs_task_definition.vivtransformer_task.arn
  desired_count   = 2
  launch_type     = "FARGATE"
  
  network_configuration {
    subnets          = aws_subnet.public_subnet[*].id
    security_groups  = [aws_security_group.ecs_sg.id]
    assign_public_ip = true
  }
  
  load_balancer {
    target_group_arn = aws_lb_target_group.vivtransformer_tg.arn
    container_name   = "vivtransformer"
    container_port   = 8000
  }
  
  depends_on = [aws_lb_listener.vivtransformer_listener]
}

# 应用负载均衡器 {#应用负载均衡器}
resource "aws_lb" "vivtransformer_alb" {
  name               = "vivtransformer-alb"
  internal           = false
  load_balancer_type = "application"
  security_groups    = [aws_security_group.alb_sg.id]
  subnets            = aws_subnet.public_subnet[*].id
  
  enable_deletion_protection = false
}

# 目标组 {#目标组}
resource "aws_lb_target_group" "vivtransformer_tg" {
  name        = "vivtransformer-tg"
  port        = 8000
  protocol    = "HTTP"
  vpc_id      = aws_vpc.vivtransformer_vpc.id
  target_type = "ip"
  
  health_check {
    enabled             = true
    healthy_threshold   = 2
    interval            = 30
    matcher             = "200"
    path                = "/health"
    port                = "traffic-port"
    protocol            = "HTTP"
    timeout             = 5
    unhealthy_threshold = 2
  }
}

# 监听器 {#监听器}
resource "aws_lb_listener" "vivtransformer_listener" {
  load_balancer_arn = aws_lb.vivtransformer_alb.arn
  port              = "80"
  protocol          = "HTTP"
  
  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.vivtransformer_tg.arn
  }
}

# 自动扩缩容 {#自动扩缩容}
resource "aws_appautoscaling_target" "ecs_target" {
  max_capacity       = 10
  min_capacity       = 2
  resource_id        = "service/${aws_ecs_cluster.vivtransformer_cluster.name}/${aws_ecs_service.vivtransformer_service.name}"
  scalable_dimension = "ecs:service:DesiredCount"
  service_namespace  = "ecs"
}

resource "aws_appautoscaling_policy" "ecs_policy_cpu" {
  name               = "cpu-scaling"
  policy_type        = "TargetTrackingScaling"
  resource_id        = aws_appautoscaling_target.ecs_target.resource_id
  scalable_dimension = aws_appautoscaling_target.ecs_target.scalable_dimension
  service_namespace  = aws_appautoscaling_target.ecs_target.service_namespace
  
  target_tracking_scaling_policy_configuration {
    predefined_metric_specification {
      predefined_metric_type = "ECSServiceAverageCPUUtilization"
    }
    target_value = 70.0
  }
}

# ECR仓库 {#ecr仓库}
resource "aws_ecr_repository" "vivtransformer_repo" {
  name                 = "vivtransformer"
  image_tag_mutability = "MUTABLE"
  
  image_scanning_configuration {
    scan_on_push = true
  }
}

# CloudWatch日志组 {#cloudwatch日志组}
resource "aws_cloudwatch_log_group" "vivtransformer_logs" {
  name              = "/ecs/vivtransformer"
  retention_in_days = 7
}

# 输出 {#输出}
output "load_balancer_dns" {
  value = aws_lb.vivtransformer_alb.dns_name
}

output "ecr_repository_url" {
  value = aws_ecr_repository.vivtransformer_repo.repository_url
}
```

## 边缘设备部署 {#边缘设备部署}

### 📱 移动设备部署 {#移动设备部署}

```python
import torch
import numpy as np
from typing import Dict, List, Optional
import json
import os

class EdgeDeploymentOptimizer:
    """边缘设备部署优化器"""
    
    def __init__(self, target_device: str = "mobile"):
        self.target_device = target_device
        self.optimization_config = self._get_device_config()
    
    def _get_device_config(self) -> Dict:
        """获取设备配置"""
        configs = {
            "mobile": {
                "max_memory_mb": 512,
                "max_model_size_mb": 50,
                "target_latency_ms": 100,
                "quantization": "int8",
                "pruning_ratio": 0.3
            },
            "raspberry_pi": {
                "max_memory_mb": 1024,
                "max_model_size_mb": 100,
                "target_latency_ms": 200,
                "quantization": "int8",
                "pruning_ratio": 0.2
            },
            "jetson_nano": {
                "max_memory_mb": 2048,
                "max_model_size_mb": 200,
                "target_latency_ms": 50,
                "quantization": "fp16",
                "pruning_ratio": 0.1
            }
        }
        return configs.get(self.target_device, configs["mobile"])
    
    def optimize_model(self, model_path: str, output_path: str) -> Dict[str, str]:
        """优化模型用于边缘部署"""
        print(f"🔧 开始为 {self.target_device} 优化模型...")
        
        # 加载模型
        model = torch.load(model_path, map_location='cpu')
        model.eval()
        
        optimized_models = {}
        
        # 1. 模型剪枝
        if self.optimization_config["pruning_ratio"] > 0:
            pruned_model = self._prune_model(model)
            pruned_path = output_path.replace('.pt', '_pruned.pt')
            torch.save(pruned_model, pruned_path)
            optimized_models['pruned'] = pruned_path
            print(f"✅ 模型剪枝完成: {pruned_path}")
        
        # 2. 模型量化
        quantized_model = self._quantize_model(model)
        quantized_path = output_path.replace('.pt', '_quantized.pt')
        torch.save(quantized_model, quantized_path)
        optimized_models['quantized'] = quantized_path
        print(f"✅ 模型量化完成: {quantized_path}")
        
        # 3. 导出为移动端格式
        if self.target_device == "mobile":
            mobile_path = self._export_mobile_model(model, output_path)
            optimized_models['mobile'] = mobile_path
            print(f"✅ 移动端模型导出完成: {mobile_path}")
        
        return optimized_models
    
    def _prune_model(self, model):
        """模型剪枝"""
        import torch.nn.utils.prune as prune
        
        # 结构化剪枝
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                prune.l1_unstructured(
                    module, 
                    name='weight', 
                    amount=self.optimization_config["pruning_ratio"]
                )
        
        return model
    
    def _quantize_model(self, model):
        """模型量化"""
        quantization_type = self.optimization_config["quantization"]
        
        if quantization_type == "int8":
            # 动态量化
            quantized_model = torch.quantization.quantize_dynamic(
                model,
                {torch.nn.Linear},
                dtype=torch.qint8
            )
        elif quantization_type == "fp16":
            # 半精度
            quantized_model = model.half()
        else:
            quantized_model = model
        
        return quantized_model
    
    def _export_mobile_model(self, model, output_path: str) -> str:
        """导出移动端模型"""
        # 创建示例输入
        dummy_input = torch.randint(0, 1000, (1, 512), dtype=torch.long)
        
        # 追踪模型
        traced_model = torch.jit.trace(model, dummy_input)
        
        # 优化移动端
        mobile_model = torch.utils.mobile_optimizer.optimize_for_mobile(traced_model)
        
        # 保存
        mobile_path = output_path.replace('.pt', '_mobile.ptl')
        mobile_model._save_for_lite_interpreter(mobile_path)
        
        return mobile_path

class EdgeInferenceEngine:
    """边缘推理引擎"""
    
    def __init__(self, model_path: str, device: str = "cpu"):
        self.model_path = model_path
        self.device = device
        self.model = None
        self.preprocessor = None
        self.load_model()
    
    def load_model(self):
        """加载模型"""
        if self.model_path.endswith('.ptl'):
            # 移动端模型
            self.model = torch.jit.load(self.model_path)
        elif self.model_path.endswith('.onnx'):
            # ONNX模型
            import onnxruntime as ort
            self.model = ort.InferenceSession(self.model_path)
        else:
            # PyTorch模型
            self.model = torch.load(self.model_path, map_location=self.device)
        
        self.model.eval()
        print(f"✅ 模型加载完成: {self.model_path}")
    
    def predict(self, input_data: np.ndarray) -> Dict:
        """执行推理"""
        import time
        
        start_time = time.time()
        
        # 预处理
        processed_input = self._preprocess(input_data)
        
        # 推理
        if isinstance(self.model, torch.jit.ScriptModule):
            # TorchScript模型
            with torch.no_grad():
                output = self.model(processed_input)
        elif hasattr(self.model, 'run'):
            # ONNX模型
            input_name = self.model.get_inputs()[0].name
            output = self.model.run(None, {input_name: processed_input.numpy()})[0]
            output = torch.from_numpy(output)
        else:
            # 普通PyTorch模型
            with torch.no_grad():
                output = self.model(processed_input)
        
        # 后处理
        result = self._postprocess(output)
        
        inference_time = (time.time() - start_time) * 1000  # 毫秒
        
        return {
            'result': result,
            'inference_time_ms': inference_time,
            'model_type': type(self.model).__name__
        }
    
    def _preprocess(self, input_data: np.ndarray) -> torch.Tensor:
        """预处理"""
        # 转换为张量
        if isinstance(input_data, np.ndarray):
            tensor = torch.from_numpy(input_data)
        else:
            tensor = torch.tensor(input_data)
        
        # 确保数据类型
        if tensor.dtype != torch.long:
            tensor = tensor.long()
        
        # 添加批次维度
        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(0)
        
        return tensor.to(self.device)
    
    def _postprocess(self, output: torch.Tensor) -> Dict:
        """后处理"""
        # 转换为numpy
        if isinstance(output, torch.Tensor):
            output = output.cpu().numpy()
        
        # 应用softmax获取概率
        if output.ndim == 2:
            probabilities = torch.softmax(torch.from_numpy(output), dim=-1).numpy()
            predicted_class = np.argmax(probabilities, axis=-1)
            confidence = np.max(probabilities, axis=-1)
        else:
            probabilities = output
            predicted_class = np.argmax(output)
            confidence = np.max(output)
        
        return {
            'predicted_class': int(predicted_class[0]) if isinstance(predicted_class, np.ndarray) else int(predicted_class),
            'confidence': float(confidence[0]) if isinstance(confidence, np.ndarray) else float(confidence),
            'probabilities': probabilities.tolist()
        }
```

### 🔌 IoT设备集成 {#iot设备集成}

```python
class IoTDeploymentManager:
    """IoT设备部署管理器"""
    
    def __init__(self, device_config: Dict):
        self.config = device_config
        self.inference_engine = None
        self.mqtt_client = None
        self.setup_device()
    
    def setup_device(self):
        """设置设备"""
        print(f"🔧 设置IoT设备: {self.config.get('device_id', 'unknown')}")
        
        # 初始化推理引擎
        model_path = self.config.get('model_path')
        if model_path and os.path.exists(model_path):
            self.inference_engine = EdgeInferenceEngine(
                model_path, 
                self.config.get('device', 'cpu')
            )
        
        # 设置MQTT客户端（如果需要）
        if self.config.get('enable_mqtt', False):
            self.setup_mqtt()
    
    def setup_mqtt(self):
        """设置MQTT连接"""
        try:
            import paho.mqtt.client as mqtt
            
            self.mqtt_client = mqtt.Client()
            self.mqtt_client.on_connect = self._on_mqtt_connect
            self.mqtt_client.on_message = self._on_mqtt_message
            
            # 连接MQTT代理
            broker = self.config.get('mqtt_broker', 'localhost')
            port = self.config.get('mqtt_port', 1883)
            self.mqtt_client.connect(broker, port, 60)
            self.mqtt_client.loop_start()
            
            print(f"✅ MQTT连接成功: {broker}:{port}")
            
        except ImportError:
            print("⚠️ 未安装paho-mqtt，跳过MQTT设置")
        except Exception as e:
            print(f"❌ MQTT连接失败: {e}")
    
    def _on_mqtt_connect(self, client, userdata, flags, rc):
        """MQTT连接回调"""
        if rc == 0:
            print("✅ MQTT连接成功")
            # 订阅推理请求主题
            topic = f"vivtransformer/{self.config.get('device_id')}/inference"
            client.subscribe(topic)
        else:
            print(f"❌ MQTT连接失败，代码: {rc}")
    
    def _on_mqtt_message(self, client, userdata, msg):
        """MQTT消息回调"""
        try:
            # 解析消息
            message = json.loads(msg.payload.decode())
            input_data = np.array(message.get('input_data', []))
            request_id = message.get('request_id', 'unknown')
            
            # 执行推理
            if self.inference_engine:
                result = self.inference_engine.predict(input_data)
                
                # 发布结果
                response_topic = f"vivtransformer/{self.config.get('device_id')}/result"
                response = {
                    'request_id': request_id,
                    'result': result,
                    'device_id': self.config.get('device_id'),
                    'timestamp': time.time()
                }
                
                client.publish(response_topic, json.dumps(response))
                print(f"📤 推理结果已发送: {request_id}")
            
        except Exception as e:
            print(f"❌ 处理MQTT消息失败: {e}")
    
    def run_local_inference(self, input_data: np.ndarray) -> Dict:
        """运行本地推理"""
        if not self.inference_engine:
            raise RuntimeError("推理引擎未初始化")
        
        return self.inference_engine.predict(input_data)
    
    def start_monitoring(self):
        """开始监控"""
        import psutil
        import time
        
        print("📊 开始系统监控...")
        
        while True:
            try:
                # 获取系统信息
                cpu_percent = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                disk = psutil.disk_usage('/')
                
                # 获取温度（如果支持）
                temperature = self._get_temperature()
                
                # 监控信息
                monitor_data = {
                    'device_id': self.config.get('device_id'),
                    'timestamp': time.time(),
                    'cpu_percent': cpu_percent,
                    'memory_percent': memory.percent,
                    'memory_available_mb': memory.available // 1024 // 1024,
                    'disk_percent': disk.percent,
                    'temperature_c': temperature
                }
                
                # 发送监控数据
                if self.mqtt_client:
                    monitor_topic = f"vivtransformer/{self.config.get('device_id')}/monitor"
                    self.mqtt_client.publish(monitor_topic, json.dumps(monitor_data))
                
                # 检查资源使用情况
                if cpu_percent > 90:
                    print(f"⚠️ CPU使用率过高: {cpu_percent}%")
                
                if memory.percent > 90:
                    print(f"⚠️ 内存使用率过高: {memory.percent}%")
                
                if temperature and temperature > 80:
                    print(f"⚠️ 设备温度过高: {temperature}°C")
                
                time.sleep(30)  # 30秒监控一次
                
            except KeyboardInterrupt:
                print("\n🛑 监控已停止")
                break
            except Exception as e:
                print(f"❌ 监控错误: {e}")
                time.sleep(5)
    
    def _get_temperature(self) -> Optional[float]:
        """获取设备温度"""
        try:
            # 尝试读取树莓派温度
            if os.path.exists('/sys/class/thermal/thermal_zone0/temp'):
                with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
                    temp = int(f.read().strip()) / 1000.0
                    return temp
        except:
            pass
        
        return None
```

## 云端部署 {#云端部署}

### ☁️ AWS部署 {#aws部署}

```yaml
# aws-deployment.yml {#aws-deployment-yml}
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vivtransformer-api
  namespace: default
spec:
  replicas: 3
  selector:
    matchLabels:
      app: vivtransformer-api
  template:
    metadata:
      labels:
        app: vivtransformer-api
    spec:
      containers:
      - name: vivtransformer
        image: your-registry/vivtransformer:latest
        ports:
        - containerPort: 8000
        env:
        - name: MODEL_PATH
          value: "/app/models/vivtransformer.pt"
        - name: DEVICE
          value: "cuda"
        - name: BATCH_SIZE
          value: "32"
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
            nvidia.com/gpu: 1
          limits:
            memory: "4Gi"
            cpu: "2000m"
            nvidia.com/gpu: 1
        volumeMounts:
        - name: model-storage
          mountPath: /app/models
      volumes:
      - name: model-storage
        persistentVolumeClaim:
          claimName: model-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: vivtransformer-service
spec:
  selector:
    app: vivtransformer-api
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer
```

```python
# aws_deployment.py {#aws-deployment-py}
import boto3
import json
from typing import Dict, List

class AWSDeploymentManager:
    """AWS部署管理器"""
    
    def __init__(self, region: str = 'us-west-2'):
        self.region = region
        self.ecs_client = boto3.client('ecs', region_name=region)
        self.ecr_client = boto3.client('ecr', region_name=region)
        self.s3_client = boto3.client('s3', region_name=region)
        self.cloudwatch = boto3.client('cloudwatch', region_name=region)
    
    def create_ecr_repository(self, repo_name: str) -> str:
        """创建ECR仓库"""
        try:
            response = self.ecr_client.create_repository(
                repositoryName=repo_name,
                imageScanningConfiguration={'scanOnPush': True}
            )
            repo_uri = response['repository']['repositoryUri']
            print(f"✅ ECR仓库创建成功: {repo_uri}")
            return repo_uri
        except self.ecr_client.exceptions.RepositoryAlreadyExistsException:
            response = self.ecr_client.describe_repositories(
                repositoryNames=[repo_name]
            )
            repo_uri = response['repositories'][0]['repositoryUri']
            print(f"ℹ️ ECR仓库已存在: {repo_uri}")
            return repo_uri
    
    def upload_model_to_s3(self, model_path: str, bucket: str, key: str):
        """上传模型到S3"""
        try:
            self.s3_client.upload_file(model_path, bucket, key)
            print(f"✅ 模型上传成功: s3://{bucket}/{key}")
        except Exception as e:
            print(f"❌ 模型上传失败: {e}")
            raise
    
    def create_ecs_cluster(self, cluster_name: str) -> str:
        """创建ECS集群"""
        try:
            response = self.ecs_client.create_cluster(
                clusterName=cluster_name,
                capacityProviders=['FARGATE', 'EC2'],
                defaultCapacityProviderStrategy=[
                    {
                        'capacityProvider': 'FARGATE',
                        'weight': 1
                    }
                ]
            )
            cluster_arn = response['cluster']['clusterArn']
            print(f"✅ ECS集群创建成功: {cluster_arn}")
            return cluster_arn
        except Exception as e:
            print(f"❌ ECS集群创建失败: {e}")
            raise
    
    def create_task_definition(self, 
                             family: str, 
                             image_uri: str, 
                             cpu: str = '1024', 
                             memory: str = '2048') -> str:
        """创建任务定义"""
        task_definition = {
            'family': family,
            'networkMode': 'awsvpc',
            'requiresCompatibilities': ['FARGATE'],
            'cpu': cpu,
            'memory': memory,
            'executionRoleArn': 'arn:aws:iam::YOUR_ACCOUNT:role/ecsTaskExecutionRole',
            'containerDefinitions': [
                {
                    'name': 'vivtransformer',
                    'image': image_uri,
                    'portMappings': [
                        {
                            'containerPort': 8000,
                            'protocol': 'tcp'
                        }
                    ],
                    'environment': [
                        {'name': 'MODEL_PATH', 'value': '/app/models/vivtransformer.pt'},
                        {'name': 'DEVICE', 'value': 'cpu'},
                        {'name': 'BATCH_SIZE', 'value': '16'}
                    ],
                    'logConfiguration': {
                        'logDriver': 'awslogs',
                        'options': {
                            'awslogs-group': f'/ecs/{family}',
                            'awslogs-region': self.region,
                            'awslogs-stream-prefix': 'ecs'
                        }
                    }
                }
            ]
        }
        
        try:
            response = self.ecs_client.register_task_definition(**task_definition)
            task_def_arn = response['taskDefinition']['taskDefinitionArn']
            print(f"✅ 任务定义创建成功: {task_def_arn}")
            return task_def_arn
        except Exception as e:
            print(f"❌ 任务定义创建失败: {e}")
            raise
    
    def create_service(self, 
                      cluster_name: str, 
                      service_name: str, 
                      task_definition: str,
                      desired_count: int = 2) -> str:
        """创建ECS服务"""
        try:
            response = self.ecs_client.create_service(
                cluster=cluster_name,
                serviceName=service_name,
                taskDefinition=task_definition,
                desiredCount=desired_count,
                launchType='FARGATE',
                networkConfiguration={
                    'awsvpcConfiguration': {
                        'subnets': ['subnet-12345', 'subnet-67890'],  # 替换为实际子网ID
                        'securityGroups': ['sg-12345'],  # 替换为实际安全组ID
                        'assignPublicIp': 'ENABLED'
                    }
                },
                loadBalancers=[
                    {
                        'targetGroupArn': 'arn:aws:elasticloadbalancing:region:account:targetgroup/name',
                        'containerName': 'vivtransformer',
                        'containerPort': 8000
                    }
                ]
            )
            service_arn = response['service']['serviceArn']
            print(f"✅ ECS服务创建成功: {service_arn}")
            return service_arn
        except Exception as e:
            print(f"❌ ECS服务创建失败: {e}")
            raise
    
    def setup_monitoring(self, cluster_name: str, service_name: str):
        """设置CloudWatch监控"""
        # 创建自定义指标
        metrics = [
            {
                'MetricName': 'InferenceLatency',
                'Namespace': 'VIVTransformer/Performance',
                'Dimensions': [
                    {'Name': 'Cluster', 'Value': cluster_name},
                    {'Name': 'Service', 'Value': service_name}
                ]
            },
            {
                'MetricName': 'InferenceCount',
                'Namespace': 'VIVTransformer/Usage',
                'Dimensions': [
                    {'Name': 'Cluster', 'Value': cluster_name},
                    {'Name': 'Service', 'Value': service_name}
                ]
            }
        ]
        
        # 创建告警
        alarms = [
            {
                'AlarmName': f'{service_name}-HighLatency',
                'ComparisonOperator': 'GreaterThanThreshold',
                'EvaluationPeriods': 2,
                'MetricName': 'InferenceLatency',
                'Namespace': 'VIVTransformer/Performance',
                'Period': 300,
                'Statistic': 'Average',
                'Threshold': 1000.0,
                'ActionsEnabled': True,
                'AlarmActions': ['arn:aws:sns:region:account:topic-name'],
                'AlarmDescription': '推理延迟过高告警',
                'Dimensions': [
                    {'Name': 'Service', 'Value': service_name}
                ]
            }
        ]
        
        for alarm in alarms:
            try:
                self.cloudwatch.put_metric_alarm(**alarm)
                print(f"✅ 告警创建成功: {alarm['AlarmName']}")
            except Exception as e:
                print(f"❌ 告警创建失败: {e}")
```

### 🔄 自动扩缩容 {#自动扩缩容}

```python
class AutoScalingManager:
    """自动扩缩容管理器"""
    
    def __init__(self, region: str = 'us-west-2'):
        self.autoscaling_client = boto3.client('application-autoscaling', region_name=region)
        self.cloudwatch = boto3.client('cloudwatch', region_name=region)
    
    def setup_autoscaling(self, 
                         cluster_name: str, 
                         service_name: str,
                         min_capacity: int = 1,
                         max_capacity: int = 10):
        """设置自动扩缩容"""
        resource_id = f'service/{cluster_name}/{service_name}'
        
        # 注册可扩展目标
        try:
            self.autoscaling_client.register_scalable_target(
                ServiceNamespace='ecs',
                ResourceId=resource_id,
                ScalableDimension='ecs:service:DesiredCount',
                MinCapacity=min_capacity,
                MaxCapacity=max_capacity
            )
            print(f"✅ 可扩展目标注册成功: {resource_id}")
        except Exception as e:
            print(f"❌ 可扩展目标注册失败: {e}")
            return
        
        # 创建扩缩容策略
        policies = [
            {
                'PolicyName': f'{service_name}-scale-up',
                'PolicyType': 'TargetTrackingScaling',
                'TargetTrackingScalingPolicyConfiguration': {
                    'TargetValue': 70.0,
                    'PredefinedMetricSpecification': {
                        'PredefinedMetricType': 'ECSServiceAverageCPUUtilization'
                    },
                    'ScaleOutCooldown': 300,
                    'ScaleInCooldown': 300
                }
            },
            {
                'PolicyName': f'{service_name}-scale-memory',
                'PolicyType': 'TargetTrackingScaling',
                'TargetTrackingScalingPolicyConfiguration': {
                    'TargetValue': 80.0,
                    'PredefinedMetricSpecification': {
                        'PredefinedMetricType': 'ECSServiceAverageMemoryUtilization'
                    },
                    'ScaleOutCooldown': 300,
                    'ScaleInCooldown': 300
                }
            }
        ]
        
        for policy in policies:
            try:
                self.autoscaling_client.put_scaling_policy(
                    ServiceNamespace='ecs',
                    ResourceId=resource_id,
                    ScalableDimension='ecs:service:DesiredCount',
                    **policy
                )
                print(f"✅ 扩缩容策略创建成功: {policy['PolicyName']}")
            except Exception as e:
                print(f"❌ 扩缩容策略创建失败: {e}")
```

## 监控与维护 {#监控与维护}

### 📊 性能监控 {#性能监控}

```python
import time
import psutil
import GPUtil
from prometheus_client import Counter, Histogram, Gauge, start_http_server

class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self, port: int = 8001):
        # Prometheus指标
        self.inference_counter = Counter(
            'vivtransformer_inference_total', 
            'Total number of inferences',
            ['model_version', 'status']
        )
        
        self.inference_latency = Histogram(
            'vivtransformer_inference_duration_seconds',
            'Inference latency in seconds',
            ['model_version']
        )
        
        self.system_cpu = Gauge(
            'vivtransformer_cpu_usage_percent',
            'CPU usage percentage'
        )
        
        self.system_memory = Gauge(
            'vivtransformer_memory_usage_percent',
            'Memory usage percentage'
        )
        
        self.gpu_utilization = Gauge(
            'vivtransformer_gpu_utilization_percent',
            'GPU utilization percentage',
            ['gpu_id']
        )
        
        self.gpu_memory = Gauge(
            'vivtransformer_gpu_memory_usage_percent',
            'GPU memory usage percentage',
            ['gpu_id']
        )
        
        # 启动Prometheus服务器
        start_http_server(port)
        print(f"📊 Prometheus监控服务启动: http://localhost:{port}")
    
    def record_inference(self, model_version: str, latency: float, status: str = 'success'):
        """记录推理指标"""
        self.inference_counter.labels(
            model_version=model_version, 
            status=status
        ).inc()
        
        if status == 'success':
            self.inference_latency.labels(
                model_version=model_version
            ).observe(latency)
    
    def update_system_metrics(self):
        """更新系统指标"""
        # CPU使用率
        cpu_percent = psutil.cpu_percent(interval=1)
        self.system_cpu.set(cpu_percent)
        
        # 内存使用率
        memory = psutil.virtual_memory()
        self.system_memory.set(memory.percent)
        
        # GPU指标
        try:
            gpus = GPUtil.getGPUs()
            for i, gpu in enumerate(gpus):
                self.gpu_utilization.labels(gpu_id=str(i)).set(gpu.load * 100)
                self.gpu_memory.labels(gpu_id=str(i)).set(gpu.memoryUtil * 100)
        except:
            pass  # 没有GPU或GPUtil不可用
    
    def start_monitoring(self, interval: int = 30):
        """开始监控"""
        print(f"🔄 开始系统监控，间隔: {interval}秒")
        
        while True:
            try:
                self.update_system_metrics()
                time.sleep(interval)
            except KeyboardInterrupt:
                print("\n🛑 监控已停止")
                break
            except Exception as e:
                print(f"❌ 监控错误: {e}")
                time.sleep(5)

class HealthChecker:
    """健康检查器"""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.last_check_time = 0
        self.check_interval = 60  # 60秒检查一次
    
    def check_model_health(self) -> Dict[str, bool]:
        """检查模型健康状态"""
        health_status = {
            'model_file_exists': os.path.exists(self.model_path),
            'model_loadable': False,
            'inference_working': False,
            'memory_sufficient': True,
            'disk_space_sufficient': True
        }
        
        # 检查模型是否可加载
        try:
            import torch
            model = torch.load(self.model_path, map_location='cpu')
            health_status['model_loadable'] = True
            
            # 检查推理是否正常
            dummy_input = torch.randint(0, 1000, (1, 512), dtype=torch.long)
            with torch.no_grad():
                output = model(dummy_input)
                health_status['inference_working'] = True
        except Exception as e:
            print(f"❌ 模型健康检查失败: {e}")
        
        # 检查内存
        memory = psutil.virtual_memory()
        if memory.percent > 90:
            health_status['memory_sufficient'] = False
        
        # 检查磁盘空间
        disk = psutil.disk_usage('/')
        if disk.percent > 90:
            health_status['disk_space_sufficient'] = False
        
        return health_status
    
    def get_health_report(self) -> Dict:
        """获取健康报告"""
        current_time = time.time()
        
        # 如果距离上次检查时间超过间隔，重新检查
        if current_time - self.last_check_time > self.check_interval:
            self.health_status = self.check_model_health()
            self.last_check_time = current_time
        
        return {
             'timestamp': current_time,
             'status': 'healthy' if all(self.health_status.values()) else 'unhealthy',
             'checks': self.health_status,
             'uptime': current_time - self.last_check_time
         }
```

### 🔧 日志管理 {#日志管理}

```python
import logging
import json
from datetime import datetime
from typing import Dict, Any

class StructuredLogger:
    """结构化日志记录器"""
    
    def __init__(self, name: str = 'vivtransformer', level: str = 'INFO'):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.upper()))
        
        # 创建格式化器
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # 控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # 文件处理器
        file_handler = logging.FileHandler('vivtransformer.log')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
    
    def log_inference(self, 
                     request_id: str, 
                     input_shape: tuple, 
                     latency: float, 
                     status: str,
                     error: str = None):
        """记录推理日志"""
        log_data = {
            'event_type': 'inference',
            'request_id': request_id,
            'timestamp': datetime.utcnow().isoformat(),
            'input_shape': input_shape,
            'latency_ms': latency * 1000,
            'status': status
        }
        
        if error:
            log_data['error'] = error
            self.logger.error(json.dumps(log_data))
        else:
            self.logger.info(json.dumps(log_data))
    
    def log_system_event(self, event_type: str, data: Dict[str, Any]):
        """记录系统事件"""
        log_data = {
            'event_type': event_type,
            'timestamp': datetime.utcnow().isoformat(),
            **data
        }
        
        self.logger.info(json.dumps(log_data))
```

## 故障排除 {#故障排除}

### 🚨 常见问题 {#常见问题}

#### 1. 内存不足错误 {#1-内存不足错误}

**症状：**
- `CUDA out of memory` 错误
- `RuntimeError: out of memory` 错误
- 系统响应缓慢

**解决方案：**

```python
# 内存优化配置 {#内存优化配置}
optimization_config = {
    "batch_size": 8,  # 减小批次大小
    "gradient_accumulation_steps": 4,  # 使用梯度累积
    "mixed_precision": True,  # 启用混合精度
    "gradient_checkpointing": True,  # 启用梯度检查点
    "dataloader_num_workers": 2,  # 减少数据加载器工作进程
}

# 清理GPU内存 {#清理gpu内存}
import torch
torch.cuda.empty_cache()

# 监控内存使用 {#监控内存使用}
def monitor_memory():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        cached = torch.cuda.memory_reserved() / 1024**3
        print(f"GPU内存 - 已分配: {allocated:.2f}GB, 已缓存: {cached:.2f}GB")
```

#### 2. 模型加载失败 {#2-模型加载失败}

**症状：**
- `FileNotFoundError` 错误
- `RuntimeError: Error(s) in loading state_dict` 错误
- 模型权重不匹配

**解决方案：**

```python
def safe_model_loading(model_path: str, device: str = 'cpu'):
    """安全的模型加载"""
    try:
        # 检查文件是否存在
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        
        # 检查文件大小
        file_size = os.path.getsize(model_path) / 1024**2  # MB
        if file_size < 1:
            raise ValueError(f"模型文件过小，可能已损坏: {file_size:.2f}MB")
        
        # 尝试加载模型
        checkpoint = torch.load(model_path, map_location=device)
        
        # 检查检查点结构
        if 'model_state_dict' in checkpoint:
            model_state = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            model_state = checkpoint['state_dict']
        else:
            model_state = checkpoint
        
        print(f"✅ 模型加载成功: {model_path}")
        return model_state
        
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        raise
```

#### 3. 推理速度慢 {#3-推理速度慢}

**症状：**
- 推理延迟过高
- 吞吐量低
- CPU/GPU利用率低

**解决方案：**

```python
class InferenceOptimizer:
    """推理优化器"""
    
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        self.optimize_model()
    
    def optimize_model(self):
        """优化模型"""
        # 设置为评估模式
        self.model.eval()
        
        # 禁用梯度计算
        for param in self.model.parameters():
            param.requires_grad = False
        
        # 使用TorchScript优化
        if hasattr(torch.jit, 'optimize_for_inference'):
            self.model = torch.jit.optimize_for_inference(self.model)
        
        # 预热模型
        self.warmup()
    
    def warmup(self, num_warmup: int = 10):
        """模型预热"""
        dummy_input = torch.randint(0, 1000, (1, 512), dtype=torch.long).to(self.device)
        
        with torch.no_grad():
            for _ in range(num_warmup):
                _ = self.model(dummy_input)
        
        print(f"✅ 模型预热完成: {num_warmup}次")
    
    def batch_inference(self, inputs: List[torch.Tensor], batch_size: int = 32):
        """批量推理"""
        results = []
        
        for i in range(0, len(inputs), batch_size):
            batch = inputs[i:i + batch_size]
            batch_tensor = torch.stack(batch).to(self.device)
            
            with torch.no_grad():
                output = self.model(batch_tensor)
                results.extend(output.cpu().numpy())
        
        return results
```

#### 4. 网络连接问题 {#4-网络连接问题}

**症状：**
- API请求超时
- 连接被拒绝
- 间歇性连接失败

**解决方案：**

```python
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

class RobustAPIClient:
    """健壮的API客户端"""
    
    def __init__(self, base_url: str, timeout: int = 30):
        self.base_url = base_url
        self.timeout = timeout
        self.session = self._create_session()
    
    def _create_session(self):
        """创建会话"""
        session = requests.Session()
        
        # 重试策略
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            method_whitelist=["HEAD", "GET", "POST"]
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        return session
    
    def predict(self, data: Dict) -> Dict:
        """执行预测"""
        try:
            response = self.session.post(
                f"{self.base_url}/predict",
                json=data,
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"❌ API请求失败: {e}")
            raise
```

### 🔍 调试工具 {#调试工具}

```python
class DeploymentDebugger:
    """部署调试器"""
    
    def __init__(self):
        self.checks = []
    
    def check_environment(self) -> Dict[str, Any]:
        """检查环境"""
        import platform
        import sys
        
        env_info = {
            'python_version': sys.version,
            'platform': platform.platform(),
            'pytorch_version': torch.__version__ if 'torch' in globals() else 'Not installed',
            'cuda_available': torch.cuda.is_available() if 'torch' in globals() else False,
            'gpu_count': torch.cuda.device_count() if 'torch' in globals() and torch.cuda.is_available() else 0
        }
        
        return env_info
    
    def check_model_compatibility(self, model_path: str) -> Dict[str, Any]:
        """检查模型兼容性"""
        try:
            checkpoint = torch.load(model_path, map_location='cpu')
            
            compatibility_info = {
                'file_size_mb': os.path.getsize(model_path) / 1024**2,
                'pytorch_version': checkpoint.get('pytorch_version', 'Unknown'),
                'model_keys': list(checkpoint.keys()) if isinstance(checkpoint, dict) else ['state_dict'],
                'loadable': True
            }
            
            return compatibility_info
            
        except Exception as e:
            return {
                'loadable': False,
                'error': str(e)
            }
    
    def run_diagnostics(self, model_path: str = None) -> Dict[str, Any]:
        """运行诊断"""
        diagnostics = {
            'timestamp': datetime.utcnow().isoformat(),
            'environment': self.check_environment(),
            'system_resources': {
                'cpu_count': psutil.cpu_count(),
                'memory_gb': psutil.virtual_memory().total / 1024**3,
                'disk_free_gb': psutil.disk_usage('/').free / 1024**3
            }
        }
        
        if model_path:
            diagnostics['model_compatibility'] = self.check_model_compatibility(model_path)
        
        return diagnostics
```

## 最佳实践 {#最佳实践}

### 📋 部署检查清单 {#部署检查清单}

#### 部署前检查 {#部署前检查}
- [ ] 模型文件完整性验证
- [ ] 依赖项版本兼容性检查
- [ ] 系统资源需求评估
- [ ] 安全配置审查
- [ ] 备份策略制定

#### 部署过程检查 {#部署过程检查}
- [ ] 渐进式部署（蓝绿部署）
- [ ] 健康检查配置
- [ ] 监控指标设置
- [ ] 日志记录配置
- [ ] 回滚计划准备

#### 部署后检查 {#部署后检查}
- [ ] 功能测试验证
- [ ] 性能基准测试
- [ ] 监控告警测试
- [ ] 文档更新
- [ ] 团队培训

### 🔒 安全最佳实践 {#安全最佳实践}

```python
class SecurityManager:
    """安全管理器"""
    
    def __init__(self):
        self.api_keys = {}
        self.rate_limits = {}
    
    def generate_api_key(self, user_id: str) -> str:
        """生成API密钥"""
        import secrets
        import hashlib
        
        # 生成随机密钥
        raw_key = secrets.token_urlsafe(32)
        
        # 创建哈希
        key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
        
        # 存储（实际应用中应存储到数据库）
        self.api_keys[key_hash] = {
            'user_id': user_id,
            'created_at': datetime.utcnow(),
            'active': True
        }
        
        return raw_key
    
    def validate_api_key(self, api_key: str) -> bool:
        """验证API密钥"""
        import hashlib
        
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        
        if key_hash in self.api_keys:
            return self.api_keys[key_hash]['active']
        
        return False
    
    def check_rate_limit(self, user_id: str, limit: int = 100) -> bool:
        """检查速率限制"""
        current_time = time.time()
        
        if user_id not in self.rate_limits:
            self.rate_limits[user_id] = []
        
        # 清理过期记录（1小时窗口）
        self.rate_limits[user_id] = [
            timestamp for timestamp in self.rate_limits[user_id]
            if current_time - timestamp < 3600
        ]
        
        # 检查是否超过限制
        if len(self.rate_limits[user_id]) >= limit:
            return False
        
        # 记录当前请求
        self.rate_limits[user_id].append(current_time)
        return True
```

### 📈 性能优化建议 {#性能优化建议}

1. **模型优化**
   - 使用模型量化减少内存占用
   - 启用混合精度训练和推理
   - 考虑模型剪枝和知识蒸馏

2. **系统优化**
   - 使用SSD存储提高I/O性能
   - 配置适当的批次大小
   - 启用GPU内存池

3. **网络优化**
   - 使用CDN加速模型下载
   - 启用HTTP/2和压缩
   - 配置适当的超时设置

4. **缓存策略**
   - 实现模型预测结果缓存
   - 使用Redis进行分布式缓存
   - 配置适当的缓存过期策略

## 总结 {#总结}

本部署指南涵盖了VIVTransformer模型从开发到生产的完整部署流程，包括：

### 🎯 核心要点 {#核心要点}

1. **多环境支持**：提供了本地、容器化、边缘设备和云端的完整部署方案
2. **自动化部署**：通过Docker、Kubernetes和云服务实现自动化部署
3. **监控体系**：建立了完善的性能监控和健康检查机制
4. **故障处理**：提供了常见问题的诊断和解决方案
5. **安全保障**：实现了API认证、速率限制等安全措施

### 🔗 相关链接 {#相关链接}

- [模型设计](model-design.html) - 了解模型架构
- [训练指南](training-guide.html) - 模型训练方法
- [性能比较](performance-comparison.html) - 性能基准测试
- [故障排除](troubleshooting.html) - 详细故障排除指南
- [开发指南](development-guide.html) - 开发环境配置

### 📞 支持 {#支持}

如果在部署过程中遇到问题，请：
1. 查阅[故障排除指南](troubleshooting.html)
2. 检查[FAQ](faq.html)中的常见问题
3. 在GitHub Issues中提交问题报告

---

*最后更新：2024年1月*

## 📚 相关文档

- [Performance Comparison](performance-comparison.html)
- [Troubleshooting](troubleshooting.html)
- [Configuration System](configuration-system.html)


---

*需要帮助？查看 [FAQ](faq.html) 或 [故障排除](troubleshooting.html) 页面。*
