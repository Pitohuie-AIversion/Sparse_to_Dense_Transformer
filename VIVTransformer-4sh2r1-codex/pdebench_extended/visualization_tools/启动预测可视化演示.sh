#!/bin/bash

# 预测可视化演示启动脚本 (Linux版本)
# Prediction Visualization Demo Launcher for Linux

echo "========================================"
echo "启动预测可视化演示 (Linux版本)"
echo "Prediction Visualization Demo (Linux)"
echo "========================================"
echo ""
echo "功能说明 / Features:"
echo "- 单样本预测对比 / Single sample prediction comparison"
echo "- 批次样本可视化 / Batch samples visualization"
echo "- 误差分析 / Error analysis"
echo "- 训练进度跟踪 / Training progress tracking"
echo "- 自动报告生成 / Automatic report generation"
echo ""

# 设置环境变量避免OpenMP冲突
export KMP_DUPLICATE_LIB_OK=TRUE
export OMP_NUM_THREADS=1

# 检查Python环境
if ! command -v python3 &> /dev/null; then
    echo "错误: 未找到Python3，请先安装Python3"
    echo "Error: Python3 not found, please install Python3 first"
    exit 1
fi

# 检查必要的Python包
echo "检查Python环境..."
echo "Checking Python environment..."

python3 -c "import torch, matplotlib, numpy, seaborn" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "警告: 某些必要的Python包可能未安装"
    echo "Warning: Some required Python packages may not be installed"
    echo "请确保已安装: torch, matplotlib, numpy, seaborn"
    echo "Please ensure installed: torch, matplotlib, numpy, seaborn"
    echo ""
fi

# 运行预测可视化演示
echo "开始运行预测可视化演示..."
echo "Starting prediction visualization demo..."
echo ""

python3 create_prediction_visualization.py

# 检查执行结果
if [ $? -eq 0 ]; then
    echo ""
    echo "========================================"
    echo "Prediction visualization demo completed successfully!"
    echo "========================================"
    echo ""
    echo "Generated visualizations can be found in:"
    echo "- demo_prediction_visualizations/single_samples/     (Individual predictions)"
    echo "- demo_prediction_visualizations/batch_comparisons/  (Batch predictions)"
    echo "- demo_prediction_visualizations/error_analysis/     (Error analysis)"
    echo "- demo_prediction_visualizations/time_series/        (Training progress)"
    echo ""
    echo "Check the visualization report:"
    echo "- demo_prediction_visualizations/epoch_10_visualization_report.md"
    echo ""
else
    echo ""
    echo "========================================"
    echo "Error occurred during prediction visualization demo!"
    echo "========================================"
    echo ""
    echo "Please check:"
    echo "1. Python environment is properly set up"
    echo "2. Required packages are installed (torch, matplotlib, numpy, seaborn)"
    echo "3. All required modules are available"
    echo "4. Current directory contains all necessary scripts"
    echo ""
    exit 1
fi

echo "Press Enter to exit..."
read