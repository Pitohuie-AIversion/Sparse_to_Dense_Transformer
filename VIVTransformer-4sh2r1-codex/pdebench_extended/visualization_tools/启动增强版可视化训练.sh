#!/bin/bash

# 增强版可视化训练启动脚本 (Linux版本)
# Enhanced Visual Training Launcher for Linux

echo "========================================"
echo "启动增强版可视化训练演示 (Linux版本)"
echo "Enhanced Visual Training Demo (Linux)"
echo "========================================"
echo ""
echo "功能说明 / Features:"
echo "- 实时训练指标可视化 / Real-time training metrics visualization"
echo "- 固定轮数预测对比 / Fixed-epoch prediction comparison"
echo "- 误差分析 / Error analysis"
echo "- 训练进度跟踪 / Training progress tracking"
echo "- 硬件监控 / Hardware monitoring"
echo "- 性能分析 / Performance analysis"
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

# 运行增强版可视化训练
echo "开始运行增强版可视化训练演示..."
echo "Starting enhanced visual training demo..."
echo ""

python3 enhanced_visual_training.py --epochs 20 --output-dir enhanced_training_output

# 检查执行结果
if [ $? -eq 0 ]; then
    echo ""
    echo "========================================"
    echo "Enhanced training demo completed successfully!"
    echo "========================================"
    echo ""
    echo "Generated visualizations can be found in:"
    echo "- enhanced_training_output/plots/              (Training curves)"
    echo "- enhanced_training_output/predictions/        (Prediction visualizations)"
    echo "  - single_samples/                            (Individual predictions)"
    echo "  - batch_comparisons/                         (Batch predictions)"
    echo "  - error_analysis/                            (Error analysis)"
    echo "  - time_series/                               (Training progress)"
    echo "- enhanced_training_output/logs/               (Training logs)"
    echo "- enhanced_training_output/hardware_logs/      (Hardware monitoring)"
    echo ""
    echo "Check the comprehensive report:"
    echo "- enhanced_training_output/comprehensive_training_report.md"
    echo ""
    echo "Open the HTML report in your browser:"
    echo "- enhanced_training_output/hardware_logs/training_report.html"
    echo ""
else
    echo ""
    echo "========================================"
    echo "Error occurred during enhanced training demo!"
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