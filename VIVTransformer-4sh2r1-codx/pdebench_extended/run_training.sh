#!/bin/bash
# =============================================================================
# VIVTransformer 压力场重建训练脚本 - Shell版本
# 适用于Linux/macOS环境的训练启动脚本
# =============================================================================

set -e  # 遇到错误立即退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# 配置参数
DATA_PATH="data/pressure_field_data.pt"
CONFIG_FILE="configs/pressure_field_training.yaml"
OUTPUT_DIR="outputs"
EXPERIMENT_NAME="pressure_field_training_$(date +%Y%m%d_%H%M%S)"

echo -e "${CYAN}╔══════════════════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║                    VIVTransformer - 压力场重建训练                          ║${NC}"
echo -e "${CYAN}║                        Shell环境版本                                        ║${NC}"
echo -e "${CYAN}╚══════════════════════════════════════════════════════════════════════════════╝${NC}"
echo

echo -e "${CYAN}=== 环境检查 ===${NC}"
echo

# 检查Python
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version 2>&1)
    echo -e "${GREEN}✓${NC} Python: $PYTHON_VERSION"
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_VERSION=$(python --version 2>&1)
    echo -e "${GREEN}✓${NC} Python: $PYTHON_VERSION"
    PYTHON_CMD="python"
else
    echo -e "${RED}❌ Python未安装或不在PATH中${NC}"
    echo "请安装Python 3.7+"
    exit 1
fi

# 检查PyTorch
if $PYTHON_CMD -c "import torch" &> /dev/null; then
    TORCH_VERSION=$($PYTHON_CMD -c "import torch; print(torch.__version__)" 2>&1)
    CUDA_AVAILABLE=$($PYTHON_CMD -c "import torch; print(torch.cuda.is_available())" 2>&1)
    echo -e "${GREEN}✓${NC} PyTorch: $TORCH_VERSION (CUDA: $CUDA_AVAILABLE)"
else
    echo -e "${RED}❌ PyTorch未安装${NC}"
    echo "请安装PyTorch: pip install torch torchvision torchaudio"
    exit 1
fi

# 检查其他依赖
echo -e "${BLUE}检查其他依赖包...${NC}"
if $PYTHON_CMD -c "import numpy, scipy, matplotlib, yaml" &> /dev/null; then
    echo -e "${GREEN}✓${NC} 基础依赖包已安装"
else
    echo -e "${YELLOW}⚠️  部分依赖包未安装${NC}"
    echo "请运行: pip install -r requirements.txt"
fi

# 检查NVIDIA GPU
if command -v nvidia-smi &> /dev/null; then
    echo -e "${GREEN}✓${NC} NVIDIA GPU驱动已安装"
    echo
    echo -e "${BLUE}=== GPU状态 ===${NC}"
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader,nounits
else
    echo -e "${YELLOW}⚠️  NVIDIA驱动未安装或GPU不可用${NC}"
    echo "将使用CPU训练（速度较慢）"
fi

echo
echo -e "${PURPLE}=== 数据集检查 ===${NC}"

# 检查数据文件
DATA_FOUND=0

# 检查指定路径的数据文件
if [ -f "$DATA_PATH" ]; then
    DATA_SIZE=$(stat -f%z "$DATA_PATH" 2>/dev/null || stat -c%s "$DATA_PATH" 2>/dev/null)
    DATA_SIZE_MB=$((DATA_SIZE / 1024 / 1024))
    echo -e "${GREEN}✓${NC} 数据文件: ${DATA_SIZE_MB}MB"
    DATA_FOUND=1
else
    echo "搜索现有数据文件..."
    
    # 搜索当前目录的数据文件
    for ext in pt hdf5 h5; do
        for file in *.$ext; do
            if [ -f "$file" ]; then
                DATA_PATH="$file"
                DATA_SIZE=$(stat -f%z "$file" 2>/dev/null || stat -c%s "$file" 2>/dev/null)
                DATA_SIZE_MB=$((DATA_SIZE / 1024 / 1024))
                echo -e "${GREEN}✓${NC} 找到数据文件: $file (${DATA_SIZE_MB}MB)"
                DATA_FOUND=1
                break 2
            fi
        done
    done
    
    # 在data目录中搜索
    if [ $DATA_FOUND -eq 0 ] && [ -d "data" ]; then
        for ext in pt hdf5 h5; do
            while IFS= read -r -d '' file; do
                DATA_PATH="$file"
                DATA_SIZE=$(stat -f%z "$file" 2>/dev/null || stat -c%s "$file" 2>/dev/null)
                DATA_SIZE_MB=$((DATA_SIZE / 1024 / 1024))
                echo -e "${GREEN}✓${NC} 找到数据文件: $file (${DATA_SIZE_MB}MB)"
                DATA_FOUND=1
                break 2
            done < <(find data -name "*.$ext" -type f -print0 2>/dev/null)
        done
    fi
    
    # 检查用户提供的数据集路径
    USER_DATA_PATH="x:/2025/Graduation_project/report/VIVTransformer-4sh2r1-codx/pdebench_extended/data/PDEBench/pdebench/data_download/data/2D/DarcyFlow/2D_DarcyFlow_beta0.1_Train.hdf5"
    if [ -f "$USER_DATA_PATH" ]; then
        DATA_PATH="$USER_DATA_PATH"
        DATA_SIZE=$(stat -f%z "$USER_DATA_PATH" 2>/dev/null || stat -c%s "$USER_DATA_PATH" 2>/dev/null)
        DATA_SIZE_MB=$((DATA_SIZE / 1024 / 1024))
        echo -e "${GREEN}✓${NC} 找到用户数据集: ${DATA_SIZE_MB}MB"
        DATA_FOUND=1
    fi
    
    if [ $DATA_FOUND -eq 0 ]; then
        echo -e "${YELLOW}⚠️  未找到数据文件${NC}"
        echo
        echo "数据获取选项:"
        echo "1. 运行数据下载脚本"
        echo "2. 手动下载数据"
        echo "3. 生成示例数据用于测试"
        echo "4. 退出"
        echo
        read -p "请选择 [1-4]: " DATA_CHOICE
        
        case $DATA_CHOICE in
            1)
                echo -e "${BLUE}启动数据下载...${NC}"
                if [ -f "quick_start.sh" ]; then
                    bash quick_start.sh
                else
                    echo -e "${RED}未找到数据下载脚本${NC}"
                    exit 1
                fi
                ;;
            2)
                echo -e "${YELLOW}请手动下载数据文件到 data/ 目录${NC}"
                echo "数据源:"
                echo "- GitHub: https://github.com/pdebench/PDEBench"
                echo "- 直接下载: https://darus.uni-stuttgart.de/dataset.xhtml?persistentId=doi:10.18419/darus-2986"
                exit 1
                ;;
            3)
                echo -e "${BLUE}生成示例数据...${NC}"
                mkdir -p data
                $PYTHON_CMD -c "
import torch
import numpy as np
import os

# 创建示例压力场数据
print('生成示例压力场数据...')
data = {
    'pressure_field': torch.randn(100, 64, 64),  # 100个样本，64x64网格
    'coordinates': torch.meshgrid(torch.linspace(0, 1, 64), torch.linspace(0, 1, 64)),
    'reynolds_number': torch.rand(100) * 1000 + 100,
    'time_steps': torch.arange(100)
}

os.makedirs('data', exist_ok=True)
torch.save(data, 'data/sample_pressure_data.pt')
print('示例数据已生成: data/sample_pressure_data.pt')
"
                DATA_PATH="data/sample_pressure_data.pt"
                DATA_FOUND=1
                ;;
            4)
                echo "退出"
                exit 0
                ;;
            *)
                echo "无效选择"
                exit 1
                ;;
        esac
    fi
fi

if [ $DATA_FOUND -eq 0 ]; then
    echo -e "${RED}❌ 未找到数据文件${NC}"
    echo "请先获取数据集后再运行训练"
    exit 1
fi

# 检查配置文件
if [ ! -f "$CONFIG_FILE" ]; then
    echo -e "${RED}❌ 配置文件不存在: $CONFIG_FILE${NC}"
    echo "请检查配置文件路径"
    exit 1
else
    echo -e "${GREEN}✓${NC} 配置文件: $CONFIG_FILE"
fi

# 创建输出目录
mkdir -p "$OUTPUT_DIR"
echo -e "${GREEN}✓${NC} 输出目录: $OUTPUT_DIR"

echo
echo -e "${CYAN}=== 训练配置 ===${NC}"
echo "数据文件: $DATA_PATH"
echo "配置文件: $CONFIG_FILE"
echo "输出目录: $OUTPUT_DIR"
echo "实验名称: $EXPERIMENT_NAME"
echo

# 显示运行选项
echo -e "${PURPLE}=== 运行选项 ===${NC}"
echo "1. 开始训练"
echo "2. 启动TensorBoard监控"
echo "3. 测试模式（快速验证）"
echo "4. 查看配置信息"
echo "5. 退出"
echo
read -p "请选择运行模式 [1-5]: " RUN_CHOICE

case $RUN_CHOICE in
    1)
        echo -e "${GREEN}开始训练...${NC}"
        echo "训练命令: $PYTHON_CMD train_pressure_field.py --config $CONFIG_FILE --data_path $DATA_PATH"
        echo
        $PYTHON_CMD train_pressure_field.py --config "$CONFIG_FILE" --data_path "$DATA_PATH"
        ;;
    2)
        echo -e "${BLUE}启动TensorBoard...${NC}"
        if [ -d "logs" ]; then
            echo "TensorBoard URL: http://localhost:6006"
            tensorboard --logdir=logs --port=6006 &
            echo "TensorBoard已在后台启动"
        else
            echo -e "${YELLOW}未找到日志目录，请先运行训练${NC}"
        fi
        ;;
    3)
        echo -e "${YELLOW}测试模式...${NC}"
        echo "运行快速验证（1个epoch）"
        $PYTHON_CMD train_pressure_field.py --config "$CONFIG_FILE" --data_path "$DATA_PATH" --epochs 1
        ;;
    4)
        echo -e "${CYAN}配置信息:${NC}"
        if command -v yq &> /dev/null; then
            yq eval '.' "$CONFIG_FILE"
        else
            echo "配置文件内容:"
            cat "$CONFIG_FILE"
        fi
        ;;
    5)
        echo "退出"
        exit 0
        ;;
    *)
        echo "无效选择，默认开始训练"
        $PYTHON_CMD train_pressure_field.py --config "$CONFIG_FILE" --data_path "$DATA_PATH"
        ;;
esac

echo
echo -e "${GREEN}脚本执行完成!${NC}"
echo -e "${CYAN}如需查看训练进度，请访问: http://localhost:6006${NC}"