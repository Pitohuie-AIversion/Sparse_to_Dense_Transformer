#!/bin/bash

# =============================================================================
# 压力场训练 - 服务器优化版一键运行脚本
# 适用于服务器环境的快速启动训练，包含完整的资源管理和监控
# =============================================================================

set -e

# 服务器配置参数 (可根据需要修改)
DATA_PATH="${DATA_PATH:-./data/pressure_data.pt}"  # 数据文件路径
CONFIG_PATH="pdebench_extended/configs/pressure_field_training.yaml"  # 配置文件路径
OUTPUT_BASE="./outputs"  # 输出基础目录
EXPERIMENT_NAME="pressure_field_$(date +%Y%m%d_%H%M%S)"  # 实验名称
LOG_DIR="./logs"                                     # 日志目录
MAX_RETRIES=3                                        # 最大重试次数
HEALTH_CHECK_INTERVAL=300                           # 健康检查间隔(秒)

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${BLUE}=== 压力场训练一键启动脚本 ===${NC}"
echo "时间: $(date)"
echo "主机: $(hostname)"
echo ""

# 服务器环境检查和初始化
echo -e "${CYAN}=== 服务器环境检查 ===${NC}"

# 创建必要目录
mkdir -p "$LOG_DIR" "$OUTPUT_BASE" "$DATA_DIR"

# 日志文件设置
SCRIPT_LOG="$LOG_DIR/training_script_$(date +%Y%m%d_%H%M%S).log"
SYSTEM_LOG="$LOG_DIR/system_monitor_$(date +%Y%m%d_%H%M%S).log"

# 日志函数
log_info() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [INFO] $1" | tee -a "$SCRIPT_LOG"
}

log_error() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [ERROR] $1" | tee -a "$SCRIPT_LOG" >&2
}

log_warn() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [WARN] $1" | tee -a "$SCRIPT_LOG"
}

# 系统信息收集
log_info "开始服务器环境检查"
echo -e "${BLUE}主机信息:${NC}"
echo "  主机名: $(hostname)"
echo "  操作系统: $(uname -s)"
echo "  内核版本: $(uname -r)"
echo "  CPU核心数: $(nproc)"
echo "  内存总量: $(free -h | awk '/^Mem:/ {print $2}')"
echo "  磁盘空间: $(df -h . | awk 'NR==2 {print $4}' | head -1) 可用"

# PyCharm环境运行 - 跳过虚拟环境检查
if [[ -n "$PYCHARM_HOSTED" ]]; then
    log_warn "检测到PyCharm环境，建议在服务器终端中运行"
    echo -e "${YELLOW}检测到PyCharm环境${NC}"
    echo "建议在服务器终端中运行此脚本以获得最佳性能"
    echo ""
fi

echo -e "${BLUE}使用当前Python环境...${NC}"
echo "注意: 确保已安装所需的Python依赖包"

# 服务器环境变量优化设置
log_info "配置服务器环境变量"

# 根据CPU核心数动态设置线程数
CPU_CORES=$(nproc)
OPTIMAL_THREADS=$((CPU_CORES / 2))
if [[ $OPTIMAL_THREADS -lt 4 ]]; then
    OPTIMAL_THREADS=4
elif [[ $OPTIMAL_THREADS -gt 16 ]]; then
    OPTIMAL_THREADS=16
fi

export PYTHONPATH="$PWD/pdebench_extended:$PYTHONPATH"
export OMP_NUM_THREADS=$OPTIMAL_THREADS
export MKL_NUM_THREADS=$OPTIMAL_THREADS
export NUMEXPR_MAX_THREADS=$OPTIMAL_THREADS
export OPENBLAS_NUM_THREADS=$OPTIMAL_THREADS

# 服务器性能优化
export MALLOC_TRIM_THRESHOLD_=100000
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

echo -e "${GREEN}服务器环境变量已优化设置${NC}"
echo "  PYTHONPATH: $PYTHONPATH"
echo "  线程数设置: $OPTIMAL_THREADS (基于 $CPU_CORES 核心)"
echo "  内存优化: 已启用"
echo ""

# 服务器GPU配置和管理
echo -e "${BLUE}=== 服务器GPU配置 ===${NC}"
log_info "开始GPU环境检测和配置"

# 检测GPU
if command -v nvidia-smi &> /dev/null; then
    echo "检测到NVIDIA GPU环境:"
    
    # 详细GPU信息
    echo -e "${CYAN}GPU详细信息:${NC}"
    nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu --format=csv,noheader,nounits | while IFS=',' read -r idx name total used free util temp; do
        echo "  GPU $idx: $name"
        echo "    内存: ${used}MB/${total}MB (空闲: ${free}MB)"
        echo "    利用率: ${util}% | 温度: ${temp}°C"
    done
    echo ""
    
    # 获取GPU数量和状态
    GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
    echo "可用GPU数量: $GPU_COUNT"
    
    # 服务器GPU智能选择策略
    if [[ -n "$CUDA_VISIBLE_DEVICES" ]]; then
        log_info "使用预设GPU: $CUDA_VISIBLE_DEVICES"
        echo -e "${GREEN}使用预设GPU: $CUDA_VISIBLE_DEVICES${NC}"
    elif [[ $GPU_COUNT -gt 1 ]]; then
        echo -e "${PURPLE}多GPU服务器环境，选择策略:${NC}"
        echo "1. 使用GPU 0 (通常为主GPU)"
        echo "2. 使用GPU 1 (推荐用于训练)"
        echo "3. 使用所有GPU并行训练"
        echo "4. 自动选择最空闲的GPU (推荐)"
        echo "5. 手动指定GPU"
        read -p "请选择 (1-5): " gpu_choice
        
        case $gpu_choice in
            1)
                export CUDA_VISIBLE_DEVICES=0
                log_info "手动选择GPU 0"
                echo -e "${GREEN}设置使用GPU 0${NC}"
                ;;
            2)
                export CUDA_VISIBLE_DEVICES=1
                log_info "手动选择GPU 1"
                echo -e "${GREEN}设置使用GPU 1${NC}"
                ;;
            3)
                export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((GPU_COUNT-1)))
                log_info "启用所有GPU并行训练: $CUDA_VISIBLE_DEVICES"
                echo -e "${GREEN}设置使用所有GPU并行训练: $CUDA_VISIBLE_DEVICES${NC}"
                ;;
            4)
                # 智能选择：综合考虑显存使用率和GPU利用率
                BEST_GPU=$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits | awk -F',' '{score = $2 + $3*10; print $1, score}' | sort -k2 -n | head -1 | cut -d' ' -f1)
                export CUDA_VISIBLE_DEVICES=$BEST_GPU
                log_info "自动选择最优GPU: $BEST_GPU"
                echo -e "${GREEN}智能选择GPU $BEST_GPU (综合负载最低)${NC}"
                ;;
            5)
                echo "可用GPU: $(seq -s' ' 0 $((GPU_COUNT-1)))"
                read -p "请输入GPU编号 (如: 0,1): " manual_gpu
                export CUDA_VISIBLE_DEVICES=$manual_gpu
                log_info "手动指定GPU: $manual_gpu"
                echo -e "${GREEN}设置使用GPU: $manual_gpu${NC}"
                ;;
            *)
                echo -e "${YELLOW}无效选择，自动选择最优GPU${NC}"
                BEST_GPU=$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits | awk -F',' '{score = $2 + $3*10; print $1, score}' | sort -k2 -n | head -1 | cut -d' ' -f1)
                export CUDA_VISIBLE_DEVICES=$BEST_GPU
                log_info "默认选择最优GPU: $BEST_GPU"
                ;;
        esac
    else
        export CUDA_VISIBLE_DEVICES=0
        log_info "单GPU环境，使用GPU 0"
        echo -e "${GREEN}单GPU环境，使用GPU 0${NC}"
    fi
    
    # 显示最终GPU配置
    echo -e "${CYAN}最终GPU配置:${NC}"
    echo "  CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
    
    # GPU状态监控
    echo -e "${CYAN}选定GPU当前状态:${NC}"
    for gpu_id in $(echo $CUDA_VISIBLE_DEVICES | tr ',' ' '); do
        gpu_info=$(nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader,nounits -i $gpu_id)
        echo "  GPU $gpu_id: $gpu_info"
    done
    
else
    log_warn "未检测到NVIDIA GPU，将使用CPU训练"
    echo -e "${YELLOW}未检测到NVIDIA GPU，将使用CPU训练${NC}"
    echo -e "${RED}警告: CPU训练速度极慢，强烈建议使用GPU服务器${NC}"
fi
echo ""

# 数据集下载和检查
echo -e "${BLUE}=== 数据集检查与下载 ===${NC}"

# 定义数据集相关变量 (基于PDEBench官方文档)
DATA_REPO_URL="https://github.com/pdebench/PDEBench.git"
DATA_DIR="./data"
DATA_SUBDIR="$DATA_DIR/2D/CFD/2D_Train_Rand"

# PDEBench官方数据仓库 (DaRUS)
PDEBENCH_DATASET_DOI="doi:10.18419/darus-2986"
PDEBENCH_DATASET_URL="https://darus.uni-stuttgart.de/dataset.xhtml?persistentId=doi:10.18419/darus-2986"
PDEBENCH_MODELS_DOI="doi:10.18419/darus-2987"
PDEBENCH_MODELS_URL="https://darus.uni-stuttgart.de/dataset.xhtml?persistentId=doi:10.18419/darus-2987"

# 备用下载链接 (基于官方文档)
ALTERNATIVE_DATA_URLS=(
    "https://darus.uni-stuttgart.de/api/access/datafile/132004"  # 2D CFD数据
    "https://darus.uni-stuttgart.de/api/access/datafile/132005"  # 备用CFD数据
)

# 检查数据文件是否存在
if [[ ! -f "$DATA_PATH" ]]; then
    echo -e "${YELLOW}数据文件不存在: $DATA_PATH${NC}"
    echo "正在搜索现有数据文件..."
    
    # 自动搜索.pt和.hdf5文件
    DATA_FILES=($(find . -name "*.pt" -o -name "*.hdf5" -type f 2>/dev/null | head -10))
    
    if [[ ${#DATA_FILES[@]} -eq 0 ]]; then
        echo -e "${YELLOW}未找到现有数据文件，开始下载数据集...${NC}"
        
        # 询问用户下载方式 (基于PDEBench官方推荐)
        echo "PDEBench数据集下载选项:"
        echo "1. 使用PDEBench官方下载脚本 (推荐，来自DaRUS数据仓库)"
        echo "2. 从GitHub克隆PDEBench仓库 (包含代码和小数据集)"
        echo "3. 直接从DaRUS下载特定数据文件 (快速)"
        echo "4. 跳过下载，手动指定数据路径"
        echo ""
        echo "注意: PDEBench官方数据集托管在DaRUS数据仓库中"
        echo "数据集DOI: $PDEBENCH_DATASET_DOI"
        echo "访问地址: $PDEBENCH_DATASET_URL"
        read -p "请选择下载方式 (1-4): " download_choice
        
        case $download_choice in
            1)
                echo -e "${BLUE}使用PDEBench官方下载脚本...${NC}"
                echo "正在设置PDEBench官方数据下载..."
                
                # 首先克隆PDEBench仓库获取下载脚本
                if [[ ! -d "$DATA_DIR/PDEBench" ]]; then
                    echo "克隆PDEBench仓库以获取官方下载脚本..."
                    git clone --depth 1 "$DATA_REPO_URL" "$DATA_DIR/PDEBench"
                fi
                
                if [[ -d "$DATA_DIR/PDEBench" ]]; then
                    echo -e "${GREEN}PDEBench仓库准备完成${NC}"
                    echo "使用官方下载脚本下载数据集..."
                    
                    # 检查是否存在官方下载脚本
                    if [[ -f "$DATA_DIR/PDEBench/pdebench/data_download/download_direct.py" ]]; then
                        echo "找到官方下载脚本，开始下载数据..."
                        cd "$DATA_DIR/PDEBench"
                        
                        # 使用官方推荐的下载方式
                        echo "可用的数据集类型:"
                        echo "  - 2D CFD (Computational Fluid Dynamics)"
                        echo "  - 1D Advection, Burgers, Reaction-Diffusion"
                        echo "  - 2D Shallow Water, Diffusion-Reaction"
                        echo "  - Compressible Navier-Stokes"
                        echo ""
                        echo "正在下载2D CFD数据集 (适合压力场训练)..."
                        
                        # 尝试使用官方下载脚本
                        if python pdebench/data_download/download_direct.py --root_folder "../../" --pde_name "CFD" --split "train" --resolution "all"; then
                            echo -e "${GREEN}官方数据下载成功${NC}"
                            # 搜索下载的数据文件
                            DOWNLOADED_FILES=($(find "../../" -name "*CFD*.hdf5" -o -name "*CFD*.h5" 2>/dev/null | head -10))
                            if [[ ${#DOWNLOADED_FILES[@]} -gt 0 ]]; then
                                DATA_PATH="${DOWNLOADED_FILES[0]}"
                                echo -e "${GREEN}找到下载的数据文件: $DATA_PATH${NC}"
                            fi
                        else
                            echo -e "${YELLOW}官方下载脚本执行失败，尝试手动下载...${NC}"
                        fi
                        cd - > /dev/null
                    else
                        echo -e "${YELLOW}未找到官方下载脚本，切换到GitHub克隆模式...${NC}"
                    fi
                else
                    echo -e "${RED}PDEBench仓库克隆失败${NC}"
                fi
                ;;
            2)
                echo -e "${BLUE}从GitHub克隆PDEBench仓库...${NC}"
                
                # 检查git是否安装
                if ! command -v git &> /dev/null; then
                    echo -e "${RED}错误: git未安装${NC}"
                    echo "请安装git: sudo yum install -y git"
                    exit 1
                fi
                
                # 检查git lfs
                if ! command -v git-lfs &> /dev/null; then
                    echo -e "${YELLOW}警告: git-lfs未安装，大文件可能下载失败${NC}"
                    
                    # 检查是否在conda环境中
                    if [[ -n "$CONDA_DEFAULT_ENV" ]] || command -v conda &> /dev/null; then
                        echo -e "${BLUE}检测到conda环境，尝试自动安装git-lfs...${NC}"
                        read -p "是否尝试通过conda安装git-lfs? (y/n): " install_choice
                        if [[ "$install_choice" =~ ^[Yy]$ ]]; then
                            echo "正在安装git-lfs..."
                            if conda install -c conda-forge git-lfs -y; then
                                echo -e "${GREEN}git-lfs安装成功${NC}"
                            else
                                echo -e "${YELLOW}conda安装失败，尝试pip安装...${NC}"
                                if pip install git-lfs; then
                                    echo -e "${GREEN}git-lfs通过pip安装成功${NC}"
                                else
                                    echo -e "${RED}自动安装失败${NC}"
                                fi
                            fi
                        fi
                    else
                        echo "安装选项:"
                        echo "1. 系统安装 (需要sudo): sudo yum install -y git-lfs"
                        echo "2. Conda安装 (推荐): conda install -c conda-forge git-lfs"
                        echo "3. 手动安装到用户目录"
                    fi
                fi
                
                # 创建数据目录
                mkdir -p "$DATA_DIR"
                
                # 克隆仓库（浅克隆以节省时间）
                if [[ ! -d "$DATA_DIR/PDEBench" ]]; then
                    echo "正在克隆PDEBench仓库..."
                    git clone --depth 1 "$DATA_REPO_URL" "$DATA_DIR/PDEBench"
                    
                    if [[ $? -eq 0 ]]; then
                        echo -e "${GREEN}仓库克隆成功${NC}"
                        
                        # 尝试下载LFS文件
                        cd "$DATA_DIR/PDEBench"
                        if command -v git-lfs &> /dev/null; then
                            echo "正在下载大文件..."
                            git lfs pull
                        fi
                        cd - > /dev/null
                        
                        # 搜索数据文件
                        echo "正在搜索下载的数据文件..."
                        DOWNLOADED_FILES=($(find "$DATA_DIR/PDEBench" -name "*.hdf5" -o -name "*.pt" -o -name "*.h5" 2>/dev/null | head -10))
                        if [[ ${#DOWNLOADED_FILES[@]} -gt 0 ]]; then
                            DATA_PATH="${DOWNLOADED_FILES[0]}"
                            echo -e "${GREEN}找到数据文件: $DATA_PATH${NC}"
                        else
                            echo -e "${YELLOW}GitHub仓库中未找到大数据文件${NC}"
                            echo -e "${BLUE}PDEBench数据集托管在DaRUS数据仓库中，需要单独下载${NC}"
                            echo ""
                            echo "推荐的数据获取方式:"
                            echo "1. 访问官方数据仓库: $PDEBENCH_DATASET_URL"
                            echo "2. 使用官方下载脚本 (如果可用):"
                            echo "   cd $DATA_DIR/PDEBench"
                            echo "   python pdebench/data_download/download_direct.py --help"
                            echo "3. 或者选择选项3直接从DaRUS下载"
                            echo ""
                            echo "注意: GitHub仓库主要包含代码，大数据集在DaRUS中"
                        fi
                    else
                        echo -e "${RED}仓库克隆失败${NC}"
                    fi
                else
                    echo -e "${GREEN}PDEBench仓库已存在${NC}"
                    # 搜索现有文件
                    echo "搜索现有数据文件..."
                    DOWNLOADED_FILES=($(find "$DATA_DIR/PDEBench" -name "*.hdf5" -o -name "*.pt" -o -name "*.h5" 2>/dev/null | head -10))
                    if [[ ${#DOWNLOADED_FILES[@]} -gt 0 ]]; then
                        DATA_PATH="${DOWNLOADED_FILES[0]}"
                        echo -e "${GREEN}使用现有数据文件: $DATA_PATH${NC}"
                    else
                        echo -e "${YELLOW}现有仓库中未找到数据文件${NC}"
                        echo "尝试更新仓库并下载LFS文件..."
                        cd "$DATA_DIR/PDEBench"
                        git pull
                        if command -v git-lfs &> /dev/null; then
                            echo "正在下载LFS文件..."
                            git lfs pull
                            cd - > /dev/null
                            # 重新搜索
                            DOWNLOADED_FILES=($(find "$DATA_DIR/PDEBench" -name "*.hdf5" -o -name "*.pt" -o -name "*.h5" 2>/dev/null | head -10))
                            if [[ ${#DOWNLOADED_FILES[@]} -gt 0 ]]; then
                                DATA_PATH="${DOWNLOADED_FILES[0]}"
                                echo -e "${GREEN}LFS下载后找到数据文件: $DATA_PATH${NC}"
                            fi
                        else
                            cd - > /dev/null
                            echo -e "${YELLOW}git-lfs未安装，无法下载大文件${NC}"
                        fi
                    fi
                fi
                ;;
            3)
                echo -e "${BLUE}直接从DaRUS下载数据文件...${NC}"
                echo "正在从PDEBench官方数据仓库下载..."
                
                # 检查wget或curl
                if command -v wget &> /dev/null; then
                    DOWNLOAD_CMD="wget -O"
                elif command -v curl &> /dev/null; then
                    DOWNLOAD_CMD="curl -L -o"
                else
                    echo -e "${RED}错误: 未找到wget或curl${NC}"
                    echo "请联系管理员安装下载工具:"
                    echo "  CentOS/RHEL: yum install -y wget curl"
                    echo "  Ubuntu/Debian: apt install -y wget curl"
                    echo "或者选择其他数据获取方式"
                    exit 1
                fi
                
                mkdir -p "$DATA_DIR"
                
                echo "可用的PDEBench数据集 (来自DaRUS数据仓库):"
                echo "1. 2D CFD (Computational Fluid Dynamics) - 适合压力场训练"
                echo "2. 1D Advection Equation"
                echo "3. 1D Burgers Equation"
                echo "4. 2D Shallow Water"
                echo "5. 2D Diffusion-Reaction"
                echo ""
                read -p "请选择要下载的数据集类型 (1-5): " dataset_choice
                
                case $dataset_choice in
                    1)
                        echo "下载2D CFD数据集..."
                        dataset_name="2D_CFD"
                        ;;
                    2)
                        echo "下载1D Advection数据集..."
                        dataset_name="1D_Advection"
                        ;;
                    3)
                        echo "下载1D Burgers数据集..."
                        dataset_name="1D_Burgers"
                        ;;
                    4)
                        echo "下载2D Shallow Water数据集..."
                        dataset_name="2D_SWE"
                        ;;
                    5)
                        echo "下载2D Diffusion-Reaction数据集..."
                        dataset_name="2D_DiffReact"
                        ;;
                    *)
                        echo "默认选择2D CFD数据集"
                        dataset_name="2D_CFD"
                        ;;
                esac
                
                # 尝试下载选定的数据集
                for i in "${!ALTERNATIVE_DATA_URLS[@]}"; do
                    url="${ALTERNATIVE_DATA_URLS[$i]}"
                    filename="${dataset_name}_data_$((i+1)).hdf5"
                    filepath="$DATA_DIR/$filename"
                    
                    echo "尝试从DaRUS下载: $url"
                    if $DOWNLOAD_CMD "$filepath" "$url"; then
                        if [[ -f "$filepath" && $(stat -f%z "$filepath" 2>/dev/null || stat -c%s "$filepath" 2>/dev/null) -gt 1000000 ]]; then
                            DATA_PATH="$filepath"
                            echo -e "${GREEN}DaRUS数据下载成功: $DATA_PATH${NC}"
                            break
                        else
                            echo -e "${YELLOW}下载的文件太小，可能下载失败${NC}"
                            rm -f "$filepath"
                        fi
                    else
                        echo -e "${YELLOW}下载失败，尝试下一个源...${NC}"
                    fi
                done
                
                if [[ ! -f "$DATA_PATH" ]]; then
                    echo -e "${RED}自动下载失败${NC}"
                    echo -e "${BLUE}请手动访问PDEBench官方数据仓库:${NC}"
                    echo "  数据集: $PDEBENCH_DATASET_URL"
                    echo "  预训练模型: $PDEBENCH_MODELS_URL"
                    echo ""
                    echo "下载步骤:"
                    echo "1. 访问上述链接"
                    echo "2. 选择所需的数据集文件"
                    echo "3. 下载到 $DATA_DIR 目录"
                    echo "4. 重新运行此脚本"
                    echo ""
                    echo "可选操作:"
                    echo "1. 重新搜索现有数据文件"
                    echo "2. 生成示例数据进行测试"
                    echo "3. 手动指定数据文件路径"
                    echo "4. 退出脚本并手动下载"
                    read -p "请选择 (1-4): " fallback_choice
                    
                    case $fallback_choice in
                        1)
                            echo "重新搜索数据文件..."
                            # 重新搜索
                            EXISTING_FILES=($(find . -name "*.hdf5" -o -name "*.pt" -o -name "*.h5" 2>/dev/null | head -10))
                            if [[ ${#EXISTING_FILES[@]} -gt 0 ]]; then
                                echo "找到以下数据文件:"
                                for i in "${!EXISTING_FILES[@]}"; do
                                    echo "$((i+1)). ${EXISTING_FILES[$i]}"
                                done
                                read -p "请选择文件编号 (1-${#EXISTING_FILES[@]}): " file_choice
                                if [[ $file_choice -ge 1 && $file_choice -le ${#EXISTING_FILES[@]} ]]; then
                                    DATA_PATH="${EXISTING_FILES[$((file_choice-1))]}"
                                    echo -e "${GREEN}选择数据文件: $DATA_PATH${NC}"
                                fi
                            else
                                echo -e "${YELLOW}未找到任何数据文件${NC}"
                                echo "建议:"
                                echo "1. 从DaRUS数据仓库手动下载数据集"
                                echo "2. 确保数据文件格式为 .hdf5, .h5 或 .pt"
                                echo "3. 将数据文件放置在项目目录或子目录中"
                            fi
                            ;;
                        2)
                            echo -e "${BLUE}生成示例数据文件...${NC}"
                            mkdir -p "$DATA_DIR/generated"
                            SAMPLE_DATA="$DATA_DIR/generated/sample_data.pt"
                            
                            # 创建一个简单的示例数据文件
                            python3 -c "
import torch
import os

# 生成示例压力场数据
data = {
    'pressure': torch.randn(100, 64, 64),  # 100个样本，64x64网格
    'velocity_x': torch.randn(100, 64, 64),
    'velocity_y': torch.randn(100, 64, 64),
    'time_steps': torch.linspace(0, 1, 100)
}

os.makedirs(os.path.dirname('$SAMPLE_DATA'), exist_ok=True)
torch.save(data, '$SAMPLE_DATA')
print('示例数据已生成: $SAMPLE_DATA')
" 2>/dev/null
                            
                            if [[ -f "$SAMPLE_DATA" ]]; then
                                DATA_PATH="$SAMPLE_DATA"
                                echo -e "${GREEN}示例数据生成成功: $DATA_PATH${NC}"
                                echo -e "${YELLOW}注意：这是示例数据，仅用于测试训练流程${NC}"
                            else
                                echo -e "${RED}示例数据生成失败${NC}"
                            fi
                            ;;
                        3)
                            read -p "请输入数据文件的完整路径: " manual_path
                            if [[ -f "$manual_path" ]]; then
                                DATA_PATH="$manual_path"
                                echo -e "${GREEN}使用手动指定的数据文件: $DATA_PATH${NC}"
                            else
                                echo -e "${RED}指定的文件不存在: $manual_path${NC}"
                            fi
                            ;;
                        4)
                            echo -e "${YELLOW}退出脚本，请手动下载数据后重新运行${NC}"
                            echo ""
                            echo -e "${BLUE}PDEBench数据集手动下载指南:${NC}"
                            echo "1. 访问官方数据仓库:"
                            echo "   数据集: $PDEBENCH_DATASET_URL"
                            echo "   预训练模型: $PDEBENCH_MODELS_URL"
                            echo ""
                            echo "2. 选择并下载所需的数据集:"
                            echo "   - 2D CFD (Computational Fluid Dynamics)"
                            echo "   - 1D Advection, Burgers, Reaction-Diffusion"
                            echo "   - 2D Shallow Water, Diffusion-Reaction"
                            echo "   - Compressible Navier-Stokes"
                            echo ""
                            echo "3. 将下载的文件放置在以下目录:"
                            echo "   $DATA_DIR/"
                            echo ""
                            echo "4. 确保文件格式为 .hdf5, .h5 或 .pt"
                            echo ""
                            echo "5. 重新运行此脚本: bash $0"
                            echo ""
                            echo "注意: PDEBench数据集托管在DaRUS数据仓库中，不在GitHub仓库中"
                            exit 0
                            ;;
                        *)
                            echo -e "${YELLOW}无效选择，跳过下载${NC}"
                            echo "请参考PDEBench官方文档获取数据集"
                            ;;
                    esac
                fi
                ;;
            4)
                echo -e "${YELLOW}跳过自动下载${NC}"
                echo "请手动下载PDEBench数据集:"
                echo "  官方数据仓库: $PDEBENCH_DATASET_URL"
                echo "  数据集DOI: $PDEBENCH_DATASET_DOI"
                ;;
            *)
                echo -e "${YELLOW}无效选择，跳过下载${NC}"
                echo "请参考PDEBench官方文档获取数据集"
                ;;
        esac
    
    # 如果所有下载源都失败，提供其他选项
    if [[ ! -f "$DATA_PATH" ]]; then
        echo -e "${RED}数据下载失败${NC}"
        echo -e "${BLUE}PDEBench数据集获取指南:${NC}"
        echo ""
        echo "官方推荐的数据获取方式:"
        echo "1. 访问DaRUS数据仓库: $PDEBENCH_DATASET_URL"
        echo "2. 下载所需的数据集文件到本地"
        echo "3. 使用PDEBench官方下载工具 (如果可用)"
        echo ""
        echo "可选操作:"
        echo "1. 重新搜索现有数据文件"
        echo "2. 生成示例数据进行测试"
        echo "3. 手动指定数据文件路径"
        echo "4. 退出脚本并手动下载"
        read -p "请选择 (1-4): " fallback_choice
        
        case $fallback_choice in
            1)
                echo "重新搜索数据文件..."
                # 重新搜索
                EXISTING_FILES=($(find . -name "*.hdf5" -o -name "*.pt" -o -name "*.h5" 2>/dev/null | head -10))
                if [[ ${#EXISTING_FILES[@]} -gt 0 ]]; then
                    echo "找到以下数据文件:"
                    for i in "${!EXISTING_FILES[@]}"; do
                        echo "$((i+1)). ${EXISTING_FILES[$i]}"
                    done
                    read -p "请选择文件编号 (1-${#EXISTING_FILES[@]}): " file_choice
                    if [[ $file_choice -ge 1 && $file_choice -le ${#EXISTING_FILES[@]} ]]; then
                        DATA_PATH="${EXISTING_FILES[$((file_choice-1))]}"
                        echo -e "${GREEN}选择数据文件: $DATA_PATH${NC}"
                    fi
                else
                    echo -e "${YELLOW}未找到任何数据文件${NC}"
                    echo "建议:"
                    echo "1. 从DaRUS数据仓库手动下载数据集"
                    echo "2. 确保数据文件格式为 .hdf5, .h5 或 .pt"
                    echo "3. 将数据文件放置在项目目录或子目录中"
                fi
                ;;
            2)
                echo -e "${BLUE}生成示例数据文件...${NC}"
                mkdir -p "$DATA_DIR/generated"
                SAMPLE_DATA="$DATA_DIR/generated/sample_data.pt"
                
                # 创建一个简单的示例数据文件
                python3 -c "
import torch
import os

# 生成示例压力场数据
data = {
    'pressure': torch.randn(100, 64, 64),  # 100个样本，64x64网格
    'velocity_x': torch.randn(100, 64, 64),
    'velocity_y': torch.randn(100, 64, 64),
    'time_steps': torch.linspace(0, 1, 100)
}

os.makedirs(os.path.dirname('$SAMPLE_DATA'), exist_ok=True)
torch.save(data, '$SAMPLE_DATA')
print('示例数据已生成: $SAMPLE_DATA')
" 2>/dev/null
                
                if [[ -f "$SAMPLE_DATA" ]]; then
                    DATA_PATH="$SAMPLE_DATA"
                    echo -e "${GREEN}示例数据生成成功: $DATA_PATH${NC}"
                    echo -e "${YELLOW}注意：这是示例数据，仅用于测试训练流程${NC}"
                else
                    echo -e "${RED}示例数据生成失败${NC}"
                fi
                ;;
            3)
                read -p "请输入数据文件的完整路径: " manual_path
                if [[ -f "$manual_path" ]]; then
                    DATA_PATH="$manual_path"
                    echo -e "${GREEN}使用手动指定的数据文件: $DATA_PATH${NC}"
                else
                    echo -e "${RED}指定的文件不存在: $manual_path${NC}"
                fi
                ;;
            4)
                echo -e "${YELLOW}退出脚本，请手动下载数据后重新运行${NC}"
                echo ""
                echo -e "${BLUE}PDEBench数据集手动下载指南:${NC}"
                echo "1. 访问官方数据仓库:"
                echo "   数据集: $PDEBENCH_DATASET_URL"
                echo "   预训练模型: $PDEBENCH_MODELS_URL"
                echo ""
                echo "2. 选择并下载所需的数据集:"
                echo "   - 2D CFD (Computational Fluid Dynamics)"
                echo "   - 1D Advection, Burgers, Reaction-Diffusion"
                echo "   - 2D Shallow Water, Diffusion-Reaction"
                echo "   - Compressible Navier-Stokes"
                echo ""
                echo "3. 将下载的文件放置在以下目录:"
                echo "   $DATA_DIR/"
                echo ""
                echo "4. 确保文件格式为 .hdf5, .h5 或 .pt"
                echo ""
                echo "5. 重新运行此脚本: bash $0"
                echo ""
                echo "注意: PDEBench数据集托管在DaRUS数据仓库中，不在GitHub仓库中"
                exit 0
                ;;
            *)
                echo -e "${YELLOW}无效选择，跳过下载${NC}"
                echo "请参考PDEBench官方文档获取数据集"
                ;;
        esac
    fi
else
    echo -e "${GREEN}数据文件已存在: $DATA_PATH${NC}"
fi

# 验证数据文件
if [[ -f "$DATA_PATH" ]]; then
    file_size=$(du -h "$DATA_PATH" 2>/dev/null | cut -f1 || echo "未知")
    echo -e "${GREEN}数据文件验证通过${NC}"
    echo "文件路径: $DATA_PATH"
    echo "文件大小: $file_size"
else
    echo -e "${RED}数据文件验证失败${NC}"
    exit 1
fi

# 检查配置文件
if [[ ! -f "$CONFIG_PATH" ]]; then
    echo -e "${RED}错误: 配置文件不存在: $CONFIG_PATH${NC}"
    exit 1
fi

# 创建输出目录
OUTPUT_DIR="$OUTPUT_BASE/$EXPERIMENT_NAME"
mkdir -p "$OUTPUT_DIR"

echo ""
echo -e "${BLUE}=== 训练配置 ===${NC}"
echo "数据文件: $DATA_PATH"
echo "配置文件: $CONFIG_PATH"
echo "输出目录: $OUTPUT_DIR"
echo "实验名称: $EXPERIMENT_NAME"
echo ""

# 服务器训练模式选择
echo -e "${PURPLE}=== 服务器训练模式选择 ===${NC}"
log_info "配置训练启动模式"

echo "选择适合服务器的运行模式:"
echo "1. 前台运行 (调试模式，可看到实时输出)"
echo "2. 后台运行 (适合长时间训练，带完整日志)"
echo "3. Screen会话运行 (推荐，可随时连接查看)"
echo "4. Tmux会话运行 (高级用户推荐)"
echo "5. 系统服务模式 (最稳定，适合生产环境)"
read -p "请选择 (1-5): " mode

# 构建增强的训练命令
TRAIN_CMD="python pdebench_extended/train_pressure_field.py \
    --config \"$CONFIG_PATH\" \
    --data_path \"$DATA_PATH\" \
    --output_dir \"$OUTPUT_DIR\" \
    --experiment_name \"$EXPERIMENT_NAME\""

# 添加服务器监控命令
MONITOR_CMD="while true; do echo \"[\$(date)] GPU状态:\"; nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader,nounits; echo \"系统负载: \$(uptime)\"; sleep $HEALTH_CHECK_INTERVAL; done"

case $mode in
    1)
        echo -e "${GREEN}前台调试模式启动...${NC}"
        log_info "启动前台训练模式"
        echo "按 Ctrl+C 可以停止训练"
        echo "实时输出模式，适合调试和短期训练"
        echo ""
        eval "$TRAIN_CMD"
        ;;
    2)
        echo -e "${GREEN}后台服务模式启动...${NC}"
        log_info "启动后台训练模式"
        
        # 创建详细的后台运行脚本
        BACKGROUND_SCRIPT="$OUTPUT_DIR/run_background.sh"
        cat > "$BACKGROUND_SCRIPT" << EOF
#!/bin/bash
set -e

# 环境变量
export PYTHONPATH="$PWD/pdebench_extended:\$PYTHONPATH"
export OMP_NUM_THREADS=$OPTIMAL_THREADS
export MKL_NUM_THREADS=$OPTIMAL_THREADS
export CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES"

# 启动系统监控
$MONITOR_CMD > "$SYSTEM_LOG" 2>&1 &
MONITOR_PID=\$!
echo \$MONITOR_PID > "$OUTPUT_DIR/monitor.pid"

# 启动训练
echo "[\$(date)] 开始训练" >> "$OUTPUT_DIR/training.log"
$TRAIN_CMD >> "$OUTPUT_DIR/training.log" 2>&1
TRAIN_EXIT_CODE=\$?

# 清理监控进程
kill \$MONITOR_PID 2>/dev/null || true

echo "[\$(date)] 训练结束，退出码: \$TRAIN_EXIT_CODE" >> "$OUTPUT_DIR/training.log"
exit \$TRAIN_EXIT_CODE
EOF
        
        chmod +x "$BACKGROUND_SCRIPT"
        nohup bash "$BACKGROUND_SCRIPT" &
        TRAIN_PID=$!
        echo $TRAIN_PID > "$OUTPUT_DIR/train.pid"
        
        echo "训练PID: $TRAIN_PID"
        echo "训练日志: $OUTPUT_DIR/training.log"
        echo "系统监控日志: $SYSTEM_LOG"
        echo ""
        echo "监控命令:"
        echo "  tail -f $OUTPUT_DIR/training.log    # 训练日志"
        echo "  tail -f $SYSTEM_LOG                 # 系统监控"
        echo "  ps aux | grep $TRAIN_PID            # 进程状态"
        echo "停止命令:"
        echo "  kill $TRAIN_PID                     # 停止训练"
        ;;
    3)
        echo -e "${GREEN}Screen会话模式启动...${NC}"
        log_info "启动Screen会话训练模式"
        
        # 检查screen是否安装
        if ! command -v screen &> /dev/null; then
            echo -e "${RED}错误: screen未安装${NC}"
            echo "请联系管理员安装screen，或使用其他运行模式:"
            echo "  CentOS/RHEL: yum install -y screen"
            echo "  Ubuntu/Debian: apt-get install -y screen"
            echo "  或者选择模式1(前台)或模式2(后台)继续"
            read -p "是否继续使用后台模式? (y/n): " -n 1 -r
            echo
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                echo "切换到后台模式..."
                mode=2
            else
                exit 1
            fi
        fi
        
        SESSION_NAME="pressure_training_$(date +%H%M%S)"
        
        # 创建screen会话并运行训练
        screen -dmS "$SESSION_NAME" bash -c "
            export PYTHONPATH=$PWD/pdebench_extended:\$PYTHONPATH
            export OMP_NUM_THREADS=$OPTIMAL_THREADS
            export MKL_NUM_THREADS=$OPTIMAL_THREADS
            export CUDA_VISIBLE_DEVICES='$CUDA_VISIBLE_DEVICES'
            
            echo '=== 压力场训练会话启动 ==='
            echo '时间: \$(date)'
            echo 'GPU: $CUDA_VISIBLE_DEVICES'
            echo '输出目录: $OUTPUT_DIR'
            echo ''
            
            # 启动监控
            $MONITOR_CMD > '$SYSTEM_LOG' 2>&1 &
            MONITOR_PID=\$!
            
            # 启动训练
            $TRAIN_CMD
            TRAIN_EXIT_CODE=\$?
            
            # 清理
            kill \$MONITOR_PID 2>/dev/null || true
            
            echo ''
            echo '=== 训练完成 ==='
            echo '退出码: '\$TRAIN_EXIT_CODE
            echo '按任意键退出会话...'
            read
        "
        
        echo "Screen会话已创建: $SESSION_NAME"
        echo ""
        echo "管理命令:"
        echo "  screen -r $SESSION_NAME              # 连接到会话"
        echo "  screen -ls                           # 查看所有会话"
        echo "  Ctrl+A+D                           # 分离会话(保持运行)"
        echo "  screen -X -S $SESSION_NAME quit     # 终止会话"
        
        # 等待并验证会话状态
        sleep 2
        if screen -list | grep -q "$SESSION_NAME"; then
            echo -e "${GREEN}Screen训练会话已启动${NC}"
            log_info "Screen会话 $SESSION_NAME 启动成功"
        else
            echo -e "${RED}Screen会话启动失败${NC}"
            log_error "Screen会话启动失败"
        fi
        ;;
    4)
        echo -e "${GREEN}Tmux会话模式启动...${NC}"
        log_info "启动Tmux会话训练模式"
        
        # 检查tmux是否安装
        if ! command -v tmux &> /dev/null; then
            echo -e "${RED}错误: tmux未安装${NC}"
            echo "请联系管理员安装tmux，或使用其他运行模式:"
            echo "  CentOS/RHEL: yum install -y tmux"
            echo "  Ubuntu/Debian: apt-get install -y tmux"
            echo "  或者选择模式3(Screen)或模式2(后台)继续"
            read -p "是否切换到Screen模式? (y/n): " -n 1 -r
            echo
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                echo "切换到Screen模式..."
                mode=3
            else
                read -p "是否切换到后台模式? (y/n): " -n 1 -r
                echo
                if [[ $REPLY =~ ^[Yy]$ ]]; then
                    echo "切换到后台模式..."
                    mode=2
                else
                    exit 1
                fi
            fi
        fi
        
        SESSION_NAME="pressure_training_$(date +%H%M%S)"
        
        # 创建tmux会话
        tmux new-session -d -s "$SESSION_NAME" -c "$PWD"
        
        # 设置环境变量
        tmux send-keys -t "$SESSION_NAME" "export PYTHONPATH=$PWD/pdebench_extended:\$PYTHONPATH" Enter
        tmux send-keys -t "$SESSION_NAME" "export OMP_NUM_THREADS=$OPTIMAL_THREADS" Enter
        tmux send-keys -t "$SESSION_NAME" "export MKL_NUM_THREADS=$OPTIMAL_THREADS" Enter
        tmux send-keys -t "$SESSION_NAME" "export CUDA_VISIBLE_DEVICES='$CUDA_VISIBLE_DEVICES'" Enter
        
        # 分割窗口：左侧训练，右侧监控
        tmux split-window -h -t "$SESSION_NAME"
        
        # 右侧窗口启动监控
        tmux send-keys -t "$SESSION_NAME:0.1" "$MONITOR_CMD" Enter
        
        # 左侧窗口启动训练
        tmux send-keys -t "$SESSION_NAME:0.0" "$TRAIN_CMD" Enter
        
        echo "Tmux会话已创建: $SESSION_NAME"
        echo ""
        echo "管理命令:"
        echo "  tmux attach -t $SESSION_NAME         # 连接到会话"
        echo "  tmux list-sessions                   # 查看所有会话"
        echo "  Ctrl+B, D                          # 分离会话"
        echo "  tmux kill-session -t $SESSION_NAME  # 终止会话"
        echo "  Ctrl+B, %                          # 垂直分割"
        echo "  Ctrl+B, 方向键                      # 切换窗格"
        
        log_info "Tmux会话 $SESSION_NAME 启动成功"
        echo -e "${GREEN}Tmux训练会话已启动，左侧训练，右侧监控${NC}"
        ;;
    5)
        echo -e "${GREEN}系统服务模式启动...${NC}"
        log_info "启动系统服务模式"
        
        # 创建systemd服务文件
        SERVICE_NAME="pressure-training-$(date +%H%M%S)"
        SERVICE_FILE="/tmp/$SERVICE_NAME.service"
        
        cat > "$SERVICE_FILE" << EOF
[Unit]
Description=Pressure Field Training Service
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$PWD
Environment=PYTHONPATH=$PWD/pdebench_extended:\$PYTHONPATH
Environment=OMP_NUM_THREADS=$OPTIMAL_THREADS
Environment=MKL_NUM_THREADS=$OPTIMAL_THREADS
Environment=CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
ExecStart=/bin/bash -c '$TRAIN_CMD'
StandardOutput=append:$OUTPUT_DIR/training.log
StandardError=append:$OUTPUT_DIR/training.log
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF
        
        echo "系统服务配置文件已创建: $SERVICE_FILE"
        echo "由于没有sudo权限，无法直接安装系统服务。"
        echo "替代方案:"
        echo ""
        echo "1. 用户级systemd服务 (推荐):"
        USER_SERVICE_DIR="$HOME/.config/systemd/user"
        mkdir -p "$USER_SERVICE_DIR"
        USER_SERVICE_FILE="$USER_SERVICE_DIR/$SERVICE_NAME.service"
        cp "$SERVICE_FILE" "$USER_SERVICE_FILE"
        
        echo "  systemctl --user daemon-reload"
        echo "  systemctl --user start $SERVICE_NAME"
        echo "  systemctl --user enable $SERVICE_NAME  # 用户登录时自启"
        echo ""
        echo "用户服务管理命令:"
        echo "  systemctl --user status $SERVICE_NAME   # 查看状态"
        echo "  systemctl --user stop $SERVICE_NAME     # 停止服务"
        echo "  systemctl --user restart $SERVICE_NAME  # 重启服务"
        echo "  journalctl --user -u $SERVICE_NAME -f   # 查看日志"
        echo ""
        echo "2. 或者使用crontab定时任务:"
        echo "  crontab -e"
        echo "  添加: @reboot cd $PWD && bash $BACKGROUND_SCRIPT"
        echo ""
        echo "3. 或者选择其他运行模式 (推荐Screen或Tmux)"
        
        log_info "用户级服务配置完成: $SERVICE_NAME"
        echo -e "${GREEN}用户级服务文件已创建: $USER_SERVICE_FILE${NC}"
        ;;
    *)
        echo -e "${RED}无效选择${NC}"
        log_error "无效的运行模式选择: $mode"
        exit 1
        ;;
esac

# 服务器TensorBoard监控配置
echo ""
echo -e "${CYAN}=== TensorBoard监控配置 ===${NC}"
read -p "是否启动TensorBoard监控? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    log_info "配置TensorBoard监控服务"
    echo -e "${BLUE}启动服务器TensorBoard监控...${NC}"
    
    # 智能端口选择
    TB_PORT=6006
    for port in 6006 6007 6008 6009 6010; do
        if ! netstat -tlnp 2>/dev/null | grep -q ":$port "; then
            TB_PORT=$port
            break
        fi
    done
    
    if netstat -tlnp 2>/dev/null | grep -q ":$TB_PORT "; then
        echo -e "${YELLOW}所有常用端口都被占用，使用随机端口${NC}"
        TB_PORT=$((RANDOM % 1000 + 7000))
    fi
    
    # 创建TensorBoard启动脚本
    TB_SCRIPT="$OUTPUT_DIR/start_tensorboard.sh"
    cat > "$TB_SCRIPT" << EOF
#!/bin/bash
set -e

# TensorBoard配置
export TENSORBOARD_PORT=$TB_PORT
export TENSORBOARD_HOST=0.0.0.0
export TENSORBOARD_LOGDIR="$OUTPUT_BASE"

# 启动TensorBoard
echo "[\$(date)] 启动TensorBoard监控服务" >> "$LOG_DIR/tensorboard.log"
tensorboard --logdir="\$TENSORBOARD_LOGDIR" --host="\$TENSORBOARD_HOST" --port="\$TENSORBOARD_PORT" --reload_interval=30 --samples_per_plugin=1000 >> "$LOG_DIR/tensorboard.log" 2>&1
EOF
    
    chmod +x "$TB_SCRIPT"
    nohup bash "$TB_SCRIPT" &
    TB_PID=$!
    echo $TB_PID > "$OUTPUT_DIR/tensorboard.pid"
    
    # 获取服务器网络信息
    SERVER_IP=$(hostname -I | awk '{print $1}' || echo "localhost")
    EXTERNAL_IP=$(curl -s ifconfig.me 2>/dev/null || echo "未知")
    
    echo -e "${GREEN}TensorBoard监控服务已启动${NC}"
    echo "服务配置:"
    echo "  端口: $TB_PORT"
    echo "  PID: $TB_PID"
    echo "  日志: $LOG_DIR/tensorboard.log"
    echo ""
    echo "访问地址:"
    echo "  内网访问: http://$SERVER_IP:$TB_PORT"
    if [[ "$EXTERNAL_IP" != "未知" ]]; then
        echo "  外网访问: http://$EXTERNAL_IP:$TB_PORT (需要防火墙开放端口)"
    fi
    echo ""
    echo "管理命令:"
    echo "  kill $TB_PID                        # 停止TensorBoard"
    echo "  tail -f $LOG_DIR/tensorboard.log    # 查看日志"
    echo "  netstat -tlnp | grep $TB_PORT       # 检查端口状态"
    
    # 防火墙提示
    echo -e "${YELLOW}防火墙配置提示 (需要管理员权限):${NC}"
    echo "  CentOS/RHEL: firewall-cmd --add-port=$TB_PORT/tcp --permanent && firewall-cmd --reload"
    echo "  Ubuntu: ufw allow $TB_PORT"
    echo "  或者请联系系统管理员开放端口 $TB_PORT"
    
    log_info "TensorBoard服务启动完成，端口: $TB_PORT"
fi

# 服务器训练启动完成总结
echo ""
echo -e "${GREEN}=== 服务器训练启动完成 ===${NC}"
log_info "训练启动流程完成"

echo -e "${CYAN}训练配置信息:${NC}"
echo "  实验名称: $EXPERIMENT_NAME"
echo "  数据文件: $DATA_PATH"
echo "  配置文件: $CONFIG_PATH"
echo "  输出目录: $OUTPUT_DIR"
echo "  GPU设备: $CUDA_VISIBLE_DEVICES"
echo "  线程配置: $OPTIMAL_THREADS"
echo ""

echo -e "${CYAN}日志文件位置:${NC}"
echo "  脚本日志: $SCRIPT_LOG"
echo "  训练日志: $OUTPUT_DIR/training.log"
echo "  系统监控: $SYSTEM_LOG"
if [[ -f "$OUTPUT_DIR/tensorboard.pid" ]]; then
    echo "  TensorBoard日志: $LOG_DIR/tensorboard.log"
fi
echo ""

echo -e "${CYAN}服务器监控命令:${NC}"
echo "  # 系统资源监控"
echo "  htop                                    # 交互式系统监控"
echo "  top -p \$(cat $OUTPUT_DIR/train.pid 2>/dev/null || echo 1)  # 训练进程监控"
echo "  iostat -x 1                            # 磁盘I/O监控"
echo "  free -h                                # 内存使用情况"
echo "  df -h                                  # 磁盘空间"
echo ""
echo "  # GPU监控"
echo "  nvidia-smi                             # GPU状态"
echo "  watch -n 1 nvidia-smi                 # 实时GPU监控"
echo "  nvidia-smi dmon                       # GPU性能监控"
echo ""
echo "  # 训练日志监控"
echo "  tail -f $OUTPUT_DIR/training.log       # 实时训练日志"
echo "  tail -f $SYSTEM_LOG                    # 实时系统监控"
echo "  grep -i error $OUTPUT_DIR/training.log # 查找错误信息"
echo ""
echo "  # 会话管理"
echo "  screen -ls                             # 查看Screen会话"
echo "  tmux list-sessions                     # 查看Tmux会话"
echo "  ps aux | grep python                   # 查看Python进程"
echo ""

echo -e "${CYAN}故障排除命令:${NC}"
echo "  # 进程管理"
echo "  kill \$(cat $OUTPUT_DIR/train.pid 2>/dev/null)     # 停止训练"
echo "  kill \$(cat $OUTPUT_DIR/tensorboard.pid 2>/dev/null) # 停止TensorBoard"
echo "  pkill -f 'python.*train_pressure_field'          # 强制停止训练"
echo ""
echo "  # 资源清理"
echo "  nvidia-smi --gpu-reset                           # 重置GPU"
echo "  sync                                             # 同步文件系统"
echo "  # 清理系统缓存需要管理员权限: echo 3 > /proc/sys/vm/drop_caches"
echo ""
echo "  # 网络检查"
echo "  netstat -tlnp | grep 6006                        # 检查TensorBoard端口"
echo "  ss -tulpn | grep python                          # 查看Python网络连接"
echo ""

echo -e "${CYAN}性能优化建议:${NC}"
echo "  1. 监控GPU利用率，保持在80%以上"
echo "  2. 注意内存使用，避免OOM错误"
echo "  3. 定期检查磁盘空间，确保有足够存储"
echo "  4. 使用TensorBoard监控训练进度"
echo "  5. 定期保存检查点，防止意外中断"
echo ""

echo -e "${GREEN}🚀 服务器训练环境已就绪！${NC}"
echo -e "${GREEN}📊 监控地址: http://$(hostname -I | awk '{print $1}'):${TB_PORT:-6006}${NC}"
echo -e "${GREEN}📝 完整日志: $SCRIPT_LOG${NC}"
echo ""
echo -e "${YELLOW}提示: 建议定期检查训练状态和系统资源使用情况${NC}"
echo -e "${YELLOW}如遇问题，请查看日志文件或联系技术支持${NC}"

log_info "服务器训练启动流程全部完成"