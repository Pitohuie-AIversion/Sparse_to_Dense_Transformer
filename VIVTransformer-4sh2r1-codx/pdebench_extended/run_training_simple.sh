#!/bin/bash

# =============================================================================
# 压力场训练 - 一键运行脚本 (简化版)
# 适用于已经部署好的系统
# =============================================================================

set -e

# 配置参数 (可根据需要修改)
DATA_PATH="${DATA_PATH:-./data/pressure_data.pt}"  # 数据文件路径
CONFIG_PATH="pdebench_extended/configs/pressure_field_training.yaml"  # 配置文件路径
OUTPUT_BASE="./outputs"  # 输出基础目录
EXPERIMENT_NAME="pressure_field_$(date +%Y%m%d_%H%M%S)"  # 实验名称

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}=== 压力场训练一键启动脚本 ===${NC}"
echo "时间: $(date)"
echo "主机: $(hostname)"
echo ""

# PyCharm环境运行 - 跳过虚拟环境检查
echo -e "${BLUE}使用PyCharm当前环境...${NC}"
echo "注意: 确保已安装所需的Python依赖包"

# 设置环境变量
export PYTHONPATH=$PWD/pdebench_extended:$PYTHONPATH
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

# GPU设置 - 适配服务器配置
if command -v nvidia-smi &> /dev/null; then
    echo -e "${GREEN}检测到GPU，分析GPU状态...${NC}"
    
    # 显示GPU状态
    echo "GPU状态信息:"
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits | while IFS=',' read -r idx name mem_used mem_total util; do
        mem_used_gb=$((mem_used / 1024))
        mem_total_gb=$((mem_total / 1024))
        mem_usage_percent=$((mem_used * 100 / mem_total))
        echo "  GPU $idx: $name - 显存: ${mem_used_gb}GB/${mem_total_gb}GB (${mem_usage_percent}%) - 利用率: ${util}%"
    done
    
    # 智能GPU选择
    echo ""
    echo "GPU选择选项:"
    echo "1. 使用GPU 0 (当前占用约18GB)"
    echo "2. 使用GPU 1 (空闲状态，推荐)"
    echo "3. 使用双GPU并行训练"
    echo "4. 自动选择最空闲的GPU"
    read -p "请选择GPU使用方式 (1-4): " gpu_choice
    
    case $gpu_choice in
        1)
            export CUDA_VISIBLE_DEVICES=0
            echo -e "${YELLOW}使用GPU 0 (注意：该GPU已有进程占用显存)${NC}"
            ;;
        2)
            export CUDA_VISIBLE_DEVICES=1
            echo -e "${GREEN}使用GPU 1 (空闲状态，性能最佳)${NC}"
            ;;
        3)
            export CUDA_VISIBLE_DEVICES=0,1
            echo -e "${BLUE}启用双GPU并行训练${NC}"
            echo -e "${YELLOW}注意：需要确保模型支持多GPU训练${NC}"
            ;;
        4)
            # 自动选择显存使用率最低的GPU
            BEST_GPU=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | sort -k2 -n | head -1 | cut -d',' -f1)
            export CUDA_VISIBLE_DEVICES=$BEST_GPU
            echo -e "${GREEN}自动选择GPU $BEST_GPU (显存使用率最低)${NC}"
            ;;
        *)
            export CUDA_VISIBLE_DEVICES=1
            echo -e "${GREEN}默认使用GPU 1${NC}"
            ;;
    esac
    
    echo "当前CUDA设备: $CUDA_VISIBLE_DEVICES"
else
    echo -e "${YELLOW}未检测到GPU，使用CPU训练${NC}"
fi

# 数据集下载和检查
echo -e "${BLUE}=== 数据集检查与下载 ===${NC}"

# 定义数据集相关变量
DATA_REPO_URL="https://github.com/pdebench/PDEBench.git"
DATA_DIR="./data"
DATA_SUBDIR="$DATA_DIR/2D/CFD/2D_Train_Rand"
ALTERNATIVE_DATA_URLS=(
    "https://darus.uni-stuttgart.de/api/access/datafile/132004"
    "https://zenodo.org/record/6222489/files/2D_CFD_Rand_M0.1_Eta1e-08_Zeta1e-08_periodic_Train.hdf5"
)

# 检查数据文件是否存在
if [[ ! -f "$DATA_PATH" ]]; then
    echo -e "${YELLOW}数据文件不存在: $DATA_PATH${NC}"
    echo "正在搜索现有数据文件..."
    
    # 自动搜索.pt和.hdf5文件
    DATA_FILES=($(find . -name "*.pt" -o -name "*.hdf5" -type f 2>/dev/null | head -10))
    
    if [[ ${#DATA_FILES[@]} -eq 0 ]]; then
        echo -e "${YELLOW}未找到现有数据文件，开始下载数据集...${NC}"
        
        # 询问用户下载方式
        echo "数据集下载选项:"
        echo "1. 从GitHub克隆完整PDEBench仓库 (推荐，包含所有数据)"
        echo "2. 直接下载压力场数据文件 (快速)"
        echo "3. 跳过下载，手动指定数据路径"
        read -p "请选择下载方式 (1-3): " download_choice
        
        case $download_choice in
            1)
                echo -e "${BLUE}克隆PDEBench仓库...${NC}"
                
                # 检查git是否安装
                if ! command -v git &> /dev/null; then
                    echo -e "${RED}错误: git未安装${NC}"
                    echo "请安装git: sudo yum install -y git"
                    exit 1
                fi
                
                # 检查git lfs
                if ! command -v git-lfs &> /dev/null; then
                    echo -e "${YELLOW}警告: git-lfs未安装，大文件可能下载失败${NC}"
                    echo "建议安装: sudo yum install -y git-lfs"
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
                            echo -e "${YELLOW}PDEBench仓库中未找到数据文件，可能需要git-lfs下载大文件${NC}"
                            echo "尝试手动下载数据文件..."
                            
                            # 提供备用下载方案
                            mkdir -p "$DATA_DIR/manual"
                            echo "正在尝试从备用源下载数据..."
                            
                            # 检查下载工具
                            if command -v wget &> /dev/null; then
                                DOWNLOAD_CMD="wget -O"
                            elif command -v curl &> /dev/null; then
                                DOWNLOAD_CMD="curl -L -o"
                            else
                                echo -e "${YELLOW}未找到下载工具，跳过自动下载${NC}"
                            fi
                            
                            if [[ -n "$DOWNLOAD_CMD" ]]; then
                                # 尝试下载示例数据文件
                                SAMPLE_FILE="$DATA_DIR/manual/sample_pressure_data.hdf5"
                                echo "尝试下载示例数据文件..."
                                
                                # 这里可以添加实际的下载链接
                                echo -e "${YELLOW}注意：需要手动下载数据文件${NC}"
                                echo "建议操作："
                                echo "1. 安装git-lfs: sudo yum install -y git-lfs"
                                echo "2. 重新下载数据: cd $DATA_DIR/PDEBench && git lfs pull"
                                echo "3. 或从以下地址手动下载:"
                                echo "   - https://darus.uni-stuttgart.de/dataset.xhtml?persistentId=doi:10.18419/darus-2986"
                                echo "   - https://github.com/pdebench/PDEBench/releases"
                            fi
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
            2)
                echo -e "${BLUE}直接下载数据文件...${NC}"
                
                # 检查wget或curl
                if command -v wget &> /dev/null; then
                    DOWNLOAD_CMD="wget -O"
                elif command -v curl &> /dev/null; then
                    DOWNLOAD_CMD="curl -L -o"
                else
                    echo -e "${RED}错误: 未找到wget或curl${NC}"
                    echo "请安装: sudo yum install -y wget"
                    exit 1
                fi
                
                mkdir -p "$DATA_DIR"
                
                # 尝试下载数据文件
                for i in "${!ALTERNATIVE_DATA_URLS[@]}"; do
                    url="${ALTERNATIVE_DATA_URLS[$i]}"
                    filename="pressure_data_$((i+1)).hdf5"
                    filepath="$DATA_DIR/$filename"
                    
                    echo "尝试下载: $url"
                    if $DOWNLOAD_CMD "$filepath" "$url"; then
                        if [[ -f "$filepath" && $(stat -f%z "$filepath" 2>/dev/null || stat -c%s "$filepath" 2>/dev/null) -gt 1000000 ]]; then
                            DATA_PATH="$filepath"
                            echo -e "${GREEN}下载成功: $DATA_PATH${NC}"
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
                    echo -e "${RED}所有下载源都失败${NC}"
                fi
                ;;
            3)
                echo -e "${YELLOW}跳过自动下载${NC}"
                ;;
            *)
                echo -e "${YELLOW}无效选择，跳过下载${NC}"
                ;;
        esac
        
        # 如果仍然没有数据文件，进行全面搜索
        if [[ ! -f "$DATA_PATH" ]]; then
            echo "重新搜索数据文件..."
            DATA_FILES=($(find . -name "*.pt" -o -name "*.hdf5" -o -name "*.h5" -type f 2>/dev/null | head -15))
            
            # 如果还是没找到，提供生成示例数据的选项
            if [[ ${#DATA_FILES[@]} -eq 0 ]]; then
                echo -e "${YELLOW}未找到任何数据文件${NC}"
                echo "选项："
                echo "1. 生成示例数据文件（用于测试）"
                echo "2. 手动指定数据文件路径"
                echo "3. 退出并手动下载数据"
                read -p "请选择 (1-3): " fallback_choice
                
                case $fallback_choice in
                    1)
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
                    2)
                        read -p "请输入数据文件的完整路径: " manual_path
                        if [[ -f "$manual_path" ]]; then
                            DATA_PATH="$manual_path"
                            echo -e "${GREEN}使用手动指定的数据文件: $DATA_PATH${NC}"
                        else
                            echo -e "${RED}指定的文件不存在: $manual_path${NC}"
                        fi
                        ;;
                    3)
                        echo -e "${YELLOW}退出脚本${NC}"
                        echo "请手动下载数据后重新运行"
                        exit 1
                        ;;
                esac
            fi
        fi
    fi
    
    # 如果找到了数据文件，让用户选择
    if [[ ${#DATA_FILES[@]} -gt 0 ]]; then
        echo "找到以下数据文件:"
        for i in "${!DATA_FILES[@]}"; do
            file_size=$(du -h "${DATA_FILES[$i]}" 2>/dev/null | cut -f1 || echo "未知")
            echo "  $((i+1)). ${DATA_FILES[$i]} (大小: $file_size)"
        done
        
        if [[ ${#DATA_FILES[@]} -eq 1 ]]; then
            DATA_PATH="${DATA_FILES[0]}"
            echo -e "${GREEN}自动选择: $DATA_PATH${NC}"
        else
            read -p "请选择数据文件 (1-${#DATA_FILES[@]}): " choice
            if [[ $choice -ge 1 && $choice -le ${#DATA_FILES[@]} ]]; then
                DATA_PATH="${DATA_FILES[$((choice-1))]}"
            else
                echo -e "${YELLOW}无效选择，使用第一个文件${NC}"
                DATA_PATH="${DATA_FILES[0]}"
            fi
        fi
    else
        echo -e "${RED}错误: 仍未找到数据文件${NC}"
        echo "请手动下载数据集或设置环境变量:"
        echo "DATA_PATH=/path/to/your/data.pt bash run_training_simple.sh"
        echo ""
        echo "数据集下载地址:"
        echo "  - PDEBench GitHub: https://github.com/pdebench/PDEBench"
        echo "  - 直接下载: https://darus.uni-stuttgart.de/dataset.xhtml?persistentId=doi:10.18419/darus-2986"
        exit 1
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

# 询问运行模式
echo "选择运行模式:"
echo "1. 前台运行 (可以看到实时输出)"
echo "2. 后台运行 (适合长时间训练)"
echo "3. Screen会话运行 (推荐)"
read -p "请选择 (1-3): " mode

# 构建训练命令
TRAIN_CMD="python pdebench_extended/train_pressure_field.py \
    --config \"$CONFIG_PATH\" \
    --data_path \"$DATA_PATH\" \
    --output_dir \"$OUTPUT_DIR\" \
    --experiment_name \"$EXPERIMENT_NAME\""

case $mode in
    1)
        echo -e "${GREEN}前台启动训练...${NC}"
        echo "按 Ctrl+C 可以停止训练"
        echo ""
        eval "$TRAIN_CMD"
        ;;
    2)
        echo -e "${GREEN}后台启动训练...${NC}"
        nohup bash -c "$TRAIN_CMD" > "$OUTPUT_DIR/training.log" 2>&1 &
        TRAIN_PID=$!
        echo $TRAIN_PID > "$OUTPUT_DIR/train.pid"
        
        echo "训练PID: $TRAIN_PID"
        echo "日志文件: $OUTPUT_DIR/training.log"
        echo ""
        echo "监控命令:"
        echo "  tail -f $OUTPUT_DIR/training.log"
        echo "停止命令:"
        echo "  kill $TRAIN_PID"
        ;;
    3)
        echo -e "${GREEN}Screen会话启动训练...${NC}"
        
        # 检查screen是否安装
        if ! command -v screen &> /dev/null; then
            echo -e "${RED}错误: screen未安装${NC}"
            echo "请安装: sudo yum install -y screen"
            exit 1
        fi
        
        SESSION_NAME="pressure_training_$(date +%H%M%S)"
        
        # 创建screen会话并运行训练
        screen -dmS "$SESSION_NAME" bash -c "
            export PYTHONPATH=$PWD/pdebench_extended:\$PYTHONPATH
            export OMP_NUM_THREADS=4
            export MKL_NUM_THREADS=4
            if command -v nvidia-smi &> /dev/null; then
                export CUDA_VISIBLE_DEVICES=\${CUDA_VISIBLE_DEVICES:-1}
            fi
            $TRAIN_CMD
            echo '训练完成，按任意键退出...'
            read
        "
        
        echo "Screen会话已创建: $SESSION_NAME"
        echo ""
        echo "管理命令:"
        echo "  screen -r $SESSION_NAME    # 连接到会话"
        echo "  screen -ls                 # 查看所有会话"
        echo "  Ctrl+A+D                  # 分离会话"
        echo "  screen -X -S $SESSION_NAME quit  # 终止会话"
        
        # 等待一下，然后显示会话状态
        sleep 2
        if screen -list | grep -q "$SESSION_NAME"; then
            echo -e "${GREEN}训练会话已启动${NC}"
        else
            echo -e "${RED}会话启动失败${NC}"
        fi
        ;;
    *)
        echo -e "${RED}无效选择${NC}"
        exit 1
        ;;
esac

# 询问是否启动TensorBoard
echo ""
read -p "是否启动TensorBoard监控? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${BLUE}启动TensorBoard...${NC}"
    
    # 检查端口是否被占用
    if netstat -tlnp 2>/dev/null | grep -q ":6006 "; then
        echo -e "${YELLOW}端口6006已被占用，尝试使用6007${NC}"
        TB_PORT=6007
    else
        TB_PORT=6006
    fi
    
    nohup tensorboard --logdir="$OUTPUT_BASE" --host=0.0.0.0 --port=$TB_PORT > tensorboard.log 2>&1 &
    TB_PID=$!
    echo $TB_PID > tensorboard.pid
    
    # 获取服务器IP
    SERVER_IP=$(hostname -I | awk '{print $1}')
    
    echo -e "${GREEN}TensorBoard已启动${NC}"
    echo "访问地址: http://$SERVER_IP:$TB_PORT"
    echo "PID: $TB_PID"
    echo "停止命令: kill $TB_PID"
fi

echo ""
echo -e "${GREEN}=== 启动完成 ===${NC}"
echo "输出目录: $OUTPUT_DIR"
echo ""
echo "常用监控命令:"
echo "  htop                       # 系统资源监控"
echo "  nvidia-smi                 # GPU监控"
echo "  tail -f $OUTPUT_DIR/training.log  # 训练日志"
echo "  screen -ls                 # 查看Screen会话"
echo ""
echo "祝训练顺利！🚀"