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

# 检查虚拟环境
if [[ ! -d "venv" ]]; then
    echo -e "${RED}错误: 虚拟环境不存在${NC}"
    echo "请先运行: bash deploy_centos.sh"
    exit 1
fi

# 激活虚拟环境
echo -e "${BLUE}激活虚拟环境...${NC}"
source venv/bin/activate

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

# 检查数据文件
if [[ ! -f "$DATA_PATH" ]]; then
    echo -e "${YELLOW}数据文件不存在: $DATA_PATH${NC}"
    echo "正在搜索数据文件..."
    
    # 自动搜索.pt文件
    DATA_FILES=($(find . -name "*.pt" -type f 2>/dev/null | head -5))
    
    if [[ ${#DATA_FILES[@]} -eq 0 ]]; then
        echo -e "${RED}错误: 未找到数据文件${NC}"
        echo "请确保数据文件存在，或设置环境变量:"
        echo "DATA_PATH=/path/to/your/data.pt bash run_training_simple.sh"
        exit 1
    else
        echo "找到以下数据文件:"
        for i in "${!DATA_FILES[@]}"; do
            echo "  $((i+1)). ${DATA_FILES[$i]}"
        done
        
        if [[ ${#DATA_FILES[@]} -eq 1 ]]; then
            DATA_PATH="${DATA_FILES[0]}"
            echo -e "${GREEN}自动选择: $DATA_PATH${NC}"
        else
            read -p "请选择数据文件 (1-${#DATA_FILES[@]}): " choice
            DATA_PATH="${DATA_FILES[$((choice-1))]}"
        fi
    fi
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
            source venv/bin/activate
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