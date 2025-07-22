#!/bin/bash

# =============================================================================
# 压力场训练 - 服务器优化版脚本
# 专门针对 node36 服务器配置优化
# GPU配置: 2x NVIDIA L40 (46GB each)
# CUDA版本: 12.3, 驱动版本: 545.23.06
# =============================================================================

set -e

# 配置参数
DATA_PATH="${DATA_PATH:-./data/pressure_data.pt}"
CONFIG_PATH="pdebench_extended/configs/pressure_field_training.yaml"
OUTPUT_BASE="./outputs"
EXPERIMENT_NAME="pressure_field_$(date +%Y%m%d_%H%M%S)"

# 服务器特定优化参数
SERVER_NAME="node36"
TOTAL_GPU_MEMORY=46068  # MB per GPU
OPTIMAL_BATCH_SIZE=32   # 根据46GB显存优化
NUM_WORKERS=8           # 根据服务器CPU核心数优化

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m'

echo -e "${PURPLE}=== 压力场训练 - 服务器优化版 ===${NC}"
echo "服务器: $SERVER_NAME"
echo "时间: $(date)"
echo "用户: $(whoami)"
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

# 服务器优化环境变量设置
echo -e "${BLUE}设置服务器优化参数...${NC}"
export PYTHONPATH=$PWD/pdebench_extended:$PYTHONPATH

# CPU优化 - 根据服务器配置
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8

# CUDA优化设置
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_CACHE_PATH=/tmp/cuda_cache_$(whoami)
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# 内存优化
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512,roundup_power2_divisions:16

# 详细GPU状态分析
analyze_gpu_status() {
    echo -e "${BLUE}=== GPU状态详细分析 ===${NC}"
    
    # 基本GPU信息
    nvidia-smi --query-gpu=index,name,driver_version,memory.total,memory.free,memory.used,utilization.gpu,temperature.gpu,power.draw,power.limit --format=csv,noheader,nounits | while IFS=',' read -r idx name driver mem_total mem_free mem_used util temp power power_limit; do
        mem_total_gb=$((mem_total / 1024))
        mem_free_gb=$((mem_free / 1024))
        mem_used_gb=$((mem_used / 1024))
        mem_usage_percent=$((mem_used * 100 / mem_total))
        
        echo "GPU $idx: $name"
        echo "  驱动版本: $driver"
        echo "  显存: ${mem_used_gb}GB/${mem_total_gb}GB (${mem_usage_percent}% 已用, ${mem_free_gb}GB 可用)"
        echo "  利用率: ${util}%"
        echo "  温度: ${temp}°C"
        echo "  功耗: ${power}W/${power_limit}W"
        
        # 状态评估
        if [ $mem_usage_percent -lt 10 ]; then
            echo -e "  状态: ${GREEN}空闲 (推荐使用)${NC}"
        elif [ $mem_usage_percent -lt 50 ]; then
            echo -e "  状态: ${YELLOW}轻度使用${NC}"
        else
            echo -e "  状态: ${RED}重度使用 (不推荐)${NC}"
        fi
        echo ""
    done
    
    # 运行中的进程
    echo "GPU进程信息:"
    nvidia-smi --query-compute-apps=pid,process_name,gpu_uuid,used_memory --format=csv,noheader,nounits 2>/dev/null | while IFS=',' read -r pid process gpu_uuid mem; do
        gpu_id=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | grep "$gpu_uuid" | cut -d',' -f1)
        mem_gb=$((mem / 1024))
        echo "  GPU $gpu_id: PID $pid - $process (${mem_gb}GB)"
    done || echo "  无运行中的GPU进程"
}

# GPU智能选择
select_optimal_gpu() {
    echo -e "${BLUE}=== 智能GPU选择 ===${NC}"
    
    # 获取GPU使用情况
    gpu_info=$(nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits)
    
    echo "可用选项:"
    echo "1. GPU 0 (自动检测状态)"
    echo "2. GPU 1 (自动检测状态)"
    echo "3. 双GPU并行训练 (实验性)"
    echo "4. 智能自动选择"
    echo "5. 手动指定"
    
    # 显示每个GPU的推荐度
    echo ""
    echo "GPU推荐度分析:"
    while IFS=',' read -r idx mem_used mem_total util; do
        mem_usage_percent=$((mem_used * 100 / mem_total))
        
        if [ $mem_usage_percent -lt 5 ] && [ $util -lt 5 ]; then
            recommendation="${GREEN}强烈推荐${NC}"
        elif [ $mem_usage_percent -lt 20 ] && [ $util -lt 20 ]; then
            recommendation="${YELLOW}推荐${NC}"
        else
            recommendation="${RED}不推荐${NC}"
        fi
        
        echo -e "  GPU $idx: 显存使用${mem_usage_percent}%, 利用率${util}% - $recommendation"
    done <<< "$gpu_info"
    
    echo ""
    read -p "请选择GPU使用方式 (1-5): " gpu_choice
    
    case $gpu_choice in
        1)
            export CUDA_VISIBLE_DEVICES=0
            echo -e "${BLUE}选择GPU 0${NC}"
            ;;
        2)
            export CUDA_VISIBLE_DEVICES=1
            echo -e "${BLUE}选择GPU 1${NC}"
            ;;
        3)
            export CUDA_VISIBLE_DEVICES=0,1
            echo -e "${PURPLE}启用双GPU并行训练${NC}"
            echo -e "${YELLOW}注意: 将自动调整批处理大小${NC}"
            OPTIMAL_BATCH_SIZE=$((OPTIMAL_BATCH_SIZE * 2))
            ;;
        4)
            # 智能选择：选择显存使用率最低且利用率最低的GPU
            BEST_GPU=$(echo "$gpu_info" | awk -F',' '{print $1, ($2*100/$3 + $4*2)}' | sort -k2 -n | head -1 | cut -d' ' -f1)
            export CUDA_VISIBLE_DEVICES=$BEST_GPU
            echo -e "${GREEN}智能选择GPU $BEST_GPU${NC}"
            ;;
        5)
            read -p "请输入GPU ID (0,1 或 0,1): " manual_gpu
            export CUDA_VISIBLE_DEVICES=$manual_gpu
            echo -e "${BLUE}手动选择GPU: $manual_gpu${NC}"
            ;;
        *)
            # 默认选择显存使用最少的GPU
            DEFAULT_GPU=$(echo "$gpu_info" | awk -F',' '{print $1, $2}' | sort -k2 -n | head -1 | cut -d' ' -f1)
            export CUDA_VISIBLE_DEVICES=$DEFAULT_GPU
            echo -e "${GREEN}默认选择GPU $DEFAULT_GPU${NC}"
            ;;
    esac
    
    echo "当前CUDA设备: $CUDA_VISIBLE_DEVICES"
}

# 检查并分析GPU
if command -v nvidia-smi &> /dev/null; then
    analyze_gpu_status
    select_optimal_gpu
    
    # 根据GPU配置优化批处理大小
    if [[ "$CUDA_VISIBLE_DEVICES" == *","* ]]; then
        echo -e "${BLUE}检测到多GPU配置，优化批处理大小${NC}"
        OPTIMAL_BATCH_SIZE=$((OPTIMAL_BATCH_SIZE * 2))
    fi
else
    echo -e "${RED}未检测到GPU，使用CPU训练${NC}"
    OPTIMAL_BATCH_SIZE=8  # CPU模式使用较小批处理
fi

# 数据文件检查和选择
echo -e "${BLUE}=== 数据文件检查 ===${NC}"
if [[ ! -f "$DATA_PATH" ]]; then
    echo -e "${YELLOW}数据文件不存在: $DATA_PATH${NC}"
    echo "搜索可用的数据文件..."
    
    # 搜索多种数据文件格式
    DATA_FILES=($(find . -name "*.pt" -o -name "*.h5" -o -name "*.hdf5" -o -name "*.npz" | head -10))
    
    if [[ ${#DATA_FILES[@]} -eq 0 ]]; then
        echo -e "${RED}错误: 未找到数据文件${NC}"
        echo "支持的格式: .pt, .h5, .hdf5, .npz"
        echo "请确保数据文件存在，或设置环境变量:"
        echo "DATA_PATH=/path/to/your/data.pt bash run_training_optimized.sh"
        exit 1
    else
        echo "找到以下数据文件:"
        for i in "${!DATA_FILES[@]}"; do
            file_size=$(du -h "${DATA_FILES[$i]}" | cut -f1)
            echo "  $((i+1)). ${DATA_FILES[$i]} (${file_size})"
        done
        
        if [[ ${#DATA_FILES[@]} -eq 1 ]]; then
            DATA_PATH="${DATA_FILES[0]}"
            echo -e "${GREEN}自动选择: $DATA_PATH${NC}"
        else
            read -p "请选择数据文件 (1-${#DATA_FILES[@]}): " choice
            if [[ $choice -ge 1 && $choice -le ${#DATA_FILES[@]} ]]; then
                DATA_PATH="${DATA_FILES[$((choice-1))]}"
            else
                echo -e "${RED}无效选择，使用第一个文件${NC}"
                DATA_PATH="${DATA_FILES[0]}"
            fi
        fi
    fi
fi

# 配置文件检查
if [[ ! -f "$CONFIG_PATH" ]]; then
    echo -e "${RED}错误: 配置文件不存在: $CONFIG_PATH${NC}"
    exit 1
fi

# 创建优化的输出目录
OUTPUT_DIR="$OUTPUT_BASE/$EXPERIMENT_NAME"
mkdir -p "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR/checkpoints"
mkdir -p "$OUTPUT_DIR/logs"
mkdir -p "$OUTPUT_DIR/visualizations"

# 保存运行配置
cat > "$OUTPUT_DIR/run_config.txt" << EOF
=== 训练运行配置 ===
服务器: $SERVER_NAME
时间: $(date)
用户: $(whoami)
GPU设备: $CUDA_VISIBLE_DEVICES
数据文件: $DATA_PATH
配置文件: $CONFIG_PATH
输出目录: $OUTPUT_DIR
实验名称: $EXPERIMENT_NAME
优化批处理大小: $OPTIMAL_BATCH_SIZE
CPU线程数: $OMP_NUM_THREADS
EOF

echo ""
echo -e "${BLUE}=== 训练配置总结 ===${NC}"
cat "$OUTPUT_DIR/run_config.txt"
echo ""

# 训练模式选择
echo "选择训练模式:"
echo "1. 前台运行 (可看到实时输出)"
echo "2. 后台运行 (适合长时间训练)"
echo "3. Screen会话运行 (推荐，可分离)"
echo "4. 性能测试模式 (短时间测试)"
read -p "请选择 (1-4): " mode

# 构建优化的训练命令
TRAIN_CMD="python pdebench_extended/train_pressure_field.py \\
    --config \"$CONFIG_PATH\" \\
    --data_path \"$DATA_PATH\" \\
    --output_dir \"$OUTPUT_DIR\" \\
    --experiment_name \"$EXPERIMENT_NAME\" \\
    --batch_size $OPTIMAL_BATCH_SIZE \\
    --num_workers $NUM_WORKERS"

# 性能测试模式的特殊参数
if [[ $mode -eq 4 ]]; then
    TRAIN_CMD="$TRAIN_CMD --max_epochs 2 --save_freq 1"
    echo -e "${YELLOW}性能测试模式: 仅运行2个epoch${NC}"
fi

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
        echo "  watch -n 1 nvidia-smi"
        echo "停止命令:"
        echo "  kill $TRAIN_PID"
        ;;
    3)
        echo -e "${GREEN}Screen会话启动训练...${NC}"
        
        if ! command -v screen &> /dev/null; then
            echo -e "${RED}错误: screen未安装${NC}"
            echo "请安装: sudo yum install -y screen"
            exit 1
        fi
        
        SESSION_NAME="pressure_training_$(date +%H%M%S)"
        
        # 创建优化的screen会话
        screen -dmS "$SESSION_NAME" bash -c "
            source venv/bin/activate
            export PYTHONPATH=$PWD/pdebench_extended:\$PYTHONPATH
            export OMP_NUM_THREADS=$OMP_NUM_THREADS
            export MKL_NUM_THREADS=$MKL_NUM_THREADS
            export CUDA_VISIBLE_DEVICES=\${CUDA_VISIBLE_DEVICES}
            export PYTORCH_CUDA_ALLOC_CONF=\${PYTORCH_CUDA_ALLOC_CONF}
            echo '=== 训练开始于 \$(date) ==='
            $TRAIN_CMD
            echo '=== 训练完成于 \$(date) ==='
            echo '按任意键退出...'
            read
        "
        
        echo "Screen会话已创建: $SESSION_NAME"
        echo ""
        echo "管理命令:"
        echo "  screen -r $SESSION_NAME    # 连接到会话"
        echo "  screen -ls                 # 查看所有会话"
        echo "  Ctrl+A+D                  # 分离会话"
        echo "  screen -X -S $SESSION_NAME quit  # 终止会话"
        
        sleep 2
        if screen -list | grep -q "$SESSION_NAME"; then
            echo -e "${GREEN}训练会话已启动${NC}"
        else
            echo -e "${RED}会话启动失败${NC}"
        fi
        ;;
    4)
        echo -e "${BLUE}性能测试模式启动...${NC}"
        echo "这将运行一个短时间的测试来验证配置"
        echo ""
        eval "$TRAIN_CMD"
        ;;
    *)
        echo -e "${RED}无效选择${NC}"
        exit 1
        ;;
esac

# TensorBoard启动
echo ""
read -p "是否启动TensorBoard监控? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${BLUE}启动TensorBoard...${NC}"
    
    # 智能端口选择
    TB_PORT=6006
    for port in 6006 6007 6008 6009; do
        if ! netstat -tlnp 2>/dev/null | grep -q ":$port "; then
            TB_PORT=$port
            break
        fi
    done
    
    nohup tensorboard --logdir="$OUTPUT_BASE" --host=0.0.0.0 --port=$TB_PORT --reload_interval=30 > "$OUTPUT_DIR/tensorboard.log" 2>&1 &
    TB_PID=$!
    echo $TB_PID > "$OUTPUT_DIR/tensorboard.pid"
    
    # 获取服务器IP
    SERVER_IP=$(hostname -I | awk '{print $1}')
    
    echo -e "${GREEN}TensorBoard已启动${NC}"
    echo "访问地址: http://$SERVER_IP:$TB_PORT"
    echo "PID: $TB_PID"
    echo "日志: $OUTPUT_DIR/tensorboard.log"
    echo "停止命令: kill $TB_PID"
fi

# 系统监控脚本
cat > "$OUTPUT_DIR/monitor.sh" << 'EOF'
#!/bin/bash
# 实时监控脚本
echo "=== 训练监控面板 ==="
echo "时间: $(date)"
echo ""

echo "=== GPU状态 ==="
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw --format=csv,noheader,nounits | while IFS=',' read -r idx name util mem_used mem_total temp power; do
    mem_gb=$((mem_used / 1024))
    total_gb=$((mem_total / 1024))
    echo "GPU $idx: ${util}% 利用率, ${mem_gb}GB/${total_gb}GB 显存, ${temp}°C, ${power}W"
done

echo ""
echo "=== 训练进程 ==="
ps aux | grep python | grep train_pressure_field | grep -v grep || echo "无训练进程"

echo ""
echo "=== 系统资源 ==="
echo "CPU: $(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | awk -F'%' '{print $1}')% 使用率"
echo "内存: $(free | grep Mem | awk '{printf "%.1f%%", $3/$2 * 100.0}')"
echo "磁盘: $(df -h . | tail -1 | awk '{print $5}') 已用"
EOF

chmod +x "$OUTPUT_DIR/monitor.sh"

echo ""
echo -e "${GREEN}=== 启动完成 ===${NC}"
echo "输出目录: $OUTPUT_DIR"
echo ""
echo "常用监控命令:"
echo "  $OUTPUT_DIR/monitor.sh      # 快速状态检查"
echo "  watch -n 5 '$OUTPUT_DIR/monitor.sh'  # 持续监控"
echo "  tail -f $OUTPUT_DIR/training.log     # 训练日志"
echo "  htop                                 # 系统资源"
echo "  nvidia-smi -l 1                     # GPU实时监控"
echo ""
echo -e "${PURPLE}祝您在 $SERVER_NAME 上训练顺利！🚀${NC}"