#!/bin/bash

# =============================================================================
# 压力场训练系统 - 快速启动脚本
# 用于已经部署好的系统快速启动训练
# =============================================================================

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# 日志函数
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查虚拟环境
check_venv() {
    if [[ ! -d "venv" ]]; then
        log_error "虚拟环境不存在，请先运行 deploy_centos.sh"
        exit 1
    fi
    
    log_info "激活虚拟环境..."
    source venv/bin/activate
    log_success "虚拟环境已激活"
}

# 检查项目文件
check_project_files() {
    log_info "检查项目文件..."
    
    required_files=(
        "pdebench_extended/train_pressure_field.py"
        "pdebench_extended/configs/pressure_field_training.yaml"
        "pdebench_extended/data/pressure_field_adapter.py"
    )
    
    for file in "${required_files[@]}"; do
        if [[ ! -f "$file" ]]; then
            log_error "缺少文件: $file"
            exit 1
        fi
    done
    
    log_success "项目文件检查完成"
}

# 设置环境变量
setup_env() {
    log_info "设置环境变量..."
    
    export PYTHONPATH=$PWD/pdebench_extended:$PYTHONPATH
    export OMP_NUM_THREADS=8
    export MKL_NUM_THREADS=8
    
    # GPU智能选择 - 适配node36服务器
    if command -v nvidia-smi &> /dev/null; then
        log_info "检测GPU状态..."
        
        # 获取GPU使用情况
        gpu_status=$(nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits)
        
        echo "GPU状态:"
        while IFS=',' read -r idx mem_used mem_total; do
            mem_usage_percent=$((mem_used * 100 / mem_total))
            mem_used_gb=$((mem_used / 1024))
            mem_total_gb=$((mem_total / 1024))
            
            if [ $mem_usage_percent -lt 10 ]; then
                status="${GREEN}空闲${NC}"
            elif [ $mem_usage_percent -lt 50 ]; then
                status="${YELLOW}使用中${NC}"
            else
                status="${RED}繁忙${NC}"
            fi
            
            echo -e "  GPU $idx: ${mem_used_gb}GB/${mem_total_gb}GB (${mem_usage_percent}%) - $status"
        done <<< "$gpu_status"
        
        # 自动选择最空闲的GPU
        BEST_GPU=$(echo "$gpu_status" | awk -F',' '{print $1, $2}' | sort -k2 -n | head -1 | cut -d' ' -f1)
        export CUDA_VISIBLE_DEVICES=$BEST_GPU
        
        log_success "自动选择GPU $BEST_GPU (显存使用最少)"
    else
        log_warning "未检测到GPU，使用CPU训练"
    fi
    
    log_success "环境变量设置完成"
}

# 检查和下载数据文件
check_data() {
    log_info "检查数据文件..."
    
    # 定义数据集相关变量
    DATA_REPO_URL="https://github.com/pdebench/PDEBench.git"
    DATA_DIR="./data"
    ALTERNATIVE_DATA_URLS=(
        "https://darus.uni-stuttgart.de/api/access/datafile/132004"
        "https://zenodo.org/record/6222489/files/2D_CFD_Rand_M0.1_Eta1e-08_Zeta1e-08_periodic_Train.hdf5"
    )
    
    if [[ -z "$DATA_PATH" ]]; then
        # 尝试自动查找数据文件
        DATA_FILES=($(find . -name "*.pt" -o -name "*.hdf5" -o -name "*.h5" -type f 2>/dev/null))
        
        if [[ ${#DATA_FILES[@]} -eq 0 ]]; then
            log_warning "未找到现有数据文件，启动快速下载..."
            
            echo "快速下载选项:"
            echo "1. 快速克隆PDEBench仓库 (推荐)"
            echo "2. 直接下载数据文件"
            echo "3. 生成示例数据 (用于测试)"
            echo "4. 手动指定路径"
            read -p "请选择 (1-4): " download_choice
            
            case $download_choice in
                1)
                    log_info "快速克隆PDEBench仓库..."
                    
                    if ! command -v git &> /dev/null; then
                        log_error "git未安装，请先安装git"
                        exit 1
                    fi
                    
                    mkdir -p "$DATA_DIR"
                    
                    if [[ ! -d "$DATA_DIR/PDEBench" ]]; then
                        log_info "正在克隆仓库..."
                        if git clone --depth 1 "$DATA_REPO_URL" "$DATA_DIR/PDEBench"; then
                            log_success "仓库克隆成功"
                            
                            # 搜索数据文件
                            DOWNLOADED_FILES=($(find "$DATA_DIR/PDEBench" -name "*.hdf5" -o -name "*.pt" -o -name "*.h5" 2>/dev/null | head -5))
                            if [[ ${#DOWNLOADED_FILES[@]} -gt 0 ]]; then
                                DATA_PATH="${DOWNLOADED_FILES[0]}"
                                log_success "找到数据文件: $DATA_PATH"
                            fi
                        else
                            log_error "仓库克隆失败"
                        fi
                    else
                        log_info "PDEBench仓库已存在"
                        DOWNLOADED_FILES=($(find "$DATA_DIR/PDEBench" -name "*.hdf5" -o -name "*.pt" -o -name "*.h5" 2>/dev/null | head -5))
                        if [[ ${#DOWNLOADED_FILES[@]} -gt 0 ]]; then
                            DATA_PATH="${DOWNLOADED_FILES[0]}"
                            log_success "使用现有数据文件: $DATA_PATH"
                        fi
                    fi
                    ;;
                2)
                    log_info "直接下载数据文件..."
                    
                    if command -v wget &> /dev/null; then
                        DOWNLOAD_CMD="wget -O"
                    elif command -v curl &> /dev/null; then
                        DOWNLOAD_CMD="curl -L -o"
                    else
                        log_error "未找到wget或curl，无法下载"
                        exit 1
                    fi
                    
                    mkdir -p "$DATA_DIR"
                    
                    for i in "${!ALTERNATIVE_DATA_URLS[@]}"; do
                        url="${ALTERNATIVE_DATA_URLS[$i]}"
                        filename="pressure_data_$((i+1)).hdf5"
                        filepath="$DATA_DIR/$filename"
                        
                        log_info "尝试下载: $url"
                        if $DOWNLOAD_CMD "$filepath" "$url"; then
                            if [[ -f "$filepath" && $(stat -c%s "$filepath" 2>/dev/null || stat -f%z "$filepath" 2>/dev/null) -gt 1000000 ]]; then
                                DATA_PATH="$filepath"
                                log_success "下载成功: $DATA_PATH"
                                break
                            else
                                log_warning "下载的文件太小，尝试下一个源"
                                rm -f "$filepath"
                            fi
                        else
                            log_warning "下载失败，尝试下一个源"
                        fi
                    done
                    ;;
                3)
                    log_info "生成示例数据..."
                    
                    mkdir -p "$DATA_DIR"
                    
                    # 创建简单的示例数据生成脚本
                    cat > "$DATA_DIR/generate_quick_sample.py" << 'EOF'
import torch
import numpy as np
from pathlib import Path

def generate_quick_sample():
    print("生成快速示例数据...")
    
    # 创建小规模示例数据
    batch_size = 50
    height, width = 32, 32
    time_steps = 5
    
    # 生成压力场数据
    pressure_data = torch.randn(batch_size, time_steps, height, width, dtype=torch.float32)
    
    # 添加边界条件
    pressure_data[:, :, 0, :] = 0  # 上边界
    pressure_data[:, :, -1, :] = 0  # 下边界
    pressure_data[:, :, :, 0] = 0  # 左边界
    pressure_data[:, :, :, -1] = 0  # 右边界
    
    # 保存为PyTorch格式
    output_file = Path("quick_sample_data.pt")
    torch.save({
        'pressure': pressure_data,
        'time': torch.linspace(0, 1, time_steps),
        'metadata': {
            'description': 'Quick sample pressure field data',
            'batch_size': batch_size,
            'spatial_dims': (height, width),
            'time_steps': time_steps
        }
    }, output_file)
    
    print(f"示例数据已生成: {output_file}")
    print(f"数据形状: {pressure_data.shape}")
    return str(output_file)

if __name__ == "__main__":
    generate_quick_sample()
EOF
                    
                    cd "$DATA_DIR"
                    if python generate_quick_sample.py; then
                        DATA_PATH="$DATA_DIR/quick_sample_data.pt"
                        log_success "示例数据生成成功: $DATA_PATH"
                        log_warning "注意: 这是示例数据，仅用于测试"
                    else
                        log_error "示例数据生成失败"
                    fi
                    cd - > /dev/null
                    ;;
                4)
                    read -p "请输入数据文件路径: " manual_path
                    if [[ -f "$manual_path" ]]; then
                        DATA_PATH="$manual_path"
                        log_success "使用手动指定的数据文件: $DATA_PATH"
                    else
                        log_error "指定的文件不存在: $manual_path"
                        exit 1
                    fi
                    ;;
                *)
                    log_error "无效选择"
                    exit 1
                    ;;
            esac
            
            # 如果仍然没有数据文件，再次搜索
            if [[ ! -f "$DATA_PATH" ]]; then
                log_info "重新搜索数据文件..."
                DATA_FILES=($(find . -name "*.pt" -o -name "*.hdf5" -o -name "*.h5" -type f 2>/dev/null))
            fi
        fi
        
        # 如果找到了数据文件，让用户选择
        if [[ ${#DATA_FILES[@]} -gt 0 && -z "$DATA_PATH" ]]; then
            if [[ ${#DATA_FILES[@]} -eq 1 ]]; then
                DATA_PATH="${DATA_FILES[0]}"
                log_info "自动检测到数据文件: $DATA_PATH"
            else
                log_warning "检测到多个数据文件:"
                for i in "${!DATA_FILES[@]}"; do
                    file_size=$(du -h "${DATA_FILES[$i]}" 2>/dev/null | cut -f1 || echo "未知")
                    echo "  $((i+1)). ${DATA_FILES[$i]} (大小: $file_size)"
                done
                read -p "请选择数据文件 (1-${#DATA_FILES[@]}): " choice
                if [[ $choice -ge 1 && $choice -le ${#DATA_FILES[@]} ]]; then
                    DATA_PATH="${DATA_FILES[$((choice-1))]}"
                else
                    DATA_PATH="${DATA_FILES[0]}"
                    log_warning "无效选择，使用第一个文件"
                fi
            fi
        fi
    fi
    
    # 最终验证
    if [[ ! -f "$DATA_PATH" ]]; then
        log_error "数据文件不存在: $DATA_PATH"
        echo "请手动下载数据集或设置环境变量:"
        echo "DATA_PATH=/path/to/data.pt ./quick_start.sh"
        exit 1
    fi
    
    # 显示数据文件信息
    file_size=$(du -h "$DATA_PATH" 2>/dev/null | cut -f1 || echo "未知")
    log_success "数据文件: $DATA_PATH (大小: $file_size)"
}

# 创建输出目录
setup_output() {
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    OUTPUT_DIR="./outputs/pressure_field_$TIMESTAMP"
    
    mkdir -p "$OUTPUT_DIR"
    log_success "输出目录: $OUTPUT_DIR"
}

# 显示系统信息
show_system_info() {
    log_info "=== 系统信息 ==="
    echo "时间: $(date)"
    echo "主机: $(hostname)"
    echo "用户: $(whoami)"
    echo "工作目录: $(pwd)"
    echo "Python版本: $(python --version)"
    
    if command -v nvidia-smi &> /dev/null; then
        echo "GPU信息:"
        nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader,nounits | while read line; do
            echo "  GPU $line"
        done
    else
        echo "GPU: 未检测到"
    fi
    
    echo "CPU核心数: $(nproc)"
    echo "内存: $(free -h | awk '/^Mem:/ {print $2}')"
    echo ""
}

# 启动训练
start_training() {
    log_info "=== 开始训练 ==="
    
    # 构建训练命令
    TRAIN_CMD="python pdebench_extended/train_pressure_field.py \
        --config pdebench_extended/configs/pressure_field_training.yaml \
        --data_path \"$DATA_PATH\" \
        --output_dir \"$OUTPUT_DIR\" \
        --experiment_name \"pressure_field_$TIMESTAMP\""
    
    log_info "训练命令:"
    echo "$TRAIN_CMD"
    echo ""
    
    # 询问是否在后台运行
    read -p "是否在后台运行? (y/n): " -n 1 -r
    echo
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        # 后台运行
        log_info "在后台启动训练..."
        nohup bash -c "$TRAIN_CMD" > "$OUTPUT_DIR/training.log" 2>&1 &
        TRAIN_PID=$!
        echo $TRAIN_PID > "$OUTPUT_DIR/train.pid"
        
        log_success "训练已在后台启动 (PID: $TRAIN_PID)"
        log_info "日志文件: $OUTPUT_DIR/training.log"
        log_info "查看日志: tail -f $OUTPUT_DIR/training.log"
        log_info "停止训练: kill $TRAIN_PID"
    else
        # 前台运行
        log_info "在前台启动训练..."
        eval "$TRAIN_CMD"
    fi
}

# 启动TensorBoard
start_tensorboard() {
    read -p "是否启动TensorBoard? (y/n): " -n 1 -r
    echo
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        log_info "启动TensorBoard..."
        nohup tensorboard --logdir=./outputs --host=0.0.0.0 --port=6006 > tensorboard.log 2>&1 &
        TB_PID=$!
        echo $TB_PID > tensorboard.pid
        
        log_success "TensorBoard已启动 (PID: $TB_PID)"
        log_info "访问地址: http://$(hostname -I | awk '{print $1}'):6006"
        log_info "停止TensorBoard: kill $TB_PID"
    fi
}

# 显示监控信息
show_monitoring() {
    echo ""
    log_info "=== 监控命令 ==="
    echo "1. 查看训练日志:"
    echo "   tail -f $OUTPUT_DIR/training.log"
    echo ""
    echo "2. 查看GPU使用情况:"
    echo "   watch -n 1 nvidia-smi"
    echo ""
    echo "3. 查看系统资源:"
    echo "   htop"
    echo ""
    echo "4. 查看训练进程:"
    echo "   ps aux | grep python"
    echo ""
    echo "5. 停止所有相关进程:"
    echo "   pkill -f train_pressure_field"
    echo "   pkill -f tensorboard"
}

# 主菜单
show_menu() {
    echo ""
    echo "=== 压力场训练系统快速启动 ==="
    echo "1. 开始训练"
    echo "2. 仅启动TensorBoard"
    echo "3. 运行测试"
    echo "4. 查看系统状态"
    echo "5. 退出"
    echo ""
    read -p "请选择操作 (1-5): " choice
    
    case $choice in
        1)
            check_data
            setup_output
            start_training
            start_tensorboard
            show_monitoring
            ;;
        2)
            start_tensorboard
            ;;
        3)
            log_info "运行测试..."
            python pdebench_extended/examples/pressure_field_example.py
            ;;
        4)
            show_system_info
            if [[ -f "./monitor.sh" ]]; then
                ./monitor.sh
            fi
            ;;
        5)
            log_info "退出"
            exit 0
            ;;
        *)
            log_error "无效选择"
            show_menu
            ;;
    esac
}

# 主函数
main() {
    echo "=== 压力场训练系统快速启动脚本 ==="
    echo ""
    
    check_venv
    check_project_files
    setup_env
    show_system_info
    
    # 如果提供了参数，直接执行
    if [[ $# -gt 0 ]]; then
        case $1 in
            "train")
                check_data
                setup_output
                start_training
                ;;
            "tensorboard")
                start_tensorboard
                ;;
            "test")
                python pdebench_extended/examples/pressure_field_example.py
                ;;
            "monitor")
                if [[ -f "./monitor.sh" ]]; then
                    ./monitor.sh
                fi
                ;;
            *)
                log_error "未知参数: $1"
                echo "可用参数: train, tensorboard, test, monitor"
                exit 1
                ;;
        esac
    else
        show_menu
    fi
}

# 执行主函数
main "$@"