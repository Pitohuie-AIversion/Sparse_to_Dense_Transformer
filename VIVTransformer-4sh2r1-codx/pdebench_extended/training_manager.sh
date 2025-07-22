#!/bin/bash

# =============================================================================
# 训练管理脚本 - 统一训练管理界面
# 集成GPU监控、系统监控、训练启动、日志查看等功能
# =============================================================================

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

# 脚本路径
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPU_MONITOR_SCRIPT="$SCRIPT_DIR/gpu_monitor.sh"
SYSTEM_MONITOR_SCRIPT="$SCRIPT_DIR/system_monitor.sh"
TRAINING_SCRIPT="$SCRIPT_DIR/run_training_optimized.sh"
QUICK_START_SCRIPT="$SCRIPT_DIR/quick_start.sh"

# 配置文件
CONFIG_FILE="$SCRIPT_DIR/configs/pressure_field_training.yaml"
DATA_PATH="$SCRIPT_DIR/data/pressure_field_data.pt"

# 显示标题
show_title() {
    clear
    echo -e "${BOLD}${PURPLE}"
    echo "╔══════════════════════════════════════════════════════════════════════════════╗"
    echo "║                          训练管理系统 v1.0                                  ║"
    echo "║                     VIV Transformer 压力场重建训练                          ║"
    echo "╚══════════════════════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
    echo -e "${CYAN}服务器: node36 | GPU: 2x NVIDIA L40 | 时间: $(date)${NC}"
    echo "="*80
}

# 显示主菜单
show_main_menu() {
    echo -e "${BOLD}${BLUE}=== 主菜单 ===${NC}"
    echo ""
    echo -e "${GREEN}1.${NC} 🚀 启动训练"
    echo -e "${GREEN}2.${NC} 📊 GPU监控"
    echo -e "${GREEN}3.${NC} 🖥️  系统监控"
    echo -e "${GREEN}4.${NC} 📋 查看训练状态"
    echo -e "${GREEN}5.${NC} 📁 管理训练任务"
    echo -e "${GREEN}6.${NC} 📈 查看训练日志"
    echo -e "${GREEN}7.${NC} ⚙️  系统配置"
    echo -e "${GREEN}8.${NC} 🔧 工具箱"
    echo -e "${GREEN}9.${NC} ❓ 帮助信息"
    echo -e "${RED}0.${NC} 🚪 退出"
    echo ""
    echo -n -e "${YELLOW}请选择操作 [0-9]: ${NC}"
}

# 启动训练菜单
show_training_menu() {
    clear
    show_title
    echo -e "${BOLD}${BLUE}=== 训练启动菜单 ===${NC}"
    echo ""
    echo -e "${GREEN}1.${NC} 🎯 快速开始训练 (推荐配置)"
    echo -e "${GREEN}2.${NC} ⚡ 优化训练 (服务器优化)"
    echo -e "${GREEN}3.${NC} 🔧 自定义训练配置"
    echo -e "${GREEN}4.${NC} 📊 性能测试模式"
    echo -e "${GREEN}5.${NC} 🔄 恢复训练 (从检查点)"
    echo -e "${RED}0.${NC} 🔙 返回主菜单"
    echo ""
    echo -n -e "${YELLOW}请选择训练模式 [0-5]: ${NC}"
}

# GPU状态检查
check_gpu_status() {
    echo -e "${BLUE}=== GPU状态检查 ===${NC}"
    
    if ! command -v nvidia-smi &> /dev/null; then
        echo -e "${RED}❌ NVIDIA驱动未安装或nvidia-smi不可用${NC}"
        return 1
    fi
    
    # 显示GPU基本信息
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader,nounits | while IFS=',' read -r idx name mem_used mem_total util temp; do
        mem_used_gb=$((mem_used / 1024))
        mem_total_gb=$((mem_total / 1024))
        mem_usage_percent=$((mem_used * 100 / mem_total))
        
        # 状态颜色
        if [ $mem_usage_percent -lt 10 ] && [ $util -lt 10 ]; then
            status="${GREEN}空闲${NC}"
        elif [ $mem_usage_percent -lt 50 ] && [ $util -lt 50 ]; then
            status="${YELLOW}轻载${NC}"
        else
            status="${RED}重载${NC}"
        fi
        
        echo -e "GPU $idx ($name): ${mem_used_gb}GB/${mem_total_gb}GB, ${util}%, ${temp}°C - $status"
    done
    
    echo ""
}

# 系统状态检查
check_system_status() {
    echo -e "${CYAN}=== 系统状态检查 ===${NC}"
    
    # CPU和内存
    cpu_usage=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | awk -F'%' '{print $1}')
    mem_usage=$(free | awk '/^Mem:/ {printf "%.1f", $3/$2*100}')
    disk_usage=$(df -h . | tail -1 | awk '{print $5}')
    
    echo "CPU使用率: ${cpu_usage}%"
    echo "内存使用率: ${mem_usage}%"
    echo "磁盘使用率: $disk_usage"
    
    # 检查训练进程
    training_count=$(ps aux | grep -E '(python.*train|train_pressure_field)' | grep -v grep | wc -l)
    echo "运行中的训练进程: $training_count 个"
    
    echo ""
}

# 检查环境
check_environment() {
    echo -e "${PURPLE}=== 环境检查 ===${NC}"
    
    # Python环境
    if command -v python &> /dev/null; then
        python_version=$(python --version 2>&1)
        echo -e "${GREEN}✓${NC} Python: $python_version"
    else
        echo -e "${RED}❌ Python未安装${NC}"
    fi
    
    # PyTorch
    if python -c "import torch" 2>/dev/null; then
        torch_version=$(python -c "import torch; print(torch.__version__)" 2>/dev/null)
        cuda_available=$(python -c "import torch; print(torch.cuda.is_available())" 2>/dev/null)
        echo -e "${GREEN}✓${NC} PyTorch: $torch_version (CUDA: $cuda_available)"
    else
        echo -e "${RED}❌ PyTorch未安装${NC}"
    fi
    
    # 数据文件
    if [ -f "$DATA_PATH" ]; then
        data_size=$(du -h "$DATA_PATH" | cut -f1)
        echo -e "${GREEN}✓${NC} 数据文件: $data_size"
    else
        echo -e "${YELLOW}⚠️  数据文件未找到: $DATA_PATH${NC}"
    fi
    
    # 配置文件
    if [ -f "$CONFIG_FILE" ]; then
        echo -e "${GREEN}✓${NC} 配置文件: 存在"
    else
        echo -e "${RED}❌ 配置文件未找到: $CONFIG_FILE${NC}"
    fi
    
    echo ""
}

# 启动快速训练
start_quick_training() {
    echo -e "${GREEN}启动快速训练...${NC}"
    
    if [ -f "$QUICK_START_SCRIPT" ]; then
        chmod +x "$QUICK_START_SCRIPT"
        bash "$QUICK_START_SCRIPT"
    else
        echo -e "${RED}快速启动脚本未找到: $QUICK_START_SCRIPT${NC}"
    fi
}

# 启动优化训练
start_optimized_training() {
    echo -e "${GREEN}启动优化训练...${NC}"
    
    if [ -f "$TRAINING_SCRIPT" ]; then
        chmod +x "$TRAINING_SCRIPT"
        bash "$TRAINING_SCRIPT"
    else
        echo -e "${RED}优化训练脚本未找到: $TRAINING_SCRIPT${NC}"
    fi
}

# GPU监控
start_gpu_monitor() {
    echo -e "${GREEN}启动GPU监控...${NC}"
    
    if [ -f "$GPU_MONITOR_SCRIPT" ]; then
        chmod +x "$GPU_MONITOR_SCRIPT"
        bash "$GPU_MONITOR_SCRIPT"
    else
        echo -e "${RED}GPU监控脚本未找到: $GPU_MONITOR_SCRIPT${NC}"
        echo "使用nvidia-smi替代:"
        watch -n 2 nvidia-smi
    fi
}

# 系统监控
start_system_monitor() {
    echo -e "${GREEN}启动系统监控...${NC}"
    
    if [ -f "$SYSTEM_MONITOR_SCRIPT" ]; then
        chmod +x "$SYSTEM_MONITOR_SCRIPT"
        bash "$SYSTEM_MONITOR_SCRIPT"
    else
        echo -e "${RED}系统监控脚本未找到: $SYSTEM_MONITOR_SCRIPT${NC}"
        echo "使用htop替代:"
        htop
    fi
}

# 查看训练状态
show_training_status() {
    clear
    show_title
    echo -e "${BOLD}${BLUE}=== 训练状态 ===${NC}"
    echo ""
    
    check_gpu_status
    check_system_status
    
    # 查看训练进程详情
    echo -e "${YELLOW}=== 训练进程详情 ===${NC}"
    python_procs=$(ps aux | grep -E '(python.*train|train_pressure_field)' | grep -v grep)
    
    if [ -z "$python_procs" ]; then
        echo "当前无训练进程运行"
    else
        echo "$python_procs" | while read line; do
            pid=$(echo "$line" | awk '{print $2}')
            cpu=$(echo "$line" | awk '{print $3}')
            mem=$(echo "$line" | awk '{print $4}')
            runtime=$(ps -o etime= -p $pid 2>/dev/null | tr -d ' ')
            cmd=$(echo "$line" | awk '{for(i=11;i<=NF;i++) printf "%s ", $i; print ""}')
            
            echo "PID $pid: CPU ${cpu}%, MEM ${mem}%, 运行时间 $runtime"
            echo "命令: $(echo "$cmd" | cut -c1-70)..."
            echo ""
        done
    fi
    
    # TensorBoard状态
    tb_proc=$(ps aux | grep tensorboard | grep -v grep)
    if [ -n "$tb_proc" ]; then
        tb_port=$(echo "$tb_proc" | grep -o 'port=[0-9]*' | cut -d'=' -f2 || echo "6006")
        server_ip=$(hostname -I | awk '{print $1}')
        echo -e "${GREEN}TensorBoard运行中: http://$server_ip:$tb_port${NC}"
    fi
    
    echo ""
    echo -n "按Enter键返回主菜单..."
    read
}

# 管理训练任务
manage_training_tasks() {
    clear
    show_title
    echo -e "${BOLD}${BLUE}=== 训练任务管理 ===${NC}"
    echo ""
    
    echo -e "${GREEN}1.${NC} 🔍 查看所有训练进程"
    echo -e "${GREEN}2.${NC} ⏹️  停止训练进程"
    echo -e "${GREEN}3.${NC} 🔄 重启训练"
    echo -e "${GREEN}4.${NC} 📊 启动TensorBoard"
    echo -e "${GREEN}5.${NC} ⏹️  停止TensorBoard"
    echo -e "${GREEN}6.${NC} 🧹 清理临时文件"
    echo -e "${RED}0.${NC} 🔙 返回主菜单"
    echo ""
    echo -n -e "${YELLOW}请选择操作 [0-6]: ${NC}"
    
    read choice
    case $choice in
        1)
            echo -e "${BLUE}=== 所有训练进程 ===${NC}"
            ps aux | grep -E '(python.*train|train_pressure_field|tensorboard)' | grep -v grep
            echo ""
            echo -n "按Enter键继续..."
            read
            ;;
        2)
            echo -e "${YELLOW}请输入要停止的进程PID: ${NC}"
            read pid
            if [ -n "$pid" ]; then
                kill $pid 2>/dev/null && echo -e "${GREEN}进程 $pid 已停止${NC}" || echo -e "${RED}停止进程失败${NC}"
            fi
            sleep 2
            ;;
        3)
            echo -e "${GREEN}重启训练功能开发中...${NC}"
            sleep 2
            ;;
        4)
            echo -e "${GREEN}启动TensorBoard...${NC}"
            nohup tensorboard --logdir=outputs --port=6006 --host=0.0.0.0 > tensorboard.log 2>&1 &
            echo "TensorBoard已启动，端口6006"
            sleep 2
            ;;
        5)
            pkill -f tensorboard && echo -e "${GREEN}TensorBoard已停止${NC}" || echo -e "${RED}停止TensorBoard失败${NC}"
            sleep 2
            ;;
        6)
            echo -e "${YELLOW}清理临时文件...${NC}"
            rm -f *.log nohup.out
            echo -e "${GREEN}清理完成${NC}"
            sleep 2
            ;;
        0)
            return
            ;;
        *)
            echo -e "${RED}无效选择${NC}"
            sleep 1
            ;;
    esac
    
    manage_training_tasks
}

# 查看训练日志
view_training_logs() {
    clear
    show_title
    echo -e "${BOLD}${BLUE}=== 训练日志查看 ===${NC}"
    echo ""
    
    # 查找日志文件
    log_files=$(find . -name "*.log" -o -name "training_*.txt" -o -name "nohup.out" 2>/dev/null | head -10)
    
    if [ -z "$log_files" ]; then
        echo "未找到日志文件"
        echo -n "按Enter键返回..."
        read
        return
    fi
    
    echo "可用的日志文件:"
    echo "$log_files" | nl
    echo ""
    echo -n "请选择日志文件编号 (或按Enter返回): "
    read choice
    
    if [ -n "$choice" ] && [ "$choice" -gt 0 ]; then
        log_file=$(echo "$log_files" | sed -n "${choice}p")
        if [ -f "$log_file" ]; then
            echo -e "${GREEN}查看日志文件: $log_file${NC}"
            echo "按q退出查看"
            sleep 2
            less "$log_file"
        fi
    fi
}

# 系统配置
show_system_config() {
    clear
    show_title
    echo -e "${BOLD}${BLUE}=== 系统配置 ===${NC}"
    echo ""
    
    check_environment
    
    echo -e "${CYAN}=== 配置文件路径 ===${NC}"
    echo "训练配置: $CONFIG_FILE"
    echo "数据路径: $DATA_PATH"
    echo "输出目录: outputs/"
    echo ""
    
    echo -e "${YELLOW}=== 环境变量 ===${NC}"
    echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-未设置}"
    echo "OMP_NUM_THREADS: ${OMP_NUM_THREADS:-未设置}"
    echo "MKL_NUM_THREADS: ${MKL_NUM_THREADS:-未设置}"
    echo ""
    
    echo -n "按Enter键返回主菜单..."
    read
}

# 工具箱
show_toolbox() {
    clear
    show_title
    echo -e "${BOLD}${BLUE}=== 工具箱 ===${NC}"
    echo ""
    
    echo -e "${GREEN}1.${NC} 🔧 环境测试"
    echo -e "${GREEN}2.${NC} 📊 性能基准测试"
    echo -e "${GREEN}3.${NC} 🧹 清理所有输出"
    echo -e "${GREEN}4.${NC} 📦 打包训练结果"
    echo -e "${GREEN}5.${NC} 🔄 重置环境"
    echo -e "${GREEN}6.${NC} 📋 生成系统报告"
    echo -e "${RED}0.${NC} 🔙 返回主菜单"
    echo ""
    echo -n -e "${YELLOW}请选择工具 [0-6]: ${NC}"
    
    read choice
    case $choice in
        1)
            echo -e "${GREEN}运行环境测试...${NC}"
            python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')"
            echo -n "按Enter键继续..."
            read
            ;;
        2)
            echo -e "${GREEN}性能基准测试功能开发中...${NC}"
            sleep 2
            ;;
        3)
            echo -e "${YELLOW}清理所有输出文件...${NC}"
            rm -rf outputs/* *.log nohup.out
            echo -e "${GREEN}清理完成${NC}"
            sleep 2
            ;;
        4)
            echo -e "${GREEN}打包功能开发中...${NC}"
            sleep 2
            ;;
        5)
            echo -e "${YELLOW}重置环境功能开发中...${NC}"
            sleep 2
            ;;
        6)
            echo -e "${GREEN}生成系统报告...${NC}"
            {
                echo "=== 系统报告 ==="
                echo "时间: $(date)"
                echo ""
                echo "=== GPU信息 ==="
                nvidia-smi
                echo ""
                echo "=== 系统信息 ==="
                uname -a
                echo ""
                echo "=== 内存信息 ==="
                free -h
                echo ""
                echo "=== 磁盘信息 ==="
                df -h
            } > system_report.txt
            echo -e "${GREEN}系统报告已生成: system_report.txt${NC}"
            sleep 2
            ;;
        0)
            return
            ;;
        *)
            echo -e "${RED}无效选择${NC}"
            sleep 1
            ;;
    esac
    
    show_toolbox
}

# 显示帮助
show_help() {
    clear
    show_title
    echo -e "${BOLD}${BLUE}=== 帮助信息 ===${NC}"
    echo ""
    
    echo -e "${CYAN}=== 快速开始 ===${NC}"
    echo "1. 选择 '启动训练' -> '快速开始训练'"
    echo "2. 系统会自动检测GPU并开始训练"
    echo "3. 使用 'GPU监控' 查看训练状态"
    echo ""
    
    echo -e "${CYAN}=== 脚本文件说明 ===${NC}"
    echo "• training_manager.sh    - 主管理脚本 (当前)"
    echo "• gpu_monitor.sh         - GPU监控脚本"
    echo "• system_monitor.sh      - 系统监控脚本"
    echo "• run_training_optimized.sh - 优化训练脚本"
    echo "• quick_start.sh         - 快速启动脚本"
    echo ""
    
    echo -e "${CYAN}=== 常用命令 ===${NC}"
    echo "• nvidia-smi             - 查看GPU状态"
    echo "• htop                   - 查看系统资源"
    echo "• ps aux | grep python   - 查看Python进程"
    echo "• kill <PID>             - 终止进程"
    echo ""
    
    echo -e "${CYAN}=== 故障排除 ===${NC}"
    echo "• 训练无法启动: 检查数据文件和配置文件"
    echo "• GPU内存不足: 减小batch_size"
    echo "• 进程卡死: 使用任务管理停止进程"
    echo ""
    
    echo -n "按Enter键返回主菜单..."
    read
}

# 主循环
main_loop() {
    while true; do
        show_title
        check_gpu_status
        check_system_status
        show_main_menu
        
        read choice
        case $choice in
            1)
                show_training_menu
                read train_choice
                case $train_choice in
                    1) start_quick_training ;;
                    2) start_optimized_training ;;
                    3) echo -e "${YELLOW}自定义配置功能开发中...${NC}"; sleep 2 ;;
                    4) echo -e "${YELLOW}性能测试功能开发中...${NC}"; sleep 2 ;;
                    5) echo -e "${YELLOW}恢复训练功能开发中...${NC}"; sleep 2 ;;
                    0) continue ;;
                    *) echo -e "${RED}无效选择${NC}"; sleep 1 ;;
                esac
                ;;
            2) start_gpu_monitor ;;
            3) start_system_monitor ;;
            4) show_training_status ;;
            5) manage_training_tasks ;;
            6) view_training_logs ;;
            7) show_system_config ;;
            8) show_toolbox ;;
            9) show_help ;;
            0) 
                echo -e "${GREEN}感谢使用训练管理系统!${NC}"
                exit 0
                ;;
            *) 
                echo -e "${RED}无效选择，请重新输入${NC}"
                sleep 1
                ;;
        esac
    done
}

# 检查依赖
check_dependencies() {
    missing_deps=()
    
    # 检查必要命令
    for cmd in nvidia-smi python; do
        if ! command -v $cmd &> /dev/null; then
            missing_deps+=("$cmd")
        fi
    done
    
    if [ ${#missing_deps[@]} -gt 0 ]; then
        echo -e "${RED}缺少必要依赖: ${missing_deps[*]}${NC}"
        echo "请安装缺少的依赖后重新运行"
        exit 1
    fi
}

# 初始化
init_environment() {
    # 创建必要目录
    mkdir -p outputs logs
    
    # 设置脚本权限
    chmod +x "$GPU_MONITOR_SCRIPT" 2>/dev/null
    chmod +x "$SYSTEM_MONITOR_SCRIPT" 2>/dev/null
    chmod +x "$TRAINING_SCRIPT" 2>/dev/null
    chmod +x "$QUICK_START_SCRIPT" 2>/dev/null
}

# 主程序入口
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    # 检查依赖
    check_dependencies
    
    # 初始化环境
    init_environment
    
    # 启动主循环
    main_loop
fi