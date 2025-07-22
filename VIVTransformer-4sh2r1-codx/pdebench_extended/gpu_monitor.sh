#!/bin/bash

# =============================================================================
# GPU监控脚本 - 专为node36服务器优化
# 实时监控GPU状态、进程和性能
# =============================================================================

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m'

# 清屏函数
clear_screen() {
    clear
    echo -e "${PURPLE}=== GPU监控面板 - node36服务器 ===${NC}"
    echo "时间: $(date)"
    echo "更新间隔: ${UPDATE_INTERVAL}秒 (按Ctrl+C退出)"
    echo "="*80
}

# GPU基本信息
show_gpu_info() {
    echo -e "${BLUE}=== GPU硬件信息 ===${NC}"
    nvidia-smi --query-gpu=index,name,driver_version,cuda_version,memory.total --format=csv,noheader,nounits | while IFS=',' read -r idx name driver cuda mem_total; do
        mem_total_gb=$((mem_total / 1024))
        echo "GPU $idx: $name"
        echo "  驱动版本: $driver, CUDA: $cuda, 总显存: ${mem_total_gb}GB"
    done
    echo ""
}

# GPU实时状态
show_gpu_status() {
    echo -e "${CYAN}=== GPU实时状态 ===${NC}"
    printf "%-5s %-15s %-8s %-15s %-8s %-8s %-8s\n" "GPU" "显存使用" "利用率" "温度" "功耗" "风扇" "状态"
    echo "─"*80
    
    nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw,power.limit,fan.speed --format=csv,noheader,nounits | while IFS=',' read -r idx mem_used mem_total util temp power power_limit fan; do
        mem_used_gb=$((mem_used / 1024))
        mem_total_gb=$((mem_total / 1024))
        mem_usage_percent=$((mem_used * 100 / mem_total))
        
        # 状态颜色
        if [ $mem_usage_percent -lt 10 ] && [ $util -lt 10 ]; then
            status="${GREEN}空闲${NC}"
        elif [ $mem_usage_percent -lt 50 ] && [ $util -lt 50 ]; then
            status="${YELLOW}轻载${NC}"
        elif [ $mem_usage_percent -lt 80 ] && [ $util -lt 80 ]; then
            status="${YELLOW}中载${NC}"
        else
            status="${RED}重载${NC}"
        fi
        
        # 温度颜色
        if [ $temp -lt 60 ]; then
            temp_color="${GREEN}${temp}°C${NC}"
        elif [ $temp -lt 80 ]; then
            temp_color="${YELLOW}${temp}°C${NC}"
        else
            temp_color="${RED}${temp}°C${NC}"
        fi
        
        # 功耗颜色
        power_percent=$((power * 100 / power_limit))
        if [ $power_percent -lt 50 ]; then
            power_color="${GREEN}${power}W${NC}"
        elif [ $power_percent -lt 80 ]; then
            power_color="${YELLOW}${power}W${NC}"
        else
            power_color="${RED}${power}W${NC}"
        fi
        
        printf "%-5s %-15s %-8s %-15s %-8s %-8s %s\n" \
            "$idx" \
            "${mem_used_gb}GB/${mem_total_gb}GB" \
            "${util}%" \
            "$temp_color" \
            "$power_color" \
            "${fan}%" \
            "$status"
    done
    echo ""
}

# GPU进程信息
show_gpu_processes() {
    echo -e "${YELLOW}=== GPU进程信息 ===${NC}"
    
    # 检查是否有GPU进程
    if ! nvidia-smi --query-compute-apps=pid,process_name,gpu_uuid,used_memory --format=csv,noheader,nounits 2>/dev/null | grep -q .; then
        echo "当前无GPU进程运行"
        echo ""
        return
    fi
    
    printf "%-5s %-8s %-20s %-10s %-15s\n" "GPU" "PID" "进程名" "显存" "用户"
    echo "─"*65
    
    nvidia-smi --query-compute-apps=pid,process_name,gpu_uuid,used_memory --format=csv,noheader,nounits 2>/dev/null | while IFS=',' read -r pid process gpu_uuid mem; do
        # 获取GPU ID
        gpu_id=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | grep "$gpu_uuid" | cut -d',' -f1)
        
        # 获取进程用户
        user=$(ps -o user= -p $pid 2>/dev/null || echo "unknown")
        
        # 格式化显存
        mem_gb=$(echo "scale=1; $mem / 1024" | bc 2>/dev/null || echo "${mem}MB")
        if [[ $mem_gb == *"."* ]]; then
            mem_display="${mem_gb}GB"
        else
            mem_display="${mem}MB"
        fi
        
        # 截断长进程名
        short_process=$(echo "$process" | sed 's/.*\///g' | cut -c1-18)
        
        printf "%-5s %-8s %-20s %-10s %-15s\n" \
            "$gpu_id" \
            "$pid" \
            "$short_process" \
            "$mem_display" \
            "$user"
    done
    echo ""
}

# 系统资源概览
show_system_overview() {
    echo -e "${GREEN}=== 系统资源概览 ===${NC}"
    
    # CPU信息
    cpu_usage=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | awk -F'%' '{print $1}')
    echo "CPU使用率: ${cpu_usage}%"
    
    # 内存信息
    mem_info=$(free -h | awk '/^Mem:/ {printf "%.1f%% (%s/%s)", $3/$2*100, $3, $2}')
    echo "内存使用: $mem_info"
    
    # 磁盘信息
    disk_usage=$(df -h . | tail -1 | awk '{print $5 " (" $3 "/" $2 ")"}')
    echo "磁盘使用: $disk_usage"
    
    # 负载信息
    load_avg=$(uptime | awk -F'load average:' '{print $2}')
    echo "系统负载:$load_avg"
    
    echo ""
}

# 训练状态检查
show_training_status() {
    echo -e "${PURPLE}=== 训练状态检查 ===${NC}"
    
    # 检查训练进程
    training_procs=$(ps aux | grep -E '(train_pressure_field|python.*train)' | grep -v grep)
    
    if [ -z "$training_procs" ]; then
        echo "当前无训练进程运行"
    else
        echo "运行中的训练进程:"
        echo "$training_procs" | while read line; do
            pid=$(echo "$line" | awk '{print $2}')
            cmd=$(echo "$line" | awk '{for(i=11;i<=NF;i++) printf "%s ", $i; print ""}')
            echo "  PID $pid: $(echo "$cmd" | cut -c1-60)..."
        done
    fi
    
    # 检查TensorBoard
    tb_proc=$(ps aux | grep tensorboard | grep -v grep)
    if [ -n "$tb_proc" ]; then
        tb_port=$(echo "$tb_proc" | grep -o 'port=[0-9]*' | cut -d'=' -f2)
        server_ip=$(hostname -I | awk '{print $1}')
        echo "TensorBoard运行中: http://$server_ip:$tb_port"
    fi
    
    echo ""
}

# GPU推荐
show_gpu_recommendations() {
    echo -e "${CYAN}=== GPU使用推荐 ===${NC}"
    
    nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits | while IFS=',' read -r idx mem_used mem_total util; do
        mem_usage_percent=$((mem_used * 100 / mem_total))
        
        if [ $mem_usage_percent -lt 5 ] && [ $util -lt 5 ]; then
            echo -e "GPU $idx: ${GREEN}强烈推荐使用${NC} (空闲状态)"
        elif [ $mem_usage_percent -lt 20 ] && [ $util -lt 20 ]; then
            echo -e "GPU $idx: ${YELLOW}推荐使用${NC} (轻度使用)"
        elif [ $mem_usage_percent -lt 50 ] && [ $util -lt 50 ]; then
            echo -e "GPU $idx: ${YELLOW}可以使用${NC} (中度使用)"
        else
            echo -e "GPU $idx: ${RED}不推荐使用${NC} (重度使用)"
        fi
    done
    echo ""
}

# 主监控函数
monitor_loop() {
    while true; do
        clear_screen
        show_gpu_status
        show_gpu_processes
        show_system_overview
        show_training_status
        show_gpu_recommendations
        
        echo -e "${BLUE}快捷命令:${NC}"
        echo "  nvidia-smi                    # 详细GPU信息"
        echo "  htop                          # 系统资源监控"
        echo "  kill <PID>                    # 终止进程"
        echo "  export CUDA_VISIBLE_DEVICES=1 # 设置使用GPU 1"
        
        sleep $UPDATE_INTERVAL
    done
}

# 简单状态显示
show_simple_status() {
    echo -e "${PURPLE}=== GPU快速状态 ===${NC}"
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader,nounits | while IFS=',' read -r idx name mem_used mem_total util temp; do
        mem_used_gb=$((mem_used / 1024))
        mem_total_gb=$((mem_total / 1024))
        mem_usage_percent=$((mem_used * 100 / mem_total))
        
        if [ $mem_usage_percent -lt 10 ]; then
            status_color="$GREEN"
        elif [ $mem_usage_percent -lt 50 ]; then
            status_color="$YELLOW"
        else
            status_color="$RED"
        fi
        
        echo -e "GPU $idx ($name): ${status_color}${mem_used_gb}GB/${mem_total_gb}GB (${mem_usage_percent}%), ${util}%, ${temp}°C${NC}"
    done
}

# 帮助信息
show_help() {
    echo -e "${BLUE}GPU监控脚本使用说明:${NC}"
    echo ""
    echo "用法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  -h, --help          显示帮助信息"
    echo "  -s, --simple        显示简单状态"
    echo "  -i, --info          显示GPU硬件信息"
    echo "  -p, --processes     显示GPU进程"
    echo "  -r, --recommend     显示GPU使用推荐"
    echo "  -t, --interval N    设置更新间隔(秒，默认2)"
    echo "  无参数              启动实时监控"
    echo ""
    echo "示例:"
    echo "  $0                  # 启动实时监控"
    echo "  $0 -s               # 显示简单状态"
    echo "  $0 -t 5             # 5秒间隔监控"
}

# 参数解析
UPDATE_INTERVAL=2

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -s|--simple)
            show_simple_status
            exit 0
            ;;
        -i|--info)
            show_gpu_info
            exit 0
            ;;
        -p|--processes)
            show_gpu_processes
            exit 0
            ;;
        -r|--recommend)
            show_gpu_recommendations
            exit 0
            ;;
        -t|--interval)
            UPDATE_INTERVAL="$2"
            shift 2
            ;;
        *)
            echo "未知参数: $1"
            show_help
            exit 1
            ;;
    esac
done

# 检查nvidia-smi是否可用
if ! command -v nvidia-smi &> /dev/null; then
    echo -e "${RED}错误: nvidia-smi未找到${NC}"
    echo "请确保NVIDIA驱动已正确安装"
    exit 1
fi

# 检查bc命令（用于浮点计算）
if ! command -v bc &> /dev/null; then
    echo -e "${YELLOW}警告: bc命令未找到，某些计算可能不准确${NC}"
fi

# 启动监控
echo -e "${GREEN}启动GPU监控...${NC}"
echo "按Ctrl+C退出"
sleep 1

# 捕获Ctrl+C信号
trap 'echo -e "\n${GREEN}监控已停止${NC}"; exit 0' INT

# 开始监控循环
monitor_loop