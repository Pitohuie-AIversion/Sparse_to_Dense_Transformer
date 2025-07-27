#!/bin/bash

# =============================================================================
# 系统监控脚本 - 训练过程监控
# 监控CPU、内存、磁盘、网络和训练进程
# =============================================================================

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m'

# 配置参数
UPDATE_INTERVAL=3
LOG_FILE="system_monitor.log"
ALERT_CPU_THRESHOLD=90
ALERT_MEM_THRESHOLD=90
ALERT_DISK_THRESHOLD=90

# 清屏函数
clear_screen() {
    clear
    echo -e "${PURPLE}=== 系统监控面板 - 训练环境监控 ===${NC}"
    echo "时间: $(date)"
    echo "更新间隔: ${UPDATE_INTERVAL}秒 | 日志文件: $LOG_FILE"
    echo "="*80
}

# CPU监控
show_cpu_status() {
    echo -e "${BLUE}=== CPU状态 ===${NC}"
    
    # CPU使用率
    cpu_usage=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | awk -F'%' '{print $1}')
    cpu_usage_int=${cpu_usage%.*}
    
    # CPU颜色
    if [ "$cpu_usage_int" -lt 50 ]; then
        cpu_color="$GREEN"
    elif [ "$cpu_usage_int" -lt 80 ]; then
        cpu_color="$YELLOW"
    else
        cpu_color="$RED"
    fi
    
    # 负载信息
    load_avg=$(uptime | awk -F'load average:' '{print $2}' | sed 's/^[ \t]*//')
    
    # CPU核心数
    cpu_cores=$(nproc)
    
    # CPU频率
    cpu_freq=$(cat /proc/cpuinfo | grep "cpu MHz" | head -1 | awk '{print $4}' | cut -d'.' -f1)
    
    echo -e "CPU使用率: ${cpu_color}${cpu_usage}%${NC}"
    echo "负载平均: $load_avg"
    echo "CPU核心: $cpu_cores 个"
    echo "CPU频率: ${cpu_freq}MHz"
    
    # CPU告警
    if [ "$cpu_usage_int" -gt "$ALERT_CPU_THRESHOLD" ]; then
        echo -e "${RED}⚠️  CPU使用率过高!${NC}"
    fi
    
    echo ""
}

# 内存监控
show_memory_status() {
    echo -e "${CYAN}=== 内存状态 ===${NC}"
    
    # 内存信息
    mem_info=$(free -h)
    mem_total=$(echo "$mem_info" | awk '/^Mem:/ {print $2}')
    mem_used=$(echo "$mem_info" | awk '/^Mem:/ {print $3}')
    mem_free=$(echo "$mem_info" | awk '/^Mem:/ {print $4}')
    mem_available=$(echo "$mem_info" | awk '/^Mem:/ {print $7}')
    
    # 内存使用百分比
    mem_usage_percent=$(free | awk '/^Mem:/ {printf "%.1f", $3/$2*100}')
    mem_usage_int=${mem_usage_percent%.*}
    
    # 内存颜色
    if [ "$mem_usage_int" -lt 60 ]; then
        mem_color="$GREEN"
    elif [ "$mem_usage_int" -lt 80 ]; then
        mem_color="$YELLOW"
    else
        mem_color="$RED"
    fi
    
    # Swap信息
    swap_total=$(echo "$mem_info" | awk '/^Swap:/ {print $2}')
    swap_used=$(echo "$mem_info" | awk '/^Swap:/ {print $3}')
    
    echo -e "内存使用: ${mem_color}${mem_used}/${mem_total} (${mem_usage_percent}%)${NC}"
    echo "可用内存: $mem_available"
    echo "Swap使用: $swap_used/$swap_total"
    
    # 内存告警
    if [ "$mem_usage_int" -gt "$ALERT_MEM_THRESHOLD" ]; then
        echo -e "${RED}⚠️  内存使用率过高!${NC}"
    fi
    
    echo ""
}

# 磁盘监控
show_disk_status() {
    echo -e "${YELLOW}=== 磁盘状态 ===${NC}"
    
    # 当前目录磁盘使用
    disk_info=$(df -h .)
    disk_usage=$(echo "$disk_info" | tail -1 | awk '{print $5}' | sed 's/%//')
    disk_used=$(echo "$disk_info" | tail -1 | awk '{print $3}')
    disk_total=$(echo "$disk_info" | tail -1 | awk '{print $2}')
    disk_mount=$(echo "$disk_info" | tail -1 | awk '{print $6}')
    
    # 磁盘颜色
    if [ "$disk_usage" -lt 70 ]; then
        disk_color="$GREEN"
    elif [ "$disk_usage" -lt 85 ]; then
        disk_color="$YELLOW"
    else
        disk_color="$RED"
    fi
    
    echo -e "当前分区: ${disk_color}${disk_used}/${disk_total} (${disk_usage}%)${NC} - $disk_mount"
    
    # 磁盘IO
    if command -v iostat &> /dev/null; then
        io_info=$(iostat -d 1 2 | tail -n +4 | tail -1)
        if [ -n "$io_info" ]; then
            read_rate=$(echo "$io_info" | awk '{print $3}')
            write_rate=$(echo "$io_info" | awk '{print $4}')
            echo "磁盘IO: 读取 ${read_rate}KB/s, 写入 ${write_rate}KB/s"
        fi
    fi
    
    # 磁盘告警
    if [ "$disk_usage" -gt "$ALERT_DISK_THRESHOLD" ]; then
        echo -e "${RED}⚠️  磁盘空间不足!${NC}"
    fi
    
    echo ""
}

# 网络监控
show_network_status() {
    echo -e "${GREEN}=== 网络状态 ===${NC}"
    
    # 网络接口信息
    if command -v ip &> /dev/null; then
        active_interfaces=$(ip link show | grep "state UP" | awk -F': ' '{print $2}' | head -3)
        for interface in $active_interfaces; do
            ip_addr=$(ip addr show $interface | grep "inet " | awk '{print $2}' | head -1)
            if [ -n "$ip_addr" ]; then
                echo "接口 $interface: $ip_addr"
            fi
        done
    fi
    
    # 网络连接数
    tcp_connections=$(netstat -tn 2>/dev/null | grep ESTABLISHED | wc -l)
    echo "TCP连接数: $tcp_connections"
    
    echo ""
}

# 训练进程监控
show_training_processes() {
    echo -e "${PURPLE}=== 训练进程监控 ===${NC}"
    
    # Python训练进程
    python_procs=$(ps aux | grep -E '(python.*train|train_pressure_field)' | grep -v grep)
    
    if [ -z "$python_procs" ]; then
        echo "当前无训练进程运行"
    else
        echo "运行中的训练进程:"
        echo "$python_procs" | while read line; do
            user=$(echo "$line" | awk '{print $1}')
            pid=$(echo "$line" | awk '{print $2}')
            cpu=$(echo "$line" | awk '{print $3}')
            mem=$(echo "$line" | awk '{print $4}')
            cmd=$(echo "$line" | awk '{for(i=11;i<=NF;i++) printf "%s ", $i; print ""}')
            
            # 进程运行时间
            runtime=$(ps -o etime= -p $pid 2>/dev/null | tr -d ' ')
            
            echo "  PID $pid ($user): CPU ${cpu}%, MEM ${mem}%, 运行时间 $runtime"
            echo "    命令: $(echo "$cmd" | cut -c1-60)..."
        done
    fi
    
    # TensorBoard进程
    tb_proc=$(ps aux | grep tensorboard | grep -v grep)
    if [ -n "$tb_proc" ]; then
        tb_pid=$(echo "$tb_proc" | awk '{print $2}')
        tb_port=$(echo "$tb_proc" | grep -o 'port=[0-9]*' | cut -d'=' -f2 || echo "6006")
        server_ip=$(hostname -I | awk '{print $1}')
        echo "TensorBoard: PID $tb_pid, 访问地址 http://$server_ip:$tb_port"
    fi
    
    # Jupyter进程
    jupyter_proc=$(ps aux | grep jupyter | grep -v grep)
    if [ -n "$jupyter_proc" ]; then
        jupyter_pid=$(echo "$jupyter_proc" | awk '{print $2}')
        echo "Jupyter: PID $jupyter_pid"
    fi
    
    echo ""
}

# 系统温度监控
show_temperature() {
    echo -e "${CYAN}=== 系统温度 ===${NC}"
    
    # CPU温度
    if command -v sensors &> /dev/null; then
        cpu_temp=$(sensors 2>/dev/null | grep -E "(Core|Package)" | head -1 | awk '{print $3}' | sed 's/+//g' | sed 's/°C//g')
        if [ -n "$cpu_temp" ]; then
            temp_value=${cpu_temp%.*}
            if [ "$temp_value" -lt 60 ]; then
                temp_color="$GREEN"
            elif [ "$temp_value" -lt 80 ]; then
                temp_color="$YELLOW"
            else
                temp_color="$RED"
            fi
            echo -e "CPU温度: ${temp_color}${cpu_temp}°C${NC}"
        fi
    fi
    
    # GPU温度（如果有nvidia-smi）
    if command -v nvidia-smi &> /dev/null; then
        gpu_temps=$(nvidia-smi --query-gpu=index,temperature.gpu --format=csv,noheader,nounits)
        if [ -n "$gpu_temps" ]; then
            echo "$gpu_temps" | while IFS=',' read -r idx temp; do
                if [ "$temp" -lt 60 ]; then
                    temp_color="$GREEN"
                elif [ "$temp" -lt 80 ]; then
                    temp_color="$YELLOW"
                else
                    temp_color="$RED"
                fi
                echo -e "GPU $idx温度: ${temp_color}${temp}°C${NC}"
            done
        fi
    fi
    
    echo ""
}

# 日志记录
log_status() {
    timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    cpu_usage=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | awk -F'%' '{print $1}')
    mem_usage=$(free | awk '/^Mem:/ {printf "%.1f", $3/$2*100}')
    
    echo "$timestamp,CPU:${cpu_usage}%,MEM:${mem_usage}%" >> "$LOG_FILE"
}

# 性能报告
show_performance_report() {
    echo -e "${BLUE}=== 性能报告 ===${NC}"
    
    if [ -f "$LOG_FILE" ]; then
        echo "最近10条记录:"
        tail -10 "$LOG_FILE" | while IFS=',' read -r timestamp cpu mem; do
            echo "  $timestamp - $cpu, $mem"
        done
    else
        echo "暂无性能日志"
    fi
    
    echo ""
}

# 主监控循环
monitor_loop() {
    while true; do
        clear_screen
        show_cpu_status
        show_memory_status
        show_disk_status
        show_network_status
        show_training_processes
        show_temperature
        
        # 记录日志
        log_status
        
        echo -e "${BLUE}快捷命令:${NC}"
        echo "  htop                    # 详细进程监控"
        echo "  iotop                   # 磁盘IO监控"
        echo "  nethogs                 # 网络使用监控"
        echo "  kill <PID>              # 终止进程"
        
        sleep $UPDATE_INTERVAL
    done
}

# 简单状态
show_simple_status() {
    cpu_usage=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | awk -F'%' '{print $1}')
    mem_usage=$(free | awk '/^Mem:/ {printf "%.1f", $3/$2*100}')
    disk_usage=$(df -h . | tail -1 | awk '{print $5}')
    load_avg=$(uptime | awk -F'load average:' '{print $2}' | sed 's/^[ \t]*//')
    
    echo -e "${GREEN}=== 系统快速状态 ===${NC}"
    echo "CPU: ${cpu_usage}% | 内存: ${mem_usage}% | 磁盘: $disk_usage | 负载:$load_avg"
    
    # 训练进程数
    training_count=$(ps aux | grep -E '(python.*train|train_pressure_field)' | grep -v grep | wc -l)
    echo "训练进程: $training_count 个"
}

# 帮助信息
show_help() {
    echo -e "${BLUE}系统监控脚本使用说明:${NC}"
    echo ""
    echo "用法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  -h, --help          显示帮助信息"
    echo "  -s, --simple        显示简单状态"
    echo "  -r, --report        显示性能报告"
    echo "  -t, --interval N    设置更新间隔(秒，默认3)"
    echo "  -l, --log FILE      设置日志文件(默认system_monitor.log)"
    echo "  --cpu-alert N       设置CPU告警阈值(默认90%)"
    echo "  --mem-alert N       设置内存告警阈值(默认90%)"
    echo "  --disk-alert N      设置磁盘告警阈值(默认90%)"
    echo "  无参数              启动实时监控"
    echo ""
    echo "示例:"
    echo "  $0                  # 启动实时监控"
    echo "  $0 -s               # 显示简单状态"
    echo "  $0 -t 5             # 5秒间隔监控"
    echo "  $0 --cpu-alert 80   # CPU告警阈值80%"
}

# 参数解析
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
        -r|--report)
            show_performance_report
            exit 0
            ;;
        -t|--interval)
            UPDATE_INTERVAL="$2"
            shift 2
            ;;
        -l|--log)
            LOG_FILE="$2"
            shift 2
            ;;
        --cpu-alert)
            ALERT_CPU_THRESHOLD="$2"
            shift 2
            ;;
        --mem-alert)
            ALERT_MEM_THRESHOLD="$2"
            shift 2
            ;;
        --disk-alert)
            ALERT_DISK_THRESHOLD="$2"
            shift 2
            ;;
        *)
            echo "未知参数: $1"
            show_help
            exit 1
            ;;
    esac
done

# 启动监控
echo -e "${GREEN}启动系统监控...${NC}"
echo "日志文件: $LOG_FILE"
echo "告警阈值: CPU ${ALERT_CPU_THRESHOLD}%, 内存 ${ALERT_MEM_THRESHOLD}%, 磁盘 ${ALERT_DISK_THRESHOLD}%"
echo "按Ctrl+C退出"
sleep 1

# 捕获Ctrl+C信号
trap 'echo -e "\n${GREEN}监控已停止${NC}"; exit 0' INT

# 开始监控循环
monitor_loop