#!/bin/bash

# System monitoring script
# Monitors CPU, memory, disk, network and training process status

# Color definitions
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
WHITE='\033[1;37m'
NC='\033[0m' # No Color

# Function: Display system basic information
show_system_info() {
    echo -e "${BLUE}=== System Basic Information ===${NC}"
    echo -e "${CYAN}Hostname:${NC} $(hostname)"
    echo -e "${CYAN}Operating System:${NC} $(uname -o)"
    echo -e "${CYAN}Kernel Version:${NC} $(uname -r)"
    echo -e "${CYAN}Architecture:${NC} $(uname -m)"
    echo -e "${CYAN}Uptime:${NC} $(uptime -p)"
    echo -e "${CYAN}Current Time:${NC} $(date)"
    echo
}

# Function: Display CPU information
show_cpu_info() {
    echo -e "${BLUE}=== CPU Information ===${NC}"
    
    # CPU model
    CPU_MODEL=$(grep "model name" /proc/cpuinfo | head -1 | cut -d: -f2 | sed 's/^ *//')
    CPU_CORES=$(nproc)
    CPU_THREADS=$(grep -c processor /proc/cpuinfo)
    
    echo -e "${CYAN}CPU Model:${NC} $CPU_MODEL"
    echo -e "${CYAN}Physical Cores:${NC} $CPU_CORES"
    echo -e "${CYAN}Logical Threads:${NC} $CPU_THREADS"
    
    # CPU usage
    CPU_USAGE=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1)
    echo -e "${CYAN}CPU Usage:${NC} ${CPU_USAGE}%"
    
    # Load average
    LOAD_AVERAGE=$(uptime | awk -F'load average:' '{print $2}')
    echo -e "${CYAN}Load Average:${NC}$LOAD_AVERAGE"
    
    # CPU temperature (if available)
    if command -v sensors >/dev/null 2>&1; then
        TEMP=$(sensors | grep -i "core 0" | awk '{print $3}' | head -1)
        if [ ! -z "$TEMP" ]; then
            echo -e "${CYAN}CPU Temperature:${NC} $TEMP"
        fi
    fi
    echo
}

# Function: Display memory information
show_memory_info() {
    echo -e "${BLUE}=== Memory Information ===${NC}"
    
    # Memory usage information
    MEMORY_INFO=$(free -h | grep "Mem:")
    TOTAL_MEM=$(echo $MEMORY_INFO | awk '{print $2}')
    USED_MEM=$(echo $MEMORY_INFO | awk '{print $3}')
    FREE_MEM=$(echo $MEMORY_INFO | awk '{print $4}')
    AVAILABLE_MEM=$(echo $MEMORY_INFO | awk '{print $7}')
    
    # Calculate memory usage percentage
    USED_PERCENT=$(free | grep Mem | awk '{printf("%.1f"), $3/$2 * 100.0}')
    
    echo -e "${CYAN}Total Memory:${NC} $TOTAL_MEM"
    echo -e "${CYAN}Used Memory:${NC} $USED_MEM (${USED_PERCENT}%)"
    echo -e "${CYAN}Free Memory:${NC} $FREE_MEM"
    echo -e "${CYAN}Available Memory:${NC} $AVAILABLE_MEM"
    
    # Swap information
    SWAP_INFO=$(free -h | grep "Swap:")
    if [ ! -z "$SWAP_INFO" ]; then
        SWAP_TOTAL=$(echo $SWAP_INFO | awk '{print $2}')
        SWAP_USED=$(echo $SWAP_INFO | awk '{print $3}')
        SWAP_FREE=$(echo $SWAP_INFO | awk '{print $4}')
        echo -e "${CYAN}Swap Total:${NC} $SWAP_TOTAL"
        echo -e "${CYAN}Swap Used:${NC} $SWAP_USED"
        echo -e "${CYAN}Swap Free:${NC} $SWAP_FREE"
    fi
    echo
}

# Function: Display disk information
show_disk_info() {
    echo -e "${BLUE}=== Disk Information ===${NC}"
    
    # Disk usage
    df -h | grep -E '^/dev/' | while read line; do
        DEVICE=$(echo $line | awk '{print $1}')
        SIZE=$(echo $line | awk '{print $2}')
        USED=$(echo $line | awk '{print $3}')
        AVAILABLE=$(echo $line | awk '{print $4}')
        USE_PERCENT=$(echo $line | awk '{print $5}')
        MOUNT_POINT=$(echo $line | awk '{print $6}')
        
        echo -e "${CYAN}Device:${NC} $DEVICE"
        echo -e "${CYAN}Mount Point:${NC} $MOUNT_POINT"
        echo -e "${CYAN}Total Size:${NC} $SIZE"
        echo -e "${CYAN}Used:${NC} $USED ($USE_PERCENT)"
        echo -e "${CYAN}Available:${NC} $AVAILABLE"
        echo "----"
    done
    
    # Disk I/O information
    if command -v iostat >/dev/null 2>&1; then
        echo -e "${CYAN}Disk I/O Status:${NC}"
        iostat -x 1 1 | grep -A 20 "Device"
    fi
    echo
}

# Function: Display network information
show_network_info() {
    echo -e "${BLUE}=== Network Information ===${NC}"
    
    # Network interface information
    ip addr show | grep -E "^\d+:|inet " | while read line; do
        if echo "$line" | grep -q "^[0-9]"; then
            INTERFACE=$(echo $line | awk '{print $2}' | sed 's/://')
            echo -e "${CYAN}Interface:${NC} $INTERFACE"
        elif echo "$line" | grep -q "inet "; then
            IP=$(echo $line | awk '{print $2}')
            echo -e "${CYAN}IP Address:${NC} $IP"
            echo "----"
        fi
    done
    
    # Network statistics
    if command -v ss >/dev/null 2>&1; then
        ESTABLISHED=$(ss -t state established | wc -l)
        LISTENING=$(ss -tln | wc -l)
        echo -e "${CYAN}Established Connections:${NC} $((ESTABLISHED-1))"
        echo -e "${CYAN}Listening Ports:${NC} $((LISTENING-1))"
    fi
    echo
}

# Function: Display GPU information
show_gpu_info() {
    echo -e "${BLUE}=== GPU Information ===${NC}"
    
    if command -v nvidia-smi >/dev/null 2>&1; then
        # GPU basic information
        nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu --format=csv,noheader,nounits | while IFS=', ' read name total used free util temp; do
            echo -e "${CYAN}GPU Model:${NC} $name"
            echo -e "${CYAN}Total VRAM:${NC} ${total}MB"
            echo -e "${CYAN}Used VRAM:${NC} ${used}MB"
            echo -e "${CYAN}Free VRAM:${NC} ${free}MB"
            echo -e "${CYAN}GPU Utilization:${NC} ${util}%"
            echo -e "${CYAN}Temperature:${NC} ${temp}°C"
            echo "----"
        done
        
        # GPU processes
        echo -e "${CYAN}GPU Processes:${NC}"
        nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader | head -10
    else
        echo -e "${YELLOW}nvidia-smi command not found, unable to display GPU information${NC}"
    fi
    echo
}

# Function: Display training process information
show_training_processes() {
    echo -e "${BLUE}=== Training Process Information ===${NC}"
    
    # Find Python training processes
    PYTHON_PROCESSES=$(ps aux | grep python | grep -v grep | grep -E "(train|training)")
    
    if [ ! -z "$PYTHON_PROCESSES" ]; then
        echo -e "${CYAN}Training Processes Found:${NC}"
        echo "$PYTHON_PROCESSES" | while read line; do
            PID=$(echo $line | awk '{print $2}')
            USER=$(echo $line | awk '{print $1}')
            CPU=$(echo $line | awk '{print $3}')
            MEM=$(echo $line | awk '{print $4}')
            CMD=$(echo $line | awk '{for(i=11;i<=NF;i++) printf "%s ", $i; print ""}')
            
            echo -e "${CYAN}PID:${NC} $PID"
            echo -e "${CYAN}User:${NC} $USER"
            echo -e "${CYAN}CPU%:${NC} $CPU"
            echo -e "${CYAN}Memory%:${NC} $MEM"
            echo -e "${CYAN}Command:${NC} $CMD"
            echo "----"
        done
    else
        echo -e "${YELLOW}No training processes found${NC}"
    fi
    
    # Check TensorBoard processes
    TB_PROCESSES=$(ps aux | grep tensorboard | grep -v grep)
    if [ ! -z "$TB_PROCESSES" ]; then
        echo -e "${CYAN}TensorBoard Processes:${NC}"
        echo "$TB_PROCESSES"
    fi
    echo
}

# Function: Display resource alert information
show_alerts() {
    echo -e "${BLUE}=== Resource Alerts ===${NC}"
    
    # Memory usage alert
    MEMORY_USED_PERCENT=$(free | grep Mem | awk '{printf("%.0f"), $3/$2 * 100.0}')
    if [ $MEMORY_USED_PERCENT -gt 90 ]; then
        echo -e "${RED}WARNING: Memory usage too high (${MEMORY_USED_PERCENT}%)${NC}"
    elif [ $MEMORY_USED_PERCENT -gt 80 ]; then
        echo -e "${YELLOW}CAUTION: Memory usage high (${MEMORY_USED_PERCENT}%)${NC}"
    else
        echo -e "${GREEN}Memory usage normal (${MEMORY_USED_PERCENT}%)${NC}"
    fi
    
    # CPU usage alert
    CPU_USED_PERCENT=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1 | cut -d',' -f1)
    CPU_USED_INT=$(printf "%.0f" $CPU_USED_PERCENT)
    if [ $CPU_USED_INT -gt 90 ]; then
        echo -e "${RED}WARNING: CPU usage too high (${CPU_USED_PERCENT}%)${NC}"
    elif [ $CPU_USED_INT -gt 80 ]; then
        echo -e "${YELLOW}CAUTION: CPU usage high (${CPU_USED_PERCENT}%)${NC}"
    else
        echo -e "${GREEN}CPU usage normal (${CPU_USED_PERCENT}%)${NC}"
    fi
    
    # Disk usage alert
    df -h | grep -E '^/dev/' | while read line; do
        USE_PERCENT=$(echo $line | awk '{print $5}' | sed 's/%//')
        MOUNT_POINT=$(echo $line | awk '{print $6}')
        
        if [ $USE_PERCENT -gt 90 ]; then
            echo -e "${RED}WARNING: Disk $MOUNT_POINT usage too high (${USE_PERCENT}%)${NC}"
        elif [ $USE_PERCENT -gt 80 ]; then
            echo -e "${YELLOW}CAUTION: Disk $MOUNT_POINT usage high (${USE_PERCENT}%)${NC}"
        fi
    done
    
    # GPU VRAM usage alert (if NVIDIA GPU exists)
    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits | while IFS=', ' read used total; do
            VRAM_PERCENT=$((used * 100 / total))
            if [ $VRAM_PERCENT -gt 90 ]; then
                echo -e "${RED}WARNING: GPU VRAM usage too high (${VRAM_PERCENT}%)${NC}"
            elif [ $VRAM_PERCENT -gt 80 ]; then
                echo -e "${YELLOW}CAUTION: GPU VRAM usage high (${VRAM_PERCENT}%)${NC}"
            fi
        done
    fi
    echo
}

# Function: Real-time monitoring
real_time_monitor() {
    echo -e "${GREEN}Starting real-time monitoring... (Press Ctrl+C to exit)${NC}"
    echo
    
    while true; do
        clear
        echo -e "${WHITE}=== System Real-Time Monitoring ===${NC}"
        echo -e "${CYAN}Update Time:${NC} $(date)"
        echo
        
        # Display key metrics
        show_cpu_info
        show_memory_info
        show_gpu_info
        show_training_processes
        show_alerts
        
        sleep 5
    done
}

# Function: Generate system report
generate_report() {
    REPORT_FILE="system_report_$(date +%Y%m%d_%H%M%S).txt"
    echo -e "${GREEN}Generating system report...${NC}"
    
    {
        echo "=== System Monitoring Report ==="
        echo "Report Generation Time: $(date)"
        echo
        
        show_system_info
        show_cpu_info
        show_memory_info
        show_disk_info
        show_network_info
        show_gpu_info
        show_training_processes
        show_alerts
        
    } > "$REPORT_FILE"
    
    echo -e "${GREEN}Report generated: $REPORT_FILE${NC}"
}

# Function: Display help information
show_help() {
    echo -e "${WHITE}System Monitoring Script Help${NC}"
    echo
    echo -e "${CYAN}Usage:${NC}"
    echo "  $0 [options]"
    echo
    echo -e "${CYAN}Options:${NC}"
    echo "  -h, --help      Show help information"
    echo "  -a, --all       Show all system information"
    echo "  -c, --cpu       Show CPU information only"
    echo "  -m, --memory    Show memory information only"
    echo "  -d, --disk      Show disk information only"
    echo "  -n, --network   Show network information only"
    echo "  -g, --gpu       Show GPU information only"
    echo "  -p, --process   Show training process information only"
    echo "  -r, --realtime  Start real-time monitoring"
    echo "  --report        Generate system report"
    echo "  --alerts        Show resource alerts only"
    echo
    echo -e "${CYAN}Examples:${NC}"
    echo "  $0 -a           # Show all information"
    echo "  $0 -c -m        # Show CPU and memory information"
    echo "  $0 -r           # Start real-time monitoring"
    echo "  $0 --report     # Generate report file"
}

# Main logic
main() {
    # Parameter handling
    if [ $# -eq 0 ]; then
        # Default: Show all information
        show_system_info
        show_cpu_info
        show_memory_info
        show_disk_info
        show_network_info
        show_gpu_info
        show_training_processes
        show_alerts
        exit 0
    fi
    
    # Process command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_help
                exit 0
                ;;
            -a|--all)
                show_system_info
                show_cpu_info
                show_memory_info
                show_disk_info
                show_network_info
                show_gpu_info
                show_training_processes
                show_alerts
                ;;
            -c|--cpu)
                show_cpu_info
                ;;
            -m|--memory)
                show_memory_info
                ;;
            -d|--disk)
                show_disk_info
                ;;
            -n|--network)
                show_network_info
                ;;
            -g|--gpu)
                show_gpu_info
                ;;
            -p|--process)
                show_training_processes
                ;;
            -r|--realtime)
                real_time_monitor
                ;;
            --report)
                generate_report
                ;;
            --alerts)
                show_alerts
                ;;
            *)
                echo -e "${RED}Unknown option: $1${NC}"
                echo "Use -h or --help to view help information"
                exit 1
                ;;
        esac
        shift
    done
}

# Execute main function
main "$@"