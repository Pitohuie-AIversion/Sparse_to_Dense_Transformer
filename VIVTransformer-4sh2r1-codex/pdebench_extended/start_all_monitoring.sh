#!/bin/bash

# =============================================================================
# VIV Transformer 监控工具启动器 (Linux版本)
# 同时启动TensorBoard、GPU监控、系统监控等工具
# =============================================================================

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m'

# 获取脚本目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo -e "${PURPLE}========================================${NC}"
echo -e "${PURPLE}    VIV Transformer 监控工具启动器${NC}"
echo -e "${PURPLE}========================================${NC}"
echo

echo -e "${BLUE}[$(date '+%H:%M:%S')] 正在启动所有监控工具...${NC}"
echo

# 检查依赖
check_dependencies() {
    echo -e "${CYAN}=== 检查系统依赖 ===${NC}"
    
    # 检查Python
    if command -v python3 &> /dev/null; then
        PYTHON_CMD="python3"
    elif command -v python &> /dev/null; then
        PYTHON_CMD="python"
    else
        echo -e "${RED}[错误] Python未找到${NC}"
        exit 1
    fi
    echo -e "${GREEN}✓ Python: $($PYTHON_CMD --version)${NC}"
    
    # 检查TensorBoard
    if $PYTHON_CMD -c "import tensorboard" &> /dev/null; then
        TENSORBOARD_AVAILABLE=1
        echo -e "${GREEN}✓ TensorBoard可用${NC}"
    else
        TENSORBOARD_AVAILABLE=0
        echo -e "${YELLOW}⚠ TensorBoard不可用${NC}"
    fi
    
    # 检查NVIDIA GPU
    if command -v nvidia-smi &> /dev/null; then
        GPU_AVAILABLE=1
        echo -e "${GREEN}✓ NVIDIA GPU可用${NC}"
    else
        GPU_AVAILABLE=0
        echo -e "${YELLOW}⚠ NVIDIA GPU不可用${NC}"
    fi
    
    # 检查tmux或screen
    if command -v tmux &> /dev/null; then
        TERMINAL_CMD="tmux"
        echo -e "${GREEN}✓ tmux可用${NC}"
    elif command -v screen &> /dev/null; then
        TERMINAL_CMD="screen"
        echo -e "${GREEN}✓ screen可用${NC}"
    else
        TERMINAL_CMD="gnome-terminal"
        echo -e "${YELLOW}⚠ 使用gnome-terminal${NC}"
    fi
    
    echo
}

# 启动TensorBoard
start_tensorboard() {
    if [ $TENSORBOARD_AVAILABLE -eq 1 ]; then
        echo -e "${CYAN}=== 启动TensorBoard ===${NC}"
        
        # 查找可用端口
        TB_PORT=6006
        while netstat -ln | grep ":$TB_PORT " > /dev/null 2>&1; do
            TB_PORT=$((TB_PORT + 1))
        done
        
        if [ "$TERMINAL_CMD" = "tmux" ]; then
            tmux new-session -d -s tensorboard "echo 'TensorBoard监控 - 访问 http://localhost:$TB_PORT'; $PYTHON_CMD start_tensorboard.py --port $TB_PORT"
        elif [ "$TERMINAL_CMD" = "screen" ]; then
            screen -dmS tensorboard bash -c "echo 'TensorBoard监控'; $PYTHON_CMD start_tensorboard.py"
        else
            gnome-terminal --title="TensorBoard监控" -- bash -c "echo 'TensorBoard监控 - 访问 http://localhost:$TB_PORT'; $PYTHON_CMD start_tensorboard.py; exec bash" &
        fi
        
        sleep 2
        echo -e "${GREEN}[$(date '+%H:%M:%S')] TensorBoard已启动 (端口: $TB_PORT)${NC}"
    else
        echo -e "${YELLOW}[跳过] TensorBoard不可用${NC}"
    fi
    echo
}

# 启动GPU监控
start_gpu_monitoring() {
    if [ $GPU_AVAILABLE -eq 1 ]; then
        echo -e "${CYAN}=== 启动GPU监控 ===${NC}"
        
        if [ -f "gpu_monitor.sh" ]; then
            if [ "$TERMINAL_CMD" = "tmux" ]; then
                tmux new-session -d -s gpu_monitor "bash gpu_monitor.sh"
            elif [ "$TERMINAL_CMD" = "screen" ]; then
                screen -dmS gpu_monitor bash -c "bash gpu_monitor.sh"
            else
                gnome-terminal --title="GPU监控" -- bash -c "bash gpu_monitor.sh; exec bash" &
            fi
        else
            # 简化GPU监控
            if [ "$TERMINAL_CMD" = "tmux" ]; then
                tmux new-session -d -s gpu_simple "while true; do clear; echo 'GPU状态监控'; nvidia-smi; sleep 5; done"
            else
                gnome-terminal --title="GPU状态" -- bash -c "while true; do clear; echo 'GPU状态监控'; nvidia-smi; sleep 5; done" &
            fi
        fi
        
        echo -e "${GREEN}[$(date '+%H:%M:%S')] GPU监控已启动${NC}"
    else
        echo -e "${YELLOW}[跳过] GPU不可用${NC}"
    fi
    echo
}

# 启动系统监控
start_system_monitoring() {
    echo -e "${CYAN}=== 启动系统监控 ===${NC}"
    
    if [ -f "system_monitor.sh" ]; then
        if [ "$TERMINAL_CMD" = "tmux" ]; then
            tmux new-session -d -s system_monitor "bash system_monitor.sh"
        elif [ "$TERMINAL_CMD" = "screen" ]; then
            screen -dmS system_monitor bash -c "bash system_monitor.sh"
        else
            gnome-terminal --title="系统监控" -- bash -c "bash system_monitor.sh; exec bash" &
        fi
    else
        # 简化系统监控
        if [ "$TERMINAL_CMD" = "tmux" ]; then
            tmux new-session -d -s system_simple "while true; do clear; echo '系统资源监控'; echo 'CPU:'; top -bn1 | grep 'Cpu(s)'; echo 'Memory:'; free -h; echo 'Disk:'; df -h .; sleep 10; done"
        else
            gnome-terminal --title="系统资源" -- bash -c "while true; do clear; echo '系统资源监控'; top -bn1 | head -20; sleep 10; done" &
        fi
    fi
    
    echo -e "${GREEN}[$(date '+%H:%M:%S')] 系统监控已启动${NC}"
    echo
}

# 启动训练日志监控
start_log_monitoring() {
    echo -e "${CYAN}=== 启动训练日志监控 ===${NC}"
    
    if [ -d "outputs" ]; then
        if [ "$TERMINAL_CMD" = "tmux" ]; then
            tmux new-session -d -s log_monitor "while true; do clear; echo '训练日志监控'; find outputs -name '*.log' -exec echo '=== {} ===' \; -exec tail -n 10 {} \; 2>/dev/null || echo '暂无日志文件'; sleep 15; done"
        else
            gnome-terminal --title="训练日志" -- bash -c "while true; do clear; echo '训练日志监控'; find outputs -name '*.log' -exec echo '=== {} ===' \; -exec tail -n 10 {} \; 2>/dev/null || echo '暂无日志文件'; sleep 15; done" &
        fi
        echo -e "${GREEN}[$(date '+%H:%M:%S')] 训练日志监控已启动${NC}"
    else
        echo -e "${YELLOW}[跳过] 输出目录不存在${NC}"
    fi
    echo
}

# 启动进程监控
start_process_monitoring() {
    echo -e "${CYAN}=== 启动进程监控 ===${NC}"
    
    if [ "$TERMINAL_CMD" = "tmux" ]; then
        tmux new-session -d -s process_monitor "while true; do clear; echo 'Python进程监控'; echo '[$(date '+%H:%M:%S')] Python训练进程:'; ps aux | grep python | grep -E '(train|pressure_field)' | grep -v grep || echo '无训练进程'; sleep 8; done"
    else
        gnome-terminal --title="进程监控" -- bash -c "while true; do clear; echo 'Python进程监控'; ps aux | grep python | head -20; sleep 8; done" &
    fi
    
    echo -e "${GREEN}[$(date '+%H:%M:%S')] 进程监控已启动${NC}"
    echo
}

# 显示监控状态
show_monitoring_status() {
    echo -e "${PURPLE}========================================${NC}"
    echo -e "${PURPLE}           监控工具启动完成！${NC}"
    echo -e "${PURPLE}========================================${NC}"
    echo
    
    echo -e "${GREEN}已启动的监控服务:${NC}"
    if [ $TENSORBOARD_AVAILABLE -eq 1 ]; then
        echo -e "  ${CYAN}• TensorBoard:${NC} http://localhost:6006"
    fi
    if [ $GPU_AVAILABLE -eq 1 ]; then
        echo -e "  ${CYAN}• GPU监控:${NC} 实时GPU状态"
    fi
    echo -e "  ${CYAN}• 系统监控:${NC} CPU/内存/磁盘状态"
    echo -e "  ${CYAN}• 训练日志:${NC} 实时训练日志"
    echo -e "  ${CYAN}• 进程监控:${NC} Python进程状态"
    echo
    
    if [ "$TERMINAL_CMD" = "tmux" ]; then
        echo -e "${YELLOW}管理命令 (tmux):${NC}"
        echo "  tmux list-sessions          # 查看所有会话"
        echo "  tmux attach -t tensorboard  # 连接到TensorBoard会话"
        echo "  tmux kill-session -t <name> # 关闭指定会话"
        echo "  tmux kill-server            # 关闭所有会话"
    elif [ "$TERMINAL_CMD" = "screen" ]; then
        echo -e "${YELLOW}管理命令 (screen):${NC}"
        echo "  screen -list                # 查看所有会话"
        echo "  screen -r tensorboard       # 连接到TensorBoard会话"
        echo "  screen -S <name> -X quit    # 关闭指定会话"
    else
        echo -e "${YELLOW}提示:${NC}"
        echo "  • 关闭任意监控窗口不会影响其他窗口"
        echo "  • 按Ctrl+C可停止对应监控"
    fi
    
    echo
    echo -e "${GREEN}重新运行此脚本可重启所有监控${NC}"
    echo
}

# 主函数
main() {
    check_dependencies
    start_tensorboard
    start_gpu_monitoring
    start_system_monitoring
    start_log_monitoring
    start_process_monitoring
    show_monitoring_status
    
    echo -e "${GREEN}监控启动完成！按Enter键退出...${NC}"
    read
}

# 信号处理
trap 'echo -e "\n${GREEN}监控启动器已退出${NC}"; exit 0' INT

# 运行主函数
main