#!/bin/bash
# VIV Transformer Monitoring Launcher (Shell Version)
# Compatible with Linux/Unix servers

echo "========================================"
echo "   VIV Transformer Monitoring Launcher"
echo "========================================"
echo ""

# Get project directory
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

echo "[$(date '+%H:%M:%S')] Starting all monitoring tools..."
echo ""

# Check Python environment
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
    PYTHON_AVAILABLE=true
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
    PYTHON_AVAILABLE=true
else
    echo "[ERROR] Python not found, please ensure Python is installed and in PATH"
    read -p "Press Enter to exit"
    exit 1
fi

# Check TensorBoard
if $PYTHON_CMD -c "import tensorboard" 2>/dev/null; then
    TENSORBOARD_AVAILABLE=true
else
    echo "[WARNING] TensorBoard not installed, skipping TensorBoard startup"
    TENSORBOARD_AVAILABLE=false
fi

# Check NVIDIA GPU
if command -v nvidia-smi &> /dev/null; then
    GPU_AVAILABLE=true
else
    echo "[WARNING] NVIDIA GPU not available, skipping GPU monitoring"
    GPU_AVAILABLE=false
fi

echo "========================================"
echo "1. Starting TensorBoard Monitoring"
if [ "$TENSORBOARD_AVAILABLE" = true ]; then
    echo "[$(date '+%H:%M:%S')] Starting TensorBoard..."
    # Start TensorBoard in background with screen/tmux if available
    if command -v screen &> /dev/null; then
        screen -dmS tensorboard bash -c "echo 'TensorBoard Monitor - Visit http://localhost:6006'; $PYTHON_CMD start_tensorboard.py"
    elif command -v tmux &> /dev/null; then
        tmux new-session -d -s tensorboard "echo 'TensorBoard Monitor - Visit http://localhost:6006'; $PYTHON_CMD start_tensorboard.py"
    else
        # Fallback: start in background
        nohup $PYTHON_CMD start_tensorboard.py > tensorboard.log 2>&1 &
        echo "TensorBoard PID: $!"
    fi
    sleep 2
    echo "[SUCCESS] TensorBoard started"
else
    echo "[SKIP] TensorBoard not available"
fi
echo ""

echo "========================================"
echo "2. Starting GPU Monitoring"
if [ "$GPU_AVAILABLE" = true ]; then
    echo "[$(date '+%H:%M:%S')] Starting GPU monitoring..."
    if command -v screen &> /dev/null; then
        screen -dmS gpu_monitor bash -c 'echo "GPU Status Monitor"; while true; do clear; echo "GPU Status Monitor"; nvidia-smi; sleep 5; done'
    elif command -v tmux &> /dev/null; then
        tmux new-session -d -s gpu_monitor 'echo "GPU Status Monitor"; while true; do clear; echo "GPU Status Monitor"; nvidia-smi; sleep 5; done'
    else
        # Fallback: create a simple monitoring script
        cat > gpu_monitor_simple.sh << 'EOF'
#!/bin/bash
echo "GPU Status Monitor"
while true; do
    clear
    echo "GPU Status Monitor"
    nvidia-smi
    sleep 5
done
EOF
        chmod +x gpu_monitor_simple.sh
        nohup ./gpu_monitor_simple.sh > gpu_monitor.log 2>&1 &
        echo "GPU Monitor PID: $!"
    fi
    echo "[SUCCESS] GPU monitoring started"
else
    echo "[SKIP] GPU not available"
fi
echo ""

echo "========================================"
echo "3. Starting System Monitoring"
echo "[$(date '+%H:%M:%S')] Starting system monitoring..."
if command -v screen &> /dev/null; then
    screen -dmS system_monitor bash -c 'echo "System Resource Monitor"; while true; do clear; echo "System Resource Monitor"; echo "[$(date "+%H:%M:%S")] System Status:"; echo "CPU Usage:"; top -bn1 | grep "Cpu(s)" | awk "{print \$2 \$3}"; echo "Memory Usage:"; free -h; echo "Disk Usage:"; df -h | head -5; sleep 10; done'
elif command -v tmux &> /dev/null; then
    tmux new-session -d -s system_monitor 'echo "System Resource Monitor"; while true; do clear; echo "System Resource Monitor"; echo "[$(date "+%H:%M:%S")] System Status:"; echo "CPU Usage:"; top -bn1 | grep "Cpu(s)" | awk "{print \$2 \$3}"; echo "Memory Usage:"; free -h; echo "Disk Usage:"; df -h | head -5; sleep 10; done'
else
    # Fallback: create a simple system monitoring script
    cat > system_monitor_simple.sh << 'EOF'
#!/bin/bash
echo "System Resource Monitor"
while true; do
    clear
    echo "System Resource Monitor"
    echo "[$(date "+%H:%M:%S")] System Status:"
    echo "CPU Usage:"
    top -bn1 | grep "Cpu(s)" | awk "{print $2 $3}"
    echo "Memory Usage:"
    free -h
    echo "Disk Usage:"
    df -h | head -5
    sleep 10
done
EOF
    chmod +x system_monitor_simple.sh
    nohup ./system_monitor_simple.sh > system_monitor.log 2>&1 &
    echo "System Monitor PID: $!"
fi
echo "[SUCCESS] System monitoring started"
echo ""

echo "========================================"
echo "4. Starting Training Log Monitoring"
if [ -d "outputs" ]; then
    echo "[$(date '+%H:%M:%S')] Starting training log monitoring..."
    if command -v screen &> /dev/null; then
        screen -dmS log_monitor bash -c 'echo "Training Log Monitor"; cd outputs; while true; do clear; echo "Training Log Monitor"; echo "Available log files:"; ls -la *.log 2>/dev/null || echo "No log files found"; echo "Latest log entries:"; tail -n 10 *.log 2>/dev/null | head -20; sleep 15; done'
    elif command -v tmux &> /dev/null; then
        tmux new-session -d -s log_monitor 'echo "Training Log Monitor"; cd outputs; while true; do clear; echo "Training Log Monitor"; echo "Available log files:"; ls -la *.log 2>/dev/null || echo "No log files found"; echo "Latest log entries:"; tail -n 10 *.log 2>/dev/null | head -20; sleep 15; done'
    else
        # Fallback: create a simple log monitoring script
        cat > log_monitor_simple.sh << 'EOF'
#!/bin/bash
echo "Training Log Monitor"
cd outputs 2>/dev/null || exit 1
while true; do
    clear
    echo "Training Log Monitor"
    echo "Available log files:"
    ls -la *.log 2>/dev/null || echo "No log files found"
    echo "Latest log entries:"
    tail -n 10 *.log 2>/dev/null | head -20
    sleep 15
done
EOF
        chmod +x log_monitor_simple.sh
        nohup ./log_monitor_simple.sh > log_monitor.log 2>&1 &
        echo "Log Monitor PID: $!"
    fi
    echo "[SUCCESS] Training log monitoring started"
else
    echo "[SKIP] Output directory does not exist"
fi
echo ""

echo "========================================"
echo "5. Starting Process Monitoring"
echo "[$(date '+%H:%M:%S')] Starting process monitoring..."
if command -v screen &> /dev/null; then
    screen -dmS process_monitor bash -c 'echo "Python Process Monitor"; while true; do clear; echo "Python Process Monitor"; echo "[$(date "+%H:%M:%S")] Python Training Processes:"; ps aux | grep python | grep -v grep | head -10; sleep 8; done'
elif command -v tmux &> /dev/null; then
    tmux new-session -d -s process_monitor 'echo "Python Process Monitor"; while true; do clear; echo "Python Process Monitor"; echo "[$(date "+%H:%M:%S")] Python Training Processes:"; ps aux | grep python | grep -v grep | head -10; sleep 8; done'
else
    # Fallback: create a simple process monitoring script
    cat > process_monitor_simple.sh << 'EOF'
#!/bin/bash
echo "Python Process Monitor"
while true; do
    clear
    echo "Python Process Monitor"
    echo "[$(date "+%H:%M:%S")] Python Training Processes:"
    ps aux | grep python | grep -v grep | head -10
    sleep 8
done
EOF
    chmod +x process_monitor_simple.sh
    nohup ./process_monitor_simple.sh > process_monitor.log 2>&1 &
    echo "Process Monitor PID: $!"
fi
echo "[SUCCESS] Process monitoring started"
echo ""

echo "========================================"
echo "        Monitoring Tools Started!"
echo "========================================"
echo ""
echo "Started monitoring sessions:"
if [ "$TENSORBOARD_AVAILABLE" = true ]; then
    echo "  * TensorBoard: http://localhost:6006"
fi
if [ "$GPU_AVAILABLE" = true ]; then
    echo "  * GPU Monitor: Real-time GPU status"
fi
echo "  * System Monitor: CPU/Memory/Disk status"
echo "  * Training Logs: Real-time training logs"
echo "  * Process Monitor: Python process status"
echo ""
echo "Session Management:"
if command -v screen &> /dev/null; then
    echo "  * View sessions: screen -ls"
    echo "  * Attach to session: screen -r <session_name>"
    echo "  * Detach from session: Ctrl+A, then D"
elif command -v tmux &> /dev/null; then
    echo "  * View sessions: tmux list-sessions"
    echo "  * Attach to session: tmux attach-session -t <session_name>"
    echo "  * Detach from session: Ctrl+B, then D"
else
    echo "  * Monitor processes: ps aux | grep monitor"
    echo "  * Kill process: kill <PID>"
    echo "  * View logs: tail -f <monitor_name>.log"
fi
echo ""
echo "Tips:"
echo "  * All monitors run in background sessions/processes"
echo "  * Re-run this script to restart all monitoring"
echo "  * Check individual log files for detailed output"
echo ""
read -p "Press Enter to exit launcher"