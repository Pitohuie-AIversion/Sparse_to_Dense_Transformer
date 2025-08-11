#!/bin/bash

# =============================================================================
# Training Management Script - Unified Training Management Interface
# Integrates GPU monitoring, system monitoring, training launch, log viewing and other functions
# =============================================================================

# Color definitions
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

# Script paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPU_MONITOR_SCRIPT="$SCRIPT_DIR/gpu_monitor.sh"
SYSTEM_MONITOR_SCRIPT="$SCRIPT_DIR/system_monitor.sh"
TRAINING_SCRIPT="$SCRIPT_DIR/run_training_optimized.sh"
QUICK_START_SCRIPT="$SCRIPT_DIR/quick_start.sh"

# Configuration files
CONFIG_FILE="$SCRIPT_DIR/configs/pressure_field_training.yaml"
DATA_PATH="$SCRIPT_DIR/data/pressure_field_data.pt"

# Display title
show_title() {
    clear
    echo -e "${BOLD}${PURPLE}"
    echo "╔══════════════════════════════════════════════════════════════════════════════╗"
    echo "║                         Training Management System v1.0                     ║"
    echo "║                      VIV Transformer Pressure Field Reconstruction         ║"
    echo "╚══════════════════════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
    echo -e "${CYAN}Server: node36 | GPU: 2x NVIDIA L40 | Time: $(date)${NC}"
    echo "="*80
}

# Display main menu
show_main_menu() {
    echo -e "${BOLD}${BLUE}=== Main Menu ===${NC}"
    echo ""
    echo -e "${GREEN}1.${NC} 🚀 Launch Training"
    echo -e "${GREEN}2.${NC} 📊 GPU Monitoring"
    echo -e "${GREEN}3.${NC} 🖥️  System Monitoring"
    echo -e "${GREEN}4.${NC} 📋 View Training Status"
    echo -e "${GREEN}5.${NC} 📁 Manage Training Tasks"
    echo -e "${GREEN}6.${NC} 📈 View Training Logs"
    echo -e "${GREEN}7.${NC} ⚙️  System Configuration"
    echo -e "${GREEN}8.${NC} 🔧 Toolbox"
    echo -e "${GREEN}9.${NC} ❓ Help Information"
    echo -e "${RED}0.${NC} 🚪 Exit"
    echo ""
    echo -n -e "${YELLOW}Please select an operation [0-9]: ${NC}"
}

# Training launch menu
show_training_menu() {
    clear
    show_title
    echo -e "${BOLD}${BLUE}=== Training Launch Menu ===${NC}"
    echo ""
    echo -e "${GREEN}1.${NC} 🎯 Quick Start Training (Recommended Configuration)"
    echo -e "${GREEN}2.${NC} ⚡ Optimized Training (Server Optimized)"
    echo -e "${GREEN}3.${NC} 🔧 Custom Training Configuration"
    echo -e "${GREEN}4.${NC} 📊 Performance Test Mode"
    echo -e "${GREEN}5.${NC} 🔄 Resume Training (From Checkpoint)"
    echo -e "${RED}0.${NC} 🔙 Return to Main Menu"
    echo ""
    echo -n -e "${YELLOW}Please select training mode [0-5]: ${NC}"
}

# GPU status check
check_gpu_status() {
    echo -e "${BLUE}=== GPU Status Check ===${NC}"
    
    if ! command -v nvidia-smi &> /dev/null; then
        echo -e "${RED}❌ NVIDIA drivers not installed or nvidia-smi unavailable${NC}"
        return 1
    fi
    
    # Display basic GPU information
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader,nounits | while IFS=',' read -r idx name mem_used mem_total util temp; do
        mem_used_gb=$((mem_used / 1024))
        mem_total_gb=$((mem_total / 1024))
        mem_usage_percent=$((mem_used * 100 / mem_total))
        
        # Status colors
        if [ $mem_usage_percent -lt 10 ] && [ $util -lt 10 ]; then
            status="${GREEN}Idle${NC}"
        elif [ $mem_usage_percent -lt 50 ] && [ $util -lt 50 ]; then
            status="${YELLOW}Light Load${NC}"
        else
            status="${RED}Heavy Load${NC}"
        fi
        
        echo -e "GPU $idx ($name): ${mem_used_gb}GB/${mem_total_gb}GB, ${util}%, ${temp}°C - $status"
    done
    
    echo ""
}

# System status check
check_system_status() {
    echo -e "${CYAN}=== System Status Check ===${NC}"
    
    # CPU and memory
    cpu_usage=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | awk -F'%' '{print $1}')
    mem_usage=$(free | awk '/^Mem:/ {printf "%.1f", $3/$2*100}')
    disk_usage=$(df -h . | tail -1 | awk '{print $5}')
    
    echo "CPU Usage: ${cpu_usage}%"
    echo "Memory Usage: ${mem_usage}%"
    echo "Disk Usage: $disk_usage"
    
    # Check training processes
    training_count=$(ps aux | grep -E '(python.*train|train_pressure_field)' | grep -v grep | wc -l)
    echo "Running Training Processes: $training_count"
    
    echo ""
}

# Dataset download functionality
download_dataset() {
    echo -e "${CYAN}=== Dataset Download ===${NC}"
    
    # Define dataset-related variables
    DATA_REPO_URL="https://github.com/pdebench/PDEBench.git"
    DATA_DIR="$SCRIPT_DIR/data"
    ALTERNATIVE_DATA_URLS=(
        "https://darus.uni-stuttgart.de/api/access/datafile/132004"
        "https://zenodo.org/record/6222489/files/2D_CFD_Rand_M0.1_Eta1e-08_Zeta1e-08_periodic_Train.hdf5"
    )
    
    echo "Data download options:"
    echo -e "${GREEN}1.${NC} Clone complete PDEBench repository (Recommended)"
    echo -e "${GREEN}2.${NC} Direct download pressure field data files"
    echo -e "${GREEN}3.${NC} Generate sample data (For testing)"
    echo -e "${RED}0.${NC} Cancel download"
    echo ""
    echo -n -e "${YELLOW}Please select download method [0-3]: ${NC}"
    read download_choice
    
    case $download_choice in
        1)
            echo -e "${GREEN}Cloning PDEBench repository...${NC}"
            
            if ! command -v git &> /dev/null; then
                echo -e "${RED}❌ Git not installed, please install git first${NC}"
                return 1
            fi
            
            mkdir -p "$DATA_DIR"
            
            if [ ! -d "$DATA_DIR/PDEBench" ]; then
                echo "Cloning repository..."
                if git clone --depth 1 "$DATA_REPO_URL" "$DATA_DIR/PDEBench"; then
                    echo -e "${GREEN}✓ Repository cloned successfully${NC}"
                    
                    # Search for data files
                    DOWNLOADED_FILES=($(find "$DATA_DIR/PDEBench" -name "*.hdf5" -o -name "*.pt" -o -name "*.h5" 2>/dev/null | head -5))
                    if [ ${#DOWNLOADED_FILES[@]} -gt 0 ]; then
                        DATA_PATH="${DOWNLOADED_FILES[0]}"
                        echo -e "${GREEN}✓ Found data file: $DATA_PATH${NC}"
                    fi
                else
                    echo -e "${RED}❌ Repository cloning failed${NC}"
                    return 1
                fi
            else
                echo -e "${GREEN}✓ PDEBench repository already exists${NC}"
                DOWNLOADED_FILES=($(find "$DATA_DIR/PDEBench" -name "*.hdf5" -o -name "*.pt" -o -name "*.h5" 2>/dev/null | head -5))
                if [ ${#DOWNLOADED_FILES[@]} -gt 0 ]; then
                    DATA_PATH="${DOWNLOADED_FILES[0]}"
                    echo -e "${GREEN}✓ Using existing data file: $DATA_PATH${NC}"
                fi
            fi
            ;;
        2)
            echo -e "${GREEN}Direct download data files...${NC}"
            
            if command -v wget &> /dev/null; then
                DOWNLOAD_CMD="wget -O"
            elif command -v curl &> /dev/null; then
                DOWNLOAD_CMD="curl -L -o"
            else
                echo -e "${RED}❌ wget or curl not found, cannot download${NC}"
                return 1
            fi
            
            mkdir -p "$DATA_DIR"
            
            for i in "${!ALTERNATIVE_DATA_URLS[@]}"; do
                url="${ALTERNATIVE_DATA_URLS[$i]}"
                filename="pressure_data_$((i+1)).hdf5"
                filepath="$DATA_DIR/$filename"
                
                echo "Attempting download: $url"
                if $DOWNLOAD_CMD "$filepath" "$url"; then
                    if [ -f "$filepath" ] && [ $(stat -c%s "$filepath" 2>/dev/null || stat -f%z "$filepath" 2>/dev/null) -gt 1000000 ]; then
                        DATA_PATH="$filepath"
                        echo -e "${GREEN}✓ Download successful: $DATA_PATH${NC}"
                        break
                    else
                        echo -e "${YELLOW}⚠️  Downloaded file too small, trying next source${NC}"
                        rm -f "$filepath"
                    fi
                else
                    echo -e "${YELLOW}⚠️  Download failed, trying next source${NC}"
                fi
            done
            ;;
        3)
            echo -e "${GREEN}Generating sample data...${NC}"
            
            mkdir -p "$DATA_DIR"
            
            # Create sample data generation script
            cat > "$DATA_DIR/generate_sample.py" << 'EOF'
import torch
import numpy as np
from pathlib import Path

def generate_sample_data():
    print("Generating sample pressure field data...")
    
    # Create small-scale sample data
    batch_size = 100
    height, width = 64, 64
    time_steps = 10
    
    # Generate pressure field data
    pressure_data = torch.randn(batch_size, time_steps, height, width, dtype=torch.float32)
    
    # Add boundary conditions
    pressure_data[:, :, 0, :] = 0  # Top boundary
    pressure_data[:, :, -1, :] = 0  # Bottom boundary
    pressure_data[:, :, :, 0] = 0  # Left boundary
    pressure_data[:, :, :, -1] = 0  # Right boundary
    
    # Save as PyTorch format
    output_file = Path("sample_pressure_data.pt")
    torch.save({
        'pressure': pressure_data,
        'time': torch.linspace(0, 1, time_steps),
        'metadata': {
            'description': 'Sample pressure field data for testing',
            'batch_size': batch_size,
            'spatial_dims': (height, width),
            'time_steps': time_steps
        }
    }, output_file)
    
    print(f"Sample data generated: {output_file}")
    print(f"Data shape: {pressure_data.shape}")
    return str(output_file)

if __name__ == "__main__":
    generate_sample_data()
EOF
            
            cd "$DATA_DIR"
            if python generate_sample.py; then
                DATA_PATH="$DATA_DIR/sample_pressure_data.pt"
                echo -e "${GREEN}✓ Sample data generated successfully: $DATA_PATH${NC}"
                echo -e "${YELLOW}⚠️  Note: This is sample data, for testing only${NC}"
            else
                echo -e "${RED}❌ Sample data generation failed${NC}"
                return 1
            fi
            cd - > /dev/null
            ;;
        0)
            echo "Cancel download"
            return 0
            ;;
        *)
            echo -e "${RED}Invalid selection${NC}"
            return 1
            ;;
    esac
    
    echo ""
    echo -n "Press Enter to continue..."
    read
}

# Check environment
check_environment() {
    echo -e "${PURPLE}=== Environment Check ===${NC}"
    
    # Python environment
    if command -v python &> /dev/null; then
        python_version=$(python --version 2>&1)
        echo -e "${GREEN}✓${NC} Python: $python_version"
    else
        echo -e "${RED}❌ Python not installed${NC}"
    fi
    
    # PyTorch
    if python -c "import torch" 2>/dev/null; then
        torch_version=$(python -c "import torch; print(torch.__version__)" 2>/dev/null)
        cuda_available=$(python -c "import torch; print(torch.cuda.is_available())" 2>/dev/null)
        echo -e "${GREEN}✓${NC} PyTorch: $torch_version (CUDA: $cuda_available)"
    else
        echo -e "${RED}❌ PyTorch not installed${NC}"
    fi
    
    # Data file check and download
    data_found=false
    
    # First check default path
    if [ -f "$DATA_PATH" ]; then
        data_size=$(du -h "$DATA_PATH" | cut -f1)
        echo -e "${GREEN}✓${NC} Data file: $data_size"
        data_found=true
    else
        # Search for other possible data files
        echo "Searching for existing data files..."
        DATA_FILES=($(find "$SCRIPT_DIR" -name "*.pt" -o -name "*.hdf5" -o -name "*.h5" -type f 2>/dev/null | head -5))
        
        if [ ${#DATA_FILES[@]} -gt 0 ]; then
            DATA_PATH="${DATA_FILES[0]}"
            data_size=$(du -h "$DATA_PATH" | cut -f1)
            echo -e "${GREEN}✓${NC} Found data file: $DATA_PATH ($data_size)"
            data_found=true
        else
            echo -e "${YELLOW}⚠️  Data file not found${NC}"
            echo -n -e "${YELLOW}Download dataset now? [y/N]: ${NC}"
            read download_now
            
            if [[ "$download_now" =~ ^[Yy]$ ]]; then
                download_dataset
                # Re-check data file
                if [ -f "$DATA_PATH" ]; then
                    data_size=$(du -h "$DATA_PATH" | cut -f1)
                    echo -e "${GREEN}✓${NC} Data file: $data_size"
                    data_found=true
                fi
            fi
        fi
    fi
    
    # Configuration file
    if [ -f "$CONFIG_FILE" ]; then
        echo -e "${GREEN}✓${NC} Configuration file: exists"
    else
        echo -e "${RED}❌ Configuration file not found: $CONFIG_FILE${NC}"
    fi
    
    echo ""
}

# Start quick training
start_quick_training() {
    echo -e "${GREEN}Starting quick training...${NC}"
    
    if [ -f "$QUICK_START_SCRIPT" ]; then
        chmod +x "$QUICK_START_SCRIPT"
        bash "$QUICK_START_SCRIPT"
    else
        echo -e "${RED}Quick start script not found: $QUICK_START_SCRIPT${NC}"
    fi
}

# Start optimized training
start_optimized_training() {
    echo -e "${GREEN}Starting optimized training...${NC}"
    
    if [ -f "$TRAINING_SCRIPT" ]; then
        chmod +x "$TRAINING_SCRIPT"
        bash "$TRAINING_SCRIPT"
    else
        echo -e "${RED}Optimized training script not found: $TRAINING_SCRIPT${NC}"
    fi
}

# GPU monitoring
start_gpu_monitor() {
    echo -e "${GREEN}Starting GPU monitoring...${NC}"
    
    if [ -f "$GPU_MONITOR_SCRIPT" ]; then
        chmod +x "$GPU_MONITOR_SCRIPT"
        bash "$GPU_MONITOR_SCRIPT"
    else
        echo -e "${RED}GPU monitoring script not found: $GPU_MONITOR_SCRIPT${NC}"
        echo "Using nvidia-smi as substitute:"
        watch -n 2 nvidia-smi
    fi
}

# System monitoring
start_system_monitor() {
    echo -e "${GREEN}Starting system monitoring...${NC}"
    
    if [ -f "$SYSTEM_MONITOR_SCRIPT" ]; then
        chmod +x "$SYSTEM_MONITOR_SCRIPT"
        bash "$SYSTEM_MONITOR_SCRIPT"
    else
        echo -e "${RED}System monitoring script not found: $SYSTEM_MONITOR_SCRIPT${NC}"
        echo "Using htop as substitute:"
        htop
    fi
}

# View training status
show_training_status() {
    clear
    show_title
    echo -e "${BOLD}${BLUE}=== Training Status ===${NC}"
    echo ""
    
    check_gpu_status
    check_system_status
    
    # View training process details
    echo -e "${YELLOW}=== Training Process Details ===${NC}"
    python_procs=$(ps aux | grep -E '(python.*train|train_pressure_field)' | grep -v grep)
    
    if [ -z "$python_procs" ]; then
        echo "No training processes currently running"
    else
        echo "$python_procs" | while read line; do
            pid=$(echo "$line" | awk '{print $2}')
            cpu=$(echo "$line" | awk '{print $3}')
            mem=$(echo "$line" | awk '{print $4}')
            runtime=$(ps -o etime= -p $pid 2>/dev/null | tr -d ' ')
            cmd=$(echo "$line" | awk '{for(i=11;i<=NF;i++) printf "%s ", $i; print ""}')
            
            echo "PID $pid: CPU ${cpu}%, MEM ${mem}%, Runtime $runtime"
            echo "Command: $(echo "$cmd" | cut -c1-70)..."
            echo ""
        done
    fi
    
    # TensorBoard status
    tb_proc=$(ps aux | grep tensorboard | grep -v grep)
    if [ -n "$tb_proc" ]; then
        tb_port=$(echo "$tb_proc" | grep -o 'port=[0-9]*' | cut -d'=' -f2 || echo "6006")
        server_ip=$(hostname -I | awk '{print $1}')
        echo -e "${GREEN}TensorBoard running: http://$server_ip:$tb_port${NC}"
    fi
    
    echo ""
    echo -n "Press Enter to return to main menu..."
    read
}

# Manage training tasks
manage_training_tasks() {
    clear
    show_title
    echo -e "${BOLD}${BLUE}=== Training Task Management ===${NC}"
    echo ""
    
    echo -e "${GREEN}1.${NC} 🔍 View All Training Processes"
    echo -e "${GREEN}2.${NC} ⏹️  Stop Training Process"
    echo -e "${GREEN}3.${NC} 🔄 Restart Training"
    echo -e "${GREEN}4.${NC} 📊 Start TensorBoard"
    echo -e "${GREEN}5.${NC} ⏹️  Stop TensorBoard"
    echo -e "${GREEN}6.${NC} 🧹 Clean Temporary Files"
    echo -e "${RED}0.${NC} 🔙 Return to Main Menu"
    echo ""
    echo -n -e "${YELLOW}Please select operation [0-6]: ${NC}"
    
    read choice
    case $choice in
        1)
            echo -e "${BLUE}=== All Training Processes ===${NC}"
            ps aux | grep -E '(python.*train|train_pressure_field|tensorboard)' | grep -v grep
            echo ""
            echo -n "Press Enter to continue..."
            read
            ;;
        2)
            echo -e "${YELLOW}Please enter PID of process to stop: ${NC}"
            read pid
            if [ -n "$pid" ]; then
                kill $pid 2>/dev/null && echo -e "${GREEN}Process $pid stopped${NC}" || echo -e "${RED}Failed to stop process${NC}"
            fi
            sleep 2
            ;;
        3)
            echo -e "${GREEN}Restart training feature under development...${NC}"
            sleep 2
            ;;
        4)
            echo -e "${GREEN}Starting TensorBoard...${NC}"
            nohup tensorboard --logdir=outputs --port=6006 --host=0.0.0.0 > tensorboard.log 2>&1 &
            echo "TensorBoard started on port 6006"
            sleep 2
            ;;
        5)
            pkill -f tensorboard && echo -e "${GREEN}TensorBoard stopped${NC}" || echo -e "${RED}Failed to stop TensorBoard${NC}"
            sleep 2
            ;;
        6)
            echo -e "${YELLOW}Cleaning temporary files...${NC}"
            rm -f *.log nohup.out
            echo -e "${GREEN}Cleanup completed${NC}"
            sleep 2
            ;;
        0)
            return
            ;;
        *)
            echo -e "${RED}Invalid selection${NC}"
            sleep 1
            ;;
    esac
    
    manage_training_tasks
}

# View training logs
view_training_logs() {
    clear
    show_title
    echo -e "${BOLD}${BLUE}=== Training Log Viewer ===${NC}"
    echo ""
    
    # Find log files
    log_files=$(find . -name "*.log" -o -name "training_*.txt" -o -name "nohup.out" 2>/dev/null | head -10)
    
    if [ -z "$log_files" ]; then
        echo "No log files found"
        echo -n "Press Enter to return..."
        read
        return
    fi
    
    echo "Available log files:"
    echo "$log_files" | nl
    echo ""
    echo -n "Please select log file number (or press Enter to return): "
    read choice
    
    if [ -n "$choice" ] && [ "$choice" -gt 0 ]; then
        log_file=$(echo "$log_files" | sed -n "${choice}p")
        if [ -f "$log_file" ]; then
            echo -e "${GREEN}Viewing log file: $log_file${NC}"
            echo "Press q to exit view"
            sleep 2
            less "$log_file"
        fi
    fi
}

# System configuration
show_system_config() {
    clear
    show_title
    echo -e "${BOLD}${BLUE}=== System Configuration ===${NC}"
    echo ""
    
    check_environment
    
    echo -e "${CYAN}=== Configuration File Paths ===${NC}"
    echo "Training config: $CONFIG_FILE"
    echo "Data path: $DATA_PATH"
    echo "Output directory: outputs/"
    echo ""
    
    echo -e "${YELLOW}=== Environment Variables ===${NC}"
    echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-Not set}"
    echo "OMP_NUM_THREADS: ${OMP_NUM_THREADS:-Not set}"
    echo "MKL_NUM_THREADS: ${MKL_NUM_THREADS:-Not set}"
    echo ""
    
    echo -n "Press Enter to return to main menu..."
    read
}

# Toolbox
show_toolbox() {
    clear
    show_title
    echo -e "${BOLD}${BLUE}=== Toolbox ===${NC}"
    echo ""
    
    echo -e "${GREEN}1.${NC} 🔧 Environment Test"
    echo -e "${GREEN}2.${NC} 📊 Performance Benchmark"
    echo -e "${GREEN}3.${NC} 🧹 Clean All Outputs"
    echo -e "${GREEN}4.${NC} 📦 Package Training Results"
    echo -e "${GREEN}5.${NC} 🔄 Reset Environment"
    echo -e "${GREEN}6.${NC} 📋 Generate System Report"
    echo -e "${RED}0.${NC} 🔙 Return to Main Menu"
    echo ""
    echo -n -e "${YELLOW}Please select tool [0-6]: ${NC}"
    
    read choice
    case $choice in
        1)
            echo -e "${GREEN}Running environment test...${NC}"
            python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')"
            echo -n "Press Enter to continue..."
            read
            ;;
        2)
            echo -e "${GREEN}Performance benchmark feature under development...${NC}"
            sleep 2
            ;;
        3)
            echo -e "${YELLOW}Cleaning all output files...${NC}"
            rm -rf outputs/* *.log nohup.out
            echo -e "${GREEN}Cleanup completed${NC}"
            sleep 2
            ;;
        4)
            echo -e "${GREEN}Packaging feature under development...${NC}"
            sleep 2
            ;;
        5)
            echo -e "${YELLOW}Reset environment feature under development...${NC}"
            sleep 2
            ;;
        6)
            echo -e "${GREEN}Generating system report...${NC}"
            {
                echo "=== System Report ==="
                echo "Time: $(date)"
                echo ""
                echo "=== GPU Information ==="
                nvidia-smi
                echo ""
                echo "=== System Information ==="
                uname -a
                echo ""
                echo "=== Memory Information ==="
                free -h
                echo ""
                echo "=== Disk Information ==="
                df -h
            } > system_report.txt
            echo -e "${GREEN}System report generated: system_report.txt${NC}"
            sleep 2
            ;;
        0)
            return
            ;;
        *)
            echo -e "${RED}Invalid selection${NC}"
            sleep 1
            ;;
    esac
    
    show_toolbox
}

# Display help
show_help() {
    clear
    show_title
    echo -e "${BOLD}${BLUE}=== Help Information ===${NC}"
    echo ""
    
    echo -e "${CYAN}=== Quick Start ===${NC}"
    echo "1. Select 'Launch Training' -> 'Quick Start Training'"
    echo "2. System will automatically detect GPU and start training"
    echo "3. Use 'GPU Monitoring' to view training status"
    echo ""
    
    echo -e "${CYAN}=== Script File Description ===${NC}"
    echo "• training_manager.sh    - Main management script (current)"
    echo "• gpu_monitor.sh         - GPU monitoring script"
    echo "• system_monitor.sh      - System monitoring script"
    echo "• run_training_optimized.sh - Optimized training script"
    echo "• quick_start.sh         - Quick start script"
    echo ""
    
    echo -e "${CYAN}=== Common Commands ===${NC}"
    echo "• nvidia-smi             - Check GPU status"
    echo "• htop                   - View system resources"
    echo "• ps aux | grep python   - View Python processes"
    echo "• kill <PID>             - Terminate process"
    echo ""
    
    echo -e "${CYAN}=== Troubleshooting ===${NC}"
    echo "• Training won't start: Check data files and configuration files"
    echo "• GPU memory insufficient: Reduce batch_size"
    echo "• Process stuck: Use task management to stop process"
    echo ""
    
    echo -n "Press Enter to return to main menu..."
    read
}

# Main loop
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
                    3) echo -e "${YELLOW}Custom configuration feature under development...${NC}"; sleep 2 ;;
                    4) echo -e "${YELLOW}Performance test feature under development...${NC}"; sleep 2 ;;
                    5) echo -e "${YELLOW}Resume training feature under development...${NC}"; sleep 2 ;;
                    0) continue ;;
                    *) echo -e "${RED}Invalid selection${NC}"; sleep 1 ;;
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
                echo -e "${GREEN}Thank you for using the Training Management System!${NC}"
                exit 0
                ;;
            *) 
                echo -e "${RED}Invalid selection, please try again${NC}"
                sleep 1
                ;;
        esac
    done
}

# Check dependencies
check_dependencies() {
    missing_deps=()
    
    # Check necessary commands
    for cmd in nvidia-smi python; do
        if ! command -v $cmd &> /dev/null; then
            missing_deps+=("$cmd")
        fi
    done
    
    if [ ${#missing_deps[@]} -gt 0 ]; then
        echo -e "${RED}Missing required dependencies: ${missing_deps[*]}${NC}"
        echo "Please install missing dependencies and run again"
        exit 1
    fi
}

# Initialize
init_environment() {
    # Create necessary directories
    mkdir -p outputs logs
    
    # Set script permissions
    chmod +x "$GPU_MONITOR_SCRIPT" 2>/dev/null
    chmod +x "$SYSTEM_MONITOR_SCRIPT" 2>/dev/null
    chmod +x "$TRAINING_SCRIPT" 2>/dev/null
    chmod +x "$QUICK_START_SCRIPT" 2>/dev/null
}

# Main program entry
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    # Check dependencies
    check_dependencies
    
    # Initialize environment
    init_environment
    
    # Start main loop
    main_loop
fi