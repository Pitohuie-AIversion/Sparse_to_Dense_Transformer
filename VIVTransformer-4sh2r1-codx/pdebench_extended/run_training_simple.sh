#!/bin/bash

# =============================================================================
# Pressure Field Training - Server Optimized One-Click Run Script
# Quick start training for server environments with complete resource management and monitoring
# =============================================================================

set -e

# Server configuration parameters (modify as needed)
DATA_PATH="${DATA_PATH:-./data/PDEBench/pdebench/data_download/data/2D/DarcyFlow/2D_DarcyFlow_beta100.0_Train.hdf5}"  # Data file path
CONFIG_PATH="pdebench_extended/configs/pressure_field_training.yaml"  # Configuration file path
OUTPUT_BASE="./outputs"  # Output base directory
EXPERIMENT_NAME="pressure_field_$(date +%Y%m%d_%H%M%S)"  # Experiment name
LOG_DIR="./logs"                                     # Log directory
MAX_RETRIES=3                                        # Maximum retry count
HEALTH_CHECK_INTERVAL=300                           # Health check interval (seconds)

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${BLUE}=== Pressure Field Training One-Click Startup Script ===${NC}"
echo "Time: $(date)"
echo "Host: $(hostname)"
echo ""

# Server environment check and initialization
echo -e "${CYAN}=== Server Environment Check ===${NC}"

# Create necessary directories
mkdir -p "$LOG_DIR" "$OUTPUT_BASE"

# Log file setup
SCRIPT_LOG="$LOG_DIR/training_script_$(date +%Y%m%d_%H%M%S).log"
SYSTEM_LOG="$LOG_DIR/system_monitor_$(date +%Y%m%d_%H%M%S).log"

# Log functions
log_info() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [INFO] $1" | tee -a "$SCRIPT_LOG"
}

log_error() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [ERROR] $1" | tee -a "$SCRIPT_LOG" >&2
}

log_warn() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [WARN] $1" | tee -a "$SCRIPT_LOG"
}

# System information collection
log_info "Starting server environment check"
echo -e "${BLUE}Host Information:${NC}"
echo "  Hostname: $(hostname)"
echo "  Operating System: $(uname -s)"
echo "  Kernel Version: $(uname -r)"
echo "  CPU Cores: $(nproc)"
echo "  Total Memory: $(free -h | awk '/^Mem:/ {print $2}')"
echo "  Available Disk Space: $(df -h . | awk 'NR==2 {print $4}' | head -1)"

# PyCharm environment running - Skip virtual environment check
if [[ -n "$PYCHARM_HOSTED" ]]; then
    log_warn "PyCharm environment detected, recommend running in server terminal"
    echo -e "${YELLOW}PyCharm environment detected${NC}"
    echo "Recommend running this script in server terminal for optimal performance"
    echo ""
fi

echo -e "${BLUE}Using current Python environment...${NC}"
echo "Note: Ensure required Python dependencies are installed"

# Server environment variable optimization settings
log_info "Configuring server environment variables"

# Dynamic thread setting based on CPU cores
CPU_CORES=$(nproc)
OPTIMAL_THREADS=$((CPU_CORES / 2))
if [[ $OPTIMAL_THREADS -lt 4 ]]; then
    OPTIMAL_THREADS=4
elif [[ $OPTIMAL_THREADS -gt 16 ]]; then
    OPTIMAL_THREADS=16
fi

export PYTHONPATH="$PWD/pdebench_extended:$PYTHONPATH"
export OMP_NUM_THREADS=$OPTIMAL_THREADS
export MKL_NUM_THREADS=$OPTIMAL_THREADS
export NUMEXPR_MAX_THREADS=$OPTIMAL_THREADS
export OPENBLAS_NUM_THREADS=$OPTIMAL_THREADS

# Server performance optimization
export MALLOC_TRIM_THRESHOLD_=100000
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

echo -e "${GREEN}Server environment variables optimized${NC}"
echo "  PYTHONPATH: $PYTHONPATH"
echo "  Thread setting: $OPTIMAL_THREADS (based on $CPU_CORES cores)"
echo "  Memory optimization: Enabled"
echo ""

# Server GPU configuration and management
echo -e "${BLUE}=== Server GPU Configuration ===${NC}"
log_info "Starting GPU environment detection and configuration"

# Detect GPU
if command -v nvidia-smi &> /dev/null; then
    echo "NVIDIA GPU environment detected:"
    
    # Detailed GPU information
    echo -e "${CYAN}Detailed GPU Information:${NC}"
    nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu --format=csv,noheader,nounits | while IFS=',' read -r idx name total used free util temp; do
        echo "  GPU $idx: $name"
        echo "    Memory: ${used}MB/${total}MB (Free: ${free}MB)"
        echo "    Utilization: ${util}% | Temperature: ${temp}°C"
    done
    echo ""
    
    # Get GPU count and status
    GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
    echo "Available GPU count: $GPU_COUNT"
    
    # Server GPU intelligent selection strategy
    if [[ -n "$CUDA_VISIBLE_DEVICES" ]]; then
        log_info "Using preset GPU: $CUDA_VISIBLE_DEVICES"
        echo -e "${GREEN}Using preset GPU: $CUDA_VISIBLE_DEVICES${NC}"
    elif [[ $GPU_COUNT -gt 1 ]]; then
        echo -e "${PURPLE}Multi-GPU server environment, selection strategy:${NC}"
        echo "1. Use GPU 0 (usually main GPU)"
        echo "2. Use GPU 1 (recommended for training)"
        echo "3. Use all GPUs for parallel training"
        echo "4. Auto-select least busy GPU (recommended)"
        echo "5. Manually specify GPU"
        read -p "Please select (1-5): " gpu_choice
        
        case $gpu_choice in
            1)
                export CUDA_VISIBLE_DEVICES=0
                log_info "Manually selected GPU 0"
                echo -e "${GREEN}Set to use GPU 0${NC}"
                ;;
            2)
                export CUDA_VISIBLE_DEVICES=1
                log_info "Manually selected GPU 1"
                echo -e "${GREEN}Set to use GPU 1${NC}"
                ;;
            3)
                export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((GPU_COUNT-1)))
                log_info "Enabled all GPU parallel training: $CUDA_VISIBLE_DEVICES"
                echo -e "${GREEN}Set to use all GPUs for parallel training: $CUDA_VISIBLE_DEVICES${NC}"
                ;;
            4)
                # Intelligent selection: Consider both memory usage and GPU utilization
                BEST_GPU=$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits | awk -F',' '{score = $2 + $3*10; print $1, score}' | sort -k2 -n | head -1 | cut -d' ' -f1)
                export CUDA_VISIBLE_DEVICES=$BEST_GPU
                log_info "Auto-selected optimal GPU: $BEST_GPU"
                echo -e "${GREEN}Intelligently selected GPU $BEST_GPU (lowest combined load)${NC}"
                ;;
            5)
                echo "Available GPUs: $(seq -s' ' 0 $((GPU_COUNT-1)))"
                read -p "Please enter GPU number (e.g.: 0,1): " manual_gpu
                export CUDA_VISIBLE_DEVICES=$manual_gpu
                log_info "Manually specified GPU: $manual_gpu"
                echo -e "${GREEN}Set to use GPU: $manual_gpu${NC}"
                ;;
            *)
                echo -e "${YELLOW}Invalid selection, auto-selecting optimal GPU${NC}"
                BEST_GPU=$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits | awk -F',' '{score = $2 + $3*10; print $1, score}' | sort -k2 -n | head -1 | cut -d' ' -f1)
                export CUDA_VISIBLE_DEVICES=$BEST_GPU
                log_info "Default selected optimal GPU: $BEST_GPU"
                ;;
        esac
    else
        export CUDA_VISIBLE_DEVICES=0
        log_info "Single GPU environment, using GPU 0"
        echo -e "${GREEN}Single GPU environment, using GPU 0${NC}"
    fi
    
    # Display final GPU configuration
    echo -e "${CYAN}Final GPU Configuration:${NC}"
    echo "  CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
    
    # GPU status monitoring
    echo -e "${CYAN}Selected GPU Current Status:${NC}"
    for gpu_id in $(echo $CUDA_VISIBLE_DEVICES | tr ',' ' '); do
        gpu_info=$(nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader,nounits -i $gpu_id)
        echo "  GPU $gpu_id: $gpu_info"
    done
    
else
    log_warn "NVIDIA GPU not detected, will use CPU training"
    echo -e "${YELLOW}NVIDIA GPU not detected, will use CPU training${NC}"
    echo -e "${RED}Warning: CPU training is extremely slow, strongly recommend using GPU server${NC}"
fi
echo ""

# Dataset download and check
echo -e "${BLUE}=== Dataset Check and Download ===${NC}"

# Define dataset-related variables (based on PDEBench official documentation)
DATA_REPO_URL="https://github.com/pdebench/PDEBench.git"
DATA_DIR="./data"
DATA_SUBDIR="$DATA_DIR/2D/CFD/2D_Train_Rand"

# PDEBench official data repository (DaRUS)
PDEBENCH_DATASET_DOI="doi:10.18419/darus-2986"
PDEBENCH_DATASET_URL="https://darus.uni-stuttgart.de/dataset.xhtml?persistentId=doi:10.18419/darus-2986"
PDEBENCH_MODELS_DOI="doi:10.18419/darus-2987"
PDEBENCH_MODELS_URL="https://darus.uni-stuttgart.de/dataset.xhtml?persistentId=doi:10.18419/darus-2987"

# Alternative download links (based on official documentation)
ALTERNATIVE_DATA_URLS=(
    "https://darus.uni-stuttgart.de/api/access/datafile/132004"  # 2D CFD data
    "https://darus.uni-stuttgart.de/api/access/datafile/132005"  # Backup CFD data
)

# Check if data file exists
if [[ ! -f "$DATA_PATH" ]]; then
    echo -e "${YELLOW}Data file not found: $DATA_PATH${NC}"
    echo "Searching for existing data files..."
    
    # Automatically search for .pt, .hdf5 and .h5 files
    DATA_FILES=($(find . -name "*.pt" -o -name "*.hdf5" -o -name "*.h5" -type f 2>/dev/null | head -10))
    
    if [[ ${#DATA_FILES[@]} -eq 0 ]]; then
        echo -e "${YELLOW}No existing data files found, starting dataset download...${NC}"
        
        # Ask user for download method (based on PDEBench official recommendations)
        echo "PDEBench dataset download options:"
        echo "1. Use PDEBench official download script (recommended, from DaRUS repository)"
        echo "2. Clone PDEBench repository from GitHub (includes code and small datasets)"
        echo "3. Direct download from DaRUS (fast)"
        echo "4. Skip download, manually specify data path"
        echo ""
        echo "Note: PDEBench official datasets are hosted in DaRUS data repository"
        echo "Dataset DOI: $PDEBENCH_DATASET_DOI"
        echo "Access URL: $PDEBENCH_DATASET_URL"
        read -p "Please select download method (1-4): " download_choice
        
        case $download_choice in
            1)
                echo -e "${BLUE}Using PDEBench official download script...${NC}"
                echo "Setting up PDEBench official data download..."
                
                # First clone PDEBench repo to get the download script
                if [[ ! -d "$DATA_DIR/PDEBench" ]]; then
                    echo "Cloning PDEBench repository to get official download script..."
                    git clone --depth 1 "$DATA_REPO_URL" "$DATA_DIR/PDEBench"
                fi
                
                if [[ -d "$DATA_DIR/PDEBench" ]]; then
                    echo -e "${GREEN}PDEBench repository ready${NC}"
                    echo "Downloading dataset using the official script..."
                    
                    # Check for official download script
                    if [[ -f "$DATA_DIR/PDEBench/pdebench/data_download/download_direct.py" ]]; then
                        echo "Found official download script, starting download..."
                        cd "$DATA_DIR/PDEBench"
                        
                        # Use the officially recommended method
                        echo "Available datasets:"
                        echo "  - 2D CFD (Computational Fluid Dynamics)"
                        echo "  - 1D Advection, Burgers, Reaction-Diffusion"
                        echo "  - 2D Shallow Water, Diffusion-Reaction"
                        echo "  - Compressible Navier-Stokes"
                        echo ""
                        echo "Downloading 2D CFD dataset (suitable for pressure field training)..."
                        
                        # Try using the official download script
                        if python pdebench/data_download/download_direct.py --root_folder "../../" --pde_name "CFD" --split "train" --resolution "all"; then
                            echo -e "${GREEN}Official data download succeeded${NC}"
                            # Search for downloaded files
                            DOWNLOADED_FILES=($(find "../../" -name "*CFD*.hdf5" -o -name "*CFD*.h5" 2>/dev/null | head -10))
                            if [[ ${#DOWNLOADED_FILES[@]} -gt 0 ]]; then
                                DATA_PATH="${DOWNLOADED_FILES[0]}"
                                echo -e "${GREEN}Found downloaded data file: $DATA_PATH${NC}"
                            fi
                        else
                            echo -e "${YELLOW}Official download script failed, trying manual download...${NC}"
                        fi
                        cd - > /dev/null
                    else
                        echo -e "${YELLOW}Official download script not found, switching to GitHub clone mode...${NC}"
                    fi
                else
                    echo -e "${RED}Failed to clone PDEBench repository${NC}"
                fi
                ;;
            2)
                echo -e "${BLUE}Cloning PDEBench repository from GitHub...${NC}"
                
                # Check if git is installed
                if ! command -v git &> /dev/null; then
                    echo -e "${RED}Error: git is not installed${NC}"
                    echo "Please install git: sudo yum install -y git"
                    exit 1
                fi
                
                # Check git lfs
                if ! command -v git-lfs &> /dev/null; then
                    echo -e "${YELLOW}Warning: git-lfs is not installed, large files may fail to download${NC}"
                    
                    # Check if in conda environment
                    if [[ -n "$CONDA_DEFAULT_ENV" ]] || command -v conda &> /dev/null; then
                        echo -e "${BLUE}Detected conda environment, trying to auto-install git-lfs...${NC}"
                        read -p "Try to install git-lfs via conda? (y/n): " install_choice
                        if [[ "$install_choice" =~ ^[Yy]$ ]]; then
                            echo "Installing git-lfs..."
                            if conda install -c conda-forge git-lfs -y; then
                                echo -e "${GREEN}git-lfs installation succeeded${NC}"
                            else
                                echo -e "${YELLOW}conda installation failed, trying pip...${NC}"
                                if pip install git-lfs; then
                                    echo -e "${GREEN}git-lfs installed via pip successfully${NC}"
                                else
                                    echo -e "${RED}Automatic installation failed${NC}"
                                fi
                            fi
                        fi
                    else
                        echo "Installation options:"
                        echo "1. System install (requires sudo): sudo yum install -y git-lfs"
                        echo "2. Conda install (recommended): conda install -c conda-forge git-lfs"
                        echo "3. Manual install to user directory"
                    fi
                fi
                
                # Create data directory
                mkdir -p "$DATA_DIR"
                
                # Clone repository (shallow clone to save time)
                if [[ ! -d "$DATA_DIR/PDEBench" ]]; then
                    echo "Cloning PDEBench repository..."
                    git clone --depth 1 "$DATA_REPO_URL" "$DATA_DIR/PDEBench"
                    
                    if [[ $? -eq 0 ]]; then
                        echo -e "${GREEN}Repository cloned successfully${NC}"
                        
                        # Try to download LFS files
                        cd "$DATA_DIR/PDEBench"
                        if command -v git-lfs &> /dev/null; then
                            echo "Downloading large files..."
                            git lfs pull
                        fi
                        cd - > /dev/null
                        
                        # Search for data files
                        echo "Searching for downloaded data files..."
                        DOWNLOADED_FILES=($(find "$DATA_DIR/PDEBench" -name "*.hdf5" -o -name "*.pt" -o -name "*.h5" 2>/dev/null | head -10))
                        if [[ ${#DOWNLOADED_FILES[@]} -gt 0 ]]; then
                            DATA_PATH="${DOWNLOADED_FILES[0]}"
                            echo -e "${GREEN}Found data file: $DATA_PATH${NC}"
                        else
                            echo -e "${YELLOW}No large data files found in GitHub repo${NC}"
                            echo -e "${BLUE}PDEBench datasets are hosted on DaRUS and need separate download${NC}"
                            echo ""
                            echo "Recommended data acquisition methods:"
                            echo "1. Visit the official data repository: $PDEBENCH_DATASET_URL"
                            echo "2. Use the official download script (if available):"
                            echo "   cd $DATA_DIR/PDEBench"
                            echo "   python pdebench/data_download/download_direct.py --help"
                            echo "3. Or choose option 3 to download directly from DaRUS"
                            echo ""
                            echo "Note: The GitHub repository mainly contains code; large datasets are on DaRUS"
                        fi
                    else
                        echo -e "${RED}Failed to clone repository${NC}"
                    fi
                else
                    echo -e "${GREEN}PDEBench repository already exists${NC}"
                    # Search existing files
                    echo "Searching for existing data files..."
                    DOWNLOADED_FILES=($(find "$DATA_DIR/PDEBench" -name "*.hdf5" -o -name "*.pt" -o -name "*.h5" 2>/dev/null | head -10))
                    if [[ ${#DOWNLOADED_FILES[@]} -gt 0 ]]; then
                        DATA_PATH="${DOWNLOADED_FILES[0]}"
                        echo -e "${GREEN}Using existing data file: $DATA_PATH${NC}"
                    else
                        echo -e "${YELLOW}No data files found in existing repository${NC}"
                        echo "Trying to update repository and download LFS files..."
                        cd "$DATA_DIR/PDEBench"
                        git pull
                        if command -v git-lfs &> /dev/null; then
                            echo "Downloading LFS files..."
                            git lfs pull
                            cd - > /dev/null
                            # Re-search
                            DOWNLOADED_FILES=($(find "$DATA_DIR/PDEBench" -name "*.hdf5" -o -name "*.pt" -o -name "*.h5" 2>/dev/null | head -10))
                            if [[ ${#DOWNLOADED_FILES[@]} -gt 0 ]]; then
                                DATA_PATH="${DOWNLOADED_FILES[0]}"
                                echo -e "${GREEN}Found data file after LFS download: $DATA_PATH${NC}"
                            fi
                        else
                            cd - > /dev/null
                            echo -e "${YELLOW}git-lfs is not installed, unable to download large files${NC}"
                        fi
                    fi
                fi
                ;;
            3)
                echo -e "${BLUE}Directly downloading data files from DaRUS...${NC}"
                echo "Downloading from PDEBench official data repository..."
                
                # Check for wget or curl
                if command -v wget &> /dev/null; then
                    DOWNLOAD_CMD="wget -O"
                elif command -v curl &> /dev/null; then
                    DOWNLOAD_CMD="curl -L -o"
                else
                    echo -e "${RED}Error: wget or curl not found${NC}"
                    echo "Please contact admin to install download tools:"
                    echo "  CentOS/RHEL: yum install -y wget curl"
                    echo "  Ubuntu/Debian: apt install -y wget curl"
                    echo "or choose other data acquisition methods"
                    exit 1
                fi
                
                mkdir -p "$DATA_DIR"
                
                echo "Available PDEBench datasets (from DaRUS repository):"
                echo "1. 2D CFD (Computational Fluid Dynamics) - suitable for pressure field training"
                echo "2. 1D Advection Equation"
                echo "3. 1D Burgers Equation"
                echo "4. 2D Shallow Water"
                echo "5. 2D Diffusion-Reaction"
                echo ""
                read -p "Please select dataset type to download (1-5): " dataset_choice
                
                case $dataset_choice in
                    1)
                        echo "Downloading 2D CFD dataset..."
                        dataset_name="2D_CFD"
                        ;;
                    2)
                        echo "Downloading 1D Advection dataset..."
                        dataset_name="1D_Advection"
                        ;;
                    3)
                        echo "Downloading 1D Burgers dataset..."
                        dataset_name="1D_Burgers"
                        ;;
                    4)
                        echo "Downloading 2D Shallow Water dataset..."
                        dataset_name="2D_SWE"
                        ;;
                    5)
                        echo "Downloading 2D Diffusion-Reaction dataset..."
                        dataset_name="2D_DiffReact"
                        ;;
                    *)
                        echo "Default selection: 2D CFD dataset"
                        dataset_name="2D_CFD"
                        ;;
                esac
                
                # Try to download the selected dataset
                for i in "${!ALTERNATIVE_DATA_URLS[@]}"; do
                    url="${ALTERNATIVE_DATA_URLS[$i]}"
                    filename="${dataset_name}_data_$((i+1)).hdf5"
                    filepath="$DATA_DIR/$filename"
                    
                    echo "Trying to download from DaRUS: $url"
                    if $DOWNLOAD_CMD "$filepath" "$url"; then
                        if [[ -f "$filepath" && $(stat -f%z "$filepath" 2>/dev/null || stat -c%s "$filepath" 2>/dev/null) -gt 1000000 ]]; then
                            DATA_PATH="$filepath"
                            echo -e "${GREEN}DaRUS data download successful: $DATA_PATH${NC}"
                            break
                        else
                            echo -e "${YELLOW}Downloaded file is too small, download may have failed${NC}"
                            rm -f "$filepath"
                        fi
                    else
                        echo -e "${YELLOW}Download failed, trying next source...${NC}"
                    fi
                done
                
                if [[ ! -f "$DATA_PATH" ]]; then
                    echo -e "${RED}Automatic download failed${NC}"
                    echo -e "${BLUE}Please manually visit PDEBench official data repository:${NC}"
                    echo "  Dataset: $PDEBENCH_DATASET_URL"
                    echo "  Pre-trained models: $PDEBENCH_MODELS_URL"
                    echo ""
                    echo "Download steps:"
                    echo "1. Visit the above links"
                    echo "2. Select the required dataset files"
                    echo "3. Download to $DATA_DIR directory"
                    echo "4. Re-run this script"
                    echo ""
                    echo "Optional actions:"
                    echo "1. Re-search for existing data files"
                    echo "2. Generate sample data for testing"
                    echo "3. Manually specify data file path"
                    echo "4. Exit script and download manually"
                    read -p "Please select (1-4): " fallback_choice
                    
                    case $fallback_choice in
                        1)
                            echo "Re-searching for data files..."
                            # Re-search
                            EXISTING_FILES=($(find . -name "*.hdf5" -o -name "*.pt" -o -name "*.h5" 2>/dev/null | head -10))
                            if [[ ${#EXISTING_FILES[@]} -gt 0 ]]; then
                                echo "Found the following data files:"
                                for i in "${!EXISTING_FILES[@]}"; do
                                    echo "$((i+1)). ${EXISTING_FILES[$i]}"
                                done
                                read -p "Please select file number (1-${#EXISTING_FILES[@]}): " file_choice
                                if [[ $file_choice -ge 1 && $file_choice -le ${#EXISTING_FILES[@]} ]]; then
                                    DATA_PATH="${EXISTING_FILES[$((file_choice-1))]}"
                                    echo -e "${GREEN}Selected data file: $DATA_PATH${NC}"
                                fi
                            else
                                echo -e "${YELLOW}No data files found${NC}"
                                echo "Suggestions:"
                                echo "1. Manually download dataset from DaRUS data repository"
                                echo "2. Ensure data file format is .hdf5, .h5 or .pt"
                                echo "3. Place data files in project directory or subdirectories"
                            fi
                            ;;
                        2)
                            echo -e "${BLUE}Generating sample data file...${NC}"
                            mkdir -p "$DATA_DIR/generated"
                            SAMPLE_DATA="$DATA_DIR/generated/sample_data.pt"
                            
                            # Create a simple sample data file
                            python3 -c "
import torch
import os

# Generate sample pressure field data
data = {
    'pressure': torch.randn(100, 64, 64),  # 100 samples, 64x64 grid
    'velocity_x': torch.randn(100, 64, 64),
    'velocity_y': torch.randn(100, 64, 64),
    'time_steps': torch.linspace(0, 1, 100)
}

os.makedirs(os.path.dirname('$SAMPLE_DATA'), exist_ok=True)
torch.save(data, '$SAMPLE_DATA')
print('Sample data generated: $SAMPLE_DATA')
" 2>/dev/null
                            
                            if [[ -f "$SAMPLE_DATA" ]]; then
                                DATA_PATH="$SAMPLE_DATA"
                                echo -e "${GREEN}Sample data generated successfully: $DATA_PATH${NC}"
                                echo -e "${YELLOW}Note: This is sample data, only for testing training process${NC}"
                            else
                                echo -e "${RED}Sample data generation failed${NC}"
                            fi
                            ;;
                        3)
                            read -p "Please enter the complete path to the data file: " manual_path
                            if [[ -f "$manual_path" ]]; then
                                DATA_PATH="$manual_path"
                                echo -e "${GREEN}Using manually specified data file: $DATA_PATH${NC}"
                            else
                                echo -e "${RED}Specified file does not exist: $manual_path${NC}"
                            fi
                            ;;
                        4)
                            echo -e "${YELLOW}Exiting script, please download data manually and re-run${NC}"
                            echo ""
                            echo -e "${BLUE}PDEBench Dataset Manual Download Guide:${NC}"
                            echo "1. Visit official data repository:"
                            echo "   Dataset: $PDEBENCH_DATASET_URL"
                            echo "   Pre-trained models: $PDEBENCH_MODELS_URL"
                            echo ""
                            echo "2. Select and download required datasets:"
                            echo "   - 2D CFD (Computational Fluid Dynamics)"
                            echo "   - 1D Advection, Burgers, Reaction-Diffusion"
                            echo "   - 2D Shallow Water, Diffusion-Reaction"
                            echo "   - Compressible Navier-Stokes"
                            echo ""
                            echo "3. Place downloaded files in the following directory:"
                            echo "   $DATA_DIR/"
                            echo ""
                            echo "4. Ensure file format is .hdf5, .h5 or .pt"
                            echo ""
                            echo "5. Re-run this script: bash $0"
                            echo ""
                            echo "Note: PDEBench dataset is hosted in DaRUS data repository, not in GitHub repository"
                            exit 0
                            ;;
                        *)
                            echo -e "${YELLOW}Invalid selection, skipping download${NC}"
                            echo "Please refer to PDEBench official documentation to get dataset"
                            ;;
                    esac
                fi
                ;;
            4)
                echo -e "${YELLOW}Skipping automatic download${NC}"
                echo "Please manually download PDEBench dataset:"
                echo "  Official data repository: $PDEBENCH_DATASET_URL"
                echo "  Dataset DOI: $PDEBENCH_DATASET_DOI"
                ;;
            *)
                echo -e "${YELLOW}Invalid selection, skipping download${NC}"
                echo "Please refer to PDEBench official documentation to get dataset"
                ;;
        esac
    
    # If all download sources fail, provide other options
    if [[ ! -f "$DATA_PATH" ]]; then
        echo -e "${RED}Data download failed${NC}"
        echo -e "${BLUE}PDEBench Dataset Acquisition Guide:${NC}"
        echo ""
        echo "Official recommended data acquisition methods:"
        echo "1. Visit DaRUS data repository: $PDEBENCH_DATASET_URL"
        echo "2. Download required dataset files to local"
        echo "3. Use PDEBench official download tool (if available)"
        echo ""
        echo "Optional actions:"
        echo "1. Re-search for existing data files"
        echo "2. Generate sample data for testing"
        echo "3. Manually specify data file path"
        echo "4. Exit script and download manually"
        read -p "Please select (1-4): " fallback_choice
        
        case $fallback_choice in
            1)
                echo "Re-searching for data files..."
                # Re-search
                EXISTING_FILES=($(find . -name "*.hdf5" -o -name "*.pt" -o -name "*.h5" 2>/dev/null | head -10))
                if [[ ${#EXISTING_FILES[@]} -gt 0 ]]; then
                    echo "Found the following data files:"
                    for i in "${!EXISTING_FILES[@]}"; do
                        echo "$((i+1)). ${EXISTING_FILES[$i]}"
                    done
                    read -p "Please select file number (1-${#EXISTING_FILES[@]}): " file_choice
                    if [[ $file_choice -ge 1 && $file_choice -le ${#EXISTING_FILES[@]} ]]; then
                        DATA_PATH="${EXISTING_FILES[$((file_choice-1))]}"
                        echo -e "${GREEN}Selected data file: $DATA_PATH${NC}"
                    fi
                else
                    echo -e "${YELLOW}No data files found${NC}"
                    echo "Suggestions:"
                    echo "1. Manually download dataset from DaRUS data repository"
                    echo "2. Ensure data file format is .hdf5, .h5 or .pt"
                    echo "3. Place data files in project directory or subdirectories"
                fi
                ;;
            2)
                echo -e "${BLUE}Generating sample data file...${NC}"
                mkdir -p "$DATA_DIR/generated"
                SAMPLE_DATA="$DATA_DIR/generated/sample_data.pt"
                
                # Create a simple sample data file
                python3 -c "
import torch
import os

# Generate sample pressure field data
data = {
    'pressure': torch.randn(100, 64, 64),  # 100 samples, 64x64 grid
    'velocity_x': torch.randn(100, 64, 64),
    'velocity_y': torch.randn(100, 64, 64),
    'time_steps': torch.linspace(0, 1, 100)
}

os.makedirs(os.path.dirname('$SAMPLE_DATA'), exist_ok=True)
torch.save(data, '$SAMPLE_DATA')
print('Sample data generated: $SAMPLE_DATA')
" 2>/dev/null
                
                if [[ -f "$SAMPLE_DATA" ]]; then
                    DATA_PATH="$SAMPLE_DATA"
                    echo -e "${GREEN}Sample data generated successfully: $DATA_PATH${NC}"
                    echo -e "${YELLOW}Note: This is sample data, only for testing training process${NC}"
                else
                    echo -e "${RED}Sample data generation failed${NC}"
                fi
                ;;
            3)
                read -p "Please enter the complete path to the data file: " manual_path
                if [[ -f "$manual_path" ]]; then
                    DATA_PATH="$manual_path"
                    echo -e "${GREEN}Using manually specified data file: $DATA_PATH${NC}"
                else
                    echo -e "${RED}Specified file does not exist: $manual_path${NC}"
                fi
                ;;
            4)
                echo -e "${YELLOW}Exiting script, please download data manually and re-run${NC}"
                echo ""
                echo -e "${BLUE}PDEBench Dataset Manual Download Guide:${NC}"
                echo "1. Visit official data repository:"
                echo "   Dataset: $PDEBENCH_DATASET_URL"
                echo "   Pre-trained models: $PDEBENCH_MODELS_URL"
                echo ""
                echo "2. Select and download required datasets:"
                echo "   - 2D CFD (Computational Fluid Dynamics)"
                echo "   - 1D Advection, Burgers, Reaction-Diffusion"
                echo "   - 2D Shallow Water, Diffusion-Reaction"
                echo "   - Compressible Navier-Stokes"
                echo ""
                echo "3. Place downloaded files in the following directory:"
                echo "   $DATA_DIR/"
                echo ""
                echo "4. Ensure file format is .hdf5, .h5 or .pt"
                echo ""
                echo "5. Re-run this script: bash $0"
                echo ""
                echo "Note: PDEBench dataset is hosted in DaRUS data repository, not in GitHub repository"
                exit 0
                ;;
            *)
                echo -e "${YELLOW}Invalid selection, skipping download${NC}"
                echo "Please refer to PDEBench official documentation to get dataset"
                ;;
        esac
    fi
else
    echo -e "${GREEN}Data file exists: $DATA_PATH${NC}"
fi

# Validate data file
if [[ -f "$DATA_PATH" ]]; then
    file_size=$(du -h "$DATA_PATH" 2>/dev/null | cut -f1 || echo "Unknown")
    echo -e "${GREEN}Data file validation passed${NC}"
    echo "File path: $DATA_PATH"
    echo "File size: $file_size"
else
    echo -e "${RED}Data file validation failed${NC}"
    exit 1
fi

# Check configuration file
if [[ ! -f "$CONFIG_PATH" ]]; then
    echo -e "${RED}Error: Configuration file does not exist: $CONFIG_PATH${NC}"
    exit 1
fi

# Create output directory
OUTPUT_DIR="$OUTPUT_BASE/$EXPERIMENT_NAME"
mkdir -p "$OUTPUT_DIR"

echo ""
echo -e "${BLUE}=== Training Configuration ===${NC}"
echo "Data file: $DATA_PATH"
echo "Config file: $CONFIG_PATH"
echo "Output directory: $OUTPUT_DIR"
echo "Experiment name: $EXPERIMENT_NAME"
echo ""

# Server training mode selection
echo -e "${PURPLE}=== Server Training Mode Selection ===${NC}"
log_info "Configuring training startup mode"

echo "Choose a suitable server running mode:"
echo "1. Foreground run (debug mode, real-time output visible)"
echo "2. Background run (suitable for long training, with full logs)"
echo "3. Screen session run (recommended, connect anytime to view)"
echo "4. Tmux session run (recommended for advanced users)"
echo "5. System service mode (most stable, suitable for production)"
read -p "Please select (1-5): " mode

# 构建增强的训练命令
TRAIN_CMD="python pdebench_extended/train_pressure_field.py \
    --config \"$CONFIG_PATH\" \
    --data_path \"$DATA_PATH\" \
    --output_dir \"$OUTPUT_DIR\" \
    --experiment_name \"$EXPERIMENT_NAME\""

# Add server monitoring command
MONITOR_CMD="while true; do echo \"[\$(date)] GPU Status:\"; nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader,nounits; echo \"System load: \$(uptime)\"; sleep $HEALTH_CHECK_INTERVAL; done"

case $mode in
    1)
        echo -e "${GREEN}Foreground debug mode starting...${NC}"
        log_info "Starting foreground training mode"
        echo "Press Ctrl+C to stop training"
        echo "Real-time output mode, suitable for debugging and short runs"
        echo ""
        eval "$TRAIN_CMD"
        ;;
    2)
        echo -e "${GREEN}Background service mode starting...${NC}"
        log_info "Starting background training mode"
        
        # Create detailed background run script
        BACKGROUND_SCRIPT="$OUTPUT_DIR/run_background.sh"
        cat > "$BACKGROUND_SCRIPT" << EOF
#!/bin/bash
set -e

# Environment variables
export PYTHONPATH="$PWD/pdebench_extended:\$PYTHONPATH"
export OMP_NUM_THREADS=$OPTIMAL_THREADS
export MKL_NUM_THREADS=$OPTIMAL_THREADS
export CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES"

# Start system monitoring
$MONITOR_CMD > "$SYSTEM_LOG" 2>&1 &
MONITOR_PID=\$!
echo \$MONITOR_PID > "$OUTPUT_DIR/monitor.pid"

# Start training
echo "[\$(date)] Start training" >> "$OUTPUT_DIR/training.log"
$TRAIN_CMD >> "$OUTPUT_DIR/training.log" 2>&1
TRAIN_EXIT_CODE=\$?

# Clean up monitoring process
kill \$MONITOR_PID 2>/dev/null || true

echo "[\$(date)] Training finished, exit code: \$TRAIN_EXIT_CODE" >> "$OUTPUT_DIR/training.log"
exit \$TRAIN_EXIT_CODE
EOF
        
        chmod +x "$BACKGROUND_SCRIPT"
        nohup bash "$BACKGROUND_SCRIPT" &
        TRAIN_PID=$!
        echo $TRAIN_PID > "$OUTPUT_DIR/train.pid"
        
        echo "Training PID: $TRAIN_PID"
        echo "Training log: $OUTPUT_DIR/training.log"
        echo "System monitor log: $SYSTEM_LOG"
        echo ""
        echo "Monitoring commands:"
        echo "  tail -f $OUTPUT_DIR/training.log    # Training log"
        echo "  tail -f $SYSTEM_LOG                 # System monitor"
        echo "  ps aux | grep $TRAIN_PID            # Process status"
        echo "Stop command:"
        echo "  kill $TRAIN_PID                     # Stop training"
        ;;
    3)
        echo -e "${GREEN}Screen session mode starting...${NC}"
        log_info "Starting Screen session training mode"
        
        # Check if screen is installed
        if ! command -v screen &> /dev/null; then
            echo -e "${RED}Error: screen not installed${NC}"
            echo "Please ask admin to install screen, or use other run modes:"
            echo "  CentOS/RHEL: yum install -y screen"
            echo "  Ubuntu/Debian: apt-get install -y screen"
            echo "  Or choose mode 1 (Foreground) or mode 2 (Background)"
            read -p "Continue with background mode? (y/n): " -n 1 -r
            echo
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                echo "Switching to background mode..."
                mode=2
            else
                exit 1
            fi
        fi
        
        SESSION_NAME="pressure_training_$(date +%H%M%S)"
        
        # Create screen session and run training
        screen -dmS "$SESSION_NAME" bash -c "
            export PYTHONPATH=$PWD/pdebench_extended:\$PYTHONPATH
            export OMP_NUM_THREADS=$OPTIMAL_THREADS
            export MKL_NUM_THREADS=$OPTIMAL_THREADS
            export CUDA_VISIBLE_DEVICES='$CUDA_VISIBLE_DEVICES'
            
            echo '=== Pressure Training Session Started ==='
            echo 'Time: \$(date)'
            echo 'GPU: $CUDA_VISIBLE_DEVICES'
            echo 'Output dir: $OUTPUT_DIR'
            echo ''
            
            # Start monitoring
            $MONITOR_CMD > '$SYSTEM_LOG' 2>&1 &
            MONITOR_PID=\$!
            
            # Start training
            $TRAIN_CMD
            TRAIN_EXIT_CODE=\$?
            
            # Cleanup
            kill \$MONITOR_PID 2>/dev/null || true
            
            echo ''
            echo '=== Training Finished ==='
            echo 'Exit code: '\$TRAIN_EXIT_CODE
            echo 'Press any key to exit the session...'
            read
        "
        
        echo "Screen session created: $SESSION_NAME"
        echo ""
        echo "Management commands:"
        echo "  screen -r $SESSION_NAME              # Attach to session"
        echo "  screen -ls                           # List sessions"
        echo "  Ctrl+A+D                           # Detach (keep running)"
        echo "  screen -X -S $SESSION_NAME quit     # Kill session"
        
        # Wait and verify session status
        sleep 2
        if screen -list | grep -q "$SESSION_NAME"; then
            echo -e "${GREEN}Screen training session started${NC}"
            log_info "Screen session $SESSION_NAME started successfully"
        else
            echo -e "${RED}Failed to start Screen session${NC}"
            log_error "Screen session failed to start"
        fi
        ;;
    4)
        echo -e "${GREEN}Tmux session mode starting...${NC}"
        log_info "Starting Tmux session training mode"
        
        # Check if tmux is installed
        if ! command -v tmux &> /dev/null; then
            echo -e "${RED}Error: tmux not installed${NC}"
            echo "Please ask admin to install tmux, or use other run modes:"
            echo "  CentOS/RHEL: yum install -y tmux"
            echo "  Ubuntu/Debian: apt-get install -y tmux"
            echo "  Or choose mode 3 (Screen) or mode 2 (Background)"
            read -p "Switch to Screen mode? (y/n): " -n 1 -r
            echo
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                echo "Switching to Screen mode..."
                mode=3
            else
                read -p "Switch to background mode? (y/n): " -n 1 -r
                echo
                if [[ $REPLY =~ ^[Yy]$ ]]; then
                    echo "Switching to background mode..."
                    mode=2
                else
                    exit 1
                fi
            fi
        fi
        
        SESSION_NAME="pressure_training_$(date +%H%M%S)"
        
        # Create tmux session
        tmux new-session -d -s "$SESSION_NAME" -c "$PWD"
        
        # Set environment variables
        tmux send-keys -t "$SESSION_NAME" "export PYTHONPATH=$PWD/pdebench_extended:\$PYTHONPATH" Enter
        tmux send-keys -t "$SESSION_NAME" "export OMP_NUM_THREADS=$OPTIMAL_THREADS" Enter
        tmux send-keys -t "$SESSION_NAME" "export MKL_NUM_THREADS=$OPTIMAL_THREADS" Enter
        tmux send-keys -t "$SESSION_NAME" "export CUDA_VISIBLE_DEVICES='$CUDA_VISIBLE_DEVICES'" Enter
        
        # Split panes: left for training, right for monitoring
        tmux split-window -h -t "$SESSION_NAME"
        
        # Start monitoring on right pane
        tmux send-keys -t "$SESSION_NAME:0.1" "$MONITOR_CMD" Enter
        
        # Start training on left pane
        tmux send-keys -t "$SESSION_NAME:0.0" "$TRAIN_CMD" Enter
        
        echo "Tmux session created: $SESSION_NAME"
        echo ""
        echo "Management commands:"
        echo "  tmux attach -t $SESSION_NAME         # Attach to session"
        echo "  tmux list-sessions                   # List sessions"
        echo "  Ctrl+B, D                          # Detach"
        echo "  tmux kill-session -t $SESSION_NAME  # Kill session"
        echo "  Ctrl+B, %                          # Vertical split"
        echo "  Ctrl+B, Arrow Keys                  # Switch panes"
        
        log_info "Tmux session $SESSION_NAME started successfully"
        echo -e "${GREEN}Tmux training session started, left: training, right: monitoring${NC}"
        ;;
    5)
        echo -e "${GREEN}System service mode starting...${NC}"
        log_info "Starting system service mode"
        
        # Create systemd service file
        SERVICE_NAME="pressure-training-$(date +%H%M%S)"
        SERVICE_FILE="/tmp/$SERVICE_NAME.service"
        
        cat > "$SERVICE_FILE" << EOF
[Unit]
Description=Pressure Field Training Service
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$PWD
Environment=PYTHONPATH=$PWD/pdebench_extended:\$PYTHONPATH
Environment=OMP_NUM_THREADS=$OPTIMAL_THREADS
Environment=MKL_NUM_THREADS=$OPTIMAL_THREADS
Environment=CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
ExecStart=/bin/bash -c '$TRAIN_CMD'
StandardOutput=append:$OUTPUT_DIR/training.log
StandardError=append:$OUTPUT_DIR/training.log
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF
        
        echo "Systemd service file created: $SERVICE_FILE"
        echo "You don't have sudo, cannot install system service directly."
        echo "Alternatives:"
        echo ""
        echo "1. User-level systemd service (recommended):"
        USER_SERVICE_DIR="$HOME/.config/systemd/user"
        mkdir -p "$USER_SERVICE_DIR"
        USER_SERVICE_FILE="$USER_SERVICE_DIR/$SERVICE_NAME.service"
        cp "$SERVICE_FILE" "$USER_SERVICE_FILE"
        
        echo "  systemctl --user daemon-reload"
        echo "  systemctl --user start $SERVICE_NAME"
        echo "  systemctl --user enable $SERVICE_NAME  # Autostart on login"
        echo ""
        echo "User service management commands:"
        echo "  systemctl --user status $SERVICE_NAME   # Status"
        echo "  systemctl --user stop $SERVICE_NAME     # Stop"
        echo "  systemctl --user restart $SERVICE_NAME  # Restart"
        echo "  journalctl --user -u $SERVICE_NAME -f   # Logs"
        echo ""
        echo "2. Or use crontab on @reboot:"
        echo "  crontab -e"
        echo "  Add: @reboot cd $PWD && bash $BACKGROUND_SCRIPT"
        echo ""
        echo "3. Or choose other run mode (recommend Screen or Tmux)"
        
        log_info "User-level service configured: $SERVICE_NAME"
        echo -e "${GREEN}User service file created: $USER_SERVICE_FILE${NC}"
        ;;
    *)
        echo -e "${RED}Invalid selection${NC}"
        log_error "Invalid run mode selection: $mode"
        exit 1
        ;;
esac

# Server TensorBoard monitoring configuration
if [ "$mode" != "1" ]; then
echo -e "${CYAN}=== TensorBoard Monitoring Configuration ===${NC}"
read -p "Start TensorBoard monitoring? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
log_info "Configuring TensorBoard monitoring service"
echo -e "${BLUE}Starting server TensorBoard monitoring...${NC}"

# Intelligent port selection
TB_PORT=6006
for port in 6006 6007 6008 6009 6010; do
    if ! netstat -tlnp 2>/dev/null | grep -q ":$port "; then
        TB_PORT=$port
        break
    fi
done

if netstat -tlnp 2>/dev/null | grep -q ":$TB_PORT "; then
echo -e "${YELLOW}All common ports are occupied, using random port${NC}"
TB_PORT=$((RANDOM % 10000 + 10000))
fi

# Create TensorBoard startup script
TB_SCRIPT="$OUTPUT_DIR/start_tensorboard.sh"
cat > "$TB_SCRIPT" << EOF
#!/bin/bash
set -e

# TensorBoard configuration
export TENSORBOARD_PORT=$TB_PORT
export TENSORBOARD_HOST=0.0.0.0
export TENSORBOARD_LOGDIR="$OUTPUT_BASE"

# Start TensorBoard
echo "[\$(date)] Starting TensorBoard monitoring service" >> "$LOG_DIR/tensorboard.log"
tensorboard --logdir="\$TENSORBOARD_LOGDIR" --host="\$TENSORBOARD_HOST" --port="\$TENSORBOARD_PORT" --reload_interval=30 --samples_per_plugin=1000 >> "$LOG_DIR/tensorboard.log" 2>&1
EOF

chmod +x "$TB_SCRIPT"
nohup bash "$TB_SCRIPT" > "$LOG_DIR/tensorboard_startup.log" 2>&1 &
TB_PID=$!
echo $TB_PID > "$OUTPUT_DIR/tensorboard.pid"

# Get server network information
SERVER_IP=$(hostname -I | awk '{print $1}' | head -1)
EXTERNAL_IP=$(curl -s ifconfig.me 2>/dev/null || echo "unknown")

echo -e "${GREEN}TensorBoard monitoring service started${NC}"
echo "Service configuration:"
echo "  Port: $TB_PORT"
echo "  PID: $TB_PID"
echo "  Log: $LOG_DIR/tensorboard.log"
echo ""
echo "Access addresses:"
echo "  Internal access: http://$SERVER_IP:$TB_PORT"
if [[ "$EXTERNAL_IP" != "unknown" ]]; then
echo "  External access: http://$EXTERNAL_IP:$TB_PORT (requires firewall port opening)"
fi
echo ""
echo "Management commands:"
echo "  kill $TB_PID                        # Stop TensorBoard"
echo "  tail -f $LOG_DIR/tensorboard.log    # View logs"
echo "  netstat -tlnp | grep $TB_PORT       # Check port status"
echo ""
# Firewall tips
echo -e "${YELLOW}Firewall configuration tips (requires administrator privileges):${NC}"
echo "  CentOS/RHEL: firewall-cmd --add-port=$TB_PORT/tcp --permanent && firewall-cmd --reload"
echo "  Ubuntu: ufw allow $TB_PORT"
echo "  Or contact system administrator to open port $TB_PORT"

log_info "TensorBoard service startup completed, port: $TB_PORT"
fi
fi

# Server training startup completion summary
echo ""
echo -e "${GREEN}=== Server Training Startup Completed ===${NC}"
log_info "Training startup process completed"

echo -e "${CYAN}Training Configuration Information:${NC}"
echo "  Experiment name: $EXPERIMENT_NAME"
echo "  Data file: $DATA_PATH"
echo "  Configuration file: $CONFIG_PATH"
echo "  Output directory: $OUTPUT_DIR"
echo "  GPU device: $CUDA_VISIBLE_DEVICES"
echo "  Thread configuration: $OPTIMAL_THREADS"

echo ""
echo -e "${CYAN}Log File Locations:${NC}"
echo "  Script log: $SCRIPT_LOG"
echo "  Training log: $OUTPUT_DIR/training.log"
echo "  System monitoring: $SYSTEM_LOG"
if [[ $TB_PID ]]; then
echo "  TensorBoard log: $LOG_DIR/tensorboard.log"
fi

echo ""
echo -e "${CYAN}Server Monitoring Commands:${NC}"
echo "  # System resource monitoring"
echo "  htop                                    # Interactive system monitoring"
echo "  top -p \$(cat $OUTPUT_DIR/train.pid 2>/dev/null || echo 1)  # Training process monitoring"
echo "  iostat -x 1                            # Disk I/O monitoring"
echo "  free -h                                # Memory usage"
echo "  df -h                                  # Disk space"
echo ""
echo "  # GPU monitoring"
echo "  nvidia-smi                             # GPU status"
echo "  watch -n 1 nvidia-smi                 # Real-time GPU monitoring"
echo "  nvidia-smi dmon                       # GPU performance monitoring"
echo ""
echo "  # Training log monitoring"
echo "  tail -f $OUTPUT_DIR/training.log       # Real-time training log"
echo "  tail -f $SYSTEM_LOG                    # Real-time system monitoring"
echo "  grep -i error $OUTPUT_DIR/training.log # Find error information"
echo ""
echo "  # Session management"
echo "  screen -ls                             # View Screen sessions"
echo "  tmux list-sessions                     # View Tmux sessions"
echo "  ps aux | grep python                   # View Python processes"

echo ""
echo -e "${CYAN}Troubleshooting Commands:${NC}"
echo "  # Process management"
echo "  kill \$(cat $OUTPUT_DIR/train.pid 2>/dev/null)     # Stop training"
echo "  kill \$(cat $OUTPUT_DIR/tensorboard.pid 2>/dev/null) # Stop TensorBoard"
echo "  pkill -f 'python.*train_pressure_field'          # Force stop training"
echo ""
echo "  # Resource cleanup"
echo "  nvidia-smi --gpu-reset                           # Reset GPU"
echo "  sync                                             # Sync filesystem"
echo "  # Clear system cache requires administrator privileges: echo 3 > /proc/sys/vm/drop_caches"
echo ""
echo "  # Network check"
echo "  netstat -tlnp | grep 6006                        # Check TensorBoard port"
echo "  ss -tulpn | grep python                          # View Python network connections"

echo ""
echo -e "${CYAN}Performance Optimization Suggestions:${NC}"
echo "  1. Monitor GPU utilization, keep above 80%"
echo "  2. Pay attention to memory usage, avoid OOM errors"
echo "  3. Regularly check disk space, ensure sufficient storage"
echo "  4. Use TensorBoard to monitor training progress"
echo "  5. Regularly save checkpoints to prevent accidental interruption"

echo ""
echo -e "${GREEN}🚀 Server training environment ready!${NC}"
echo -e "${GREEN}📊 Monitoring address: http://$(hostname -I | awk '{print $1}'):${TB_PORT:-6006}${NC}"
echo -e "${GREEN}📝 Complete log: $SCRIPT_LOG${NC}"
echo ""
echo -e "${YELLOW}Tip: Recommend regularly checking training status and system resource usage${NC}"
echo -e "${YELLOW}If you encounter problems, please check log files or contact technical support${NC}"

log_info "Server training startup process fully completed"
fi