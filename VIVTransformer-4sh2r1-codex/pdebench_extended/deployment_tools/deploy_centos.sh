#!/bin/bash

# =============================================================================
# CentOS 服务器一键部署脚本 - 压力场训练系统
# 适用于 CentOS 7/8 系统
# =============================================================================

set -e  # 遇到错误立即退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

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

# 检查是否为root用户
check_root() {
    if [[ $EUID -eq 0 ]]; then
        log_warning "检测到root用户，建议使用普通用户运行"
        read -p "是否继续? (y/n): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
}

# 检测系统版本
detect_os() {
    if [[ -f /etc/redhat-release ]]; then
        OS_VERSION=$(cat /etc/redhat-release)
        log_info "检测到系统: $OS_VERSION"
    else
        log_error "不支持的操作系统，请使用CentOS 7/8"
        exit 1
    fi
}

# 安装系统依赖
install_system_deps() {
    log_info "安装系统依赖包..."
    
    # 更新系统
    sudo yum update -y
    
    # 安装基础工具
    sudo yum groupinstall -y "Development Tools"
    sudo yum install -y \
        git \
        wget \
        curl \
        vim \
        htop \
        screen \
        tmux \
        tree \
        unzip \
        bzip2-devel \
        openssl-devel \
        libffi-devel \
        sqlite-devel \
        readline-devel \
        tk-devel \
        gdbm-devel \
        db4-devel \
        libpcap-devel \
        xz-devel \
        expat-devel
    
    log_success "系统依赖安装完成"
}

# 安装Python 3.9
install_python() {
    log_info "检查Python版本..."
    
    if command -v python3.9 &> /dev/null; then
        PYTHON_VERSION=$(python3.9 --version)
        log_success "Python已安装: $PYTHON_VERSION"
        return
    fi
    
    log_info "安装Python 3.9..."
    
    # 下载Python源码
    cd /tmp
    wget https://www.python.org/ftp/python/3.9.18/Python-3.9.18.tgz
    tar xzf Python-3.9.18.tgz
    cd Python-3.9.18
    
    # 编译安装
    ./configure --enable-optimizations --with-ensurepip=install
    make -j$(nproc)
    sudo make altinstall
    
    # 创建软链接
    sudo ln -sf /usr/local/bin/python3.9 /usr/local/bin/python3
    sudo ln -sf /usr/local/bin/pip3.9 /usr/local/bin/pip3
    
    # 清理
    cd /
    rm -rf /tmp/Python-3.9.18*
    
    log_success "Python 3.9 安装完成"
}

# 安装CUDA (可选)
install_cuda() {
    read -p "是否安装CUDA? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        log_info "跳过CUDA安装"
        return
    fi
    
    log_info "安装CUDA 11.8..."
    
    # 添加CUDA仓库
    sudo yum-config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel7/x86_64/cuda-rhel7.repo
    sudo yum clean all
    
    # 安装CUDA
    sudo yum install -y cuda-11-8
    
    # 设置环境变量
    echo 'export PATH=/usr/local/cuda-11.8/bin:$PATH' >> ~/.bashrc
    echo 'export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
    
    log_success "CUDA安装完成，请重新登录以生效环境变量"
}

# 创建项目目录
setup_project() {
    log_info "设置项目目录..."
    
    PROJECT_DIR="$HOME/pressure_field_training"
    mkdir -p $PROJECT_DIR
    cd $PROJECT_DIR
    
    log_success "项目目录创建: $PROJECT_DIR"
}

# 创建虚拟环境
setup_venv() {
    log_info "创建Python虚拟环境..."
    
    python3.9 -m venv venv
    source venv/bin/activate
    
    # 升级pip
    pip install --upgrade pip setuptools wheel
    
    log_success "虚拟环境创建完成"
}

# 安装Python依赖
install_python_deps() {
    log_info "安装Python依赖包..."
    
    # 激活虚拟环境
    source venv/bin/activate
    
    # 安装PyTorch (CPU版本)
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    
    # 如果有CUDA，安装GPU版本
    if command -v nvcc &> /dev/null; then
        log_info "检测到CUDA，安装GPU版本PyTorch..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    fi
    
    # 安装其他依赖
    pip install \
        numpy \
        scipy \
        matplotlib \
        seaborn \
        pandas \
        scikit-learn \
        h5py \
        pyyaml \
        tensorboard \
        tqdm \
        psutil \
        GPUtil \
        opencv-python-headless \
        Pillow \
        jupyter \
        ipython
    
    log_success "Python依赖安装完成"
}

# 复制项目文件
copy_project_files() {
    log_info "请手动上传项目文件到: $PROJECT_DIR"
    log_info "需要上传的文件包括:"
    echo "  - pdebench_extended/"
    echo "  - configs/"
    echo "  - 数据文件 (.pt)"
    echo ""
    read -p "文件上传完成后按回车继续..."
}

# 创建启动脚本
create_run_scripts() {
    log_info "创建运行脚本..."
    
    # 训练脚本
    cat > run_training.sh << 'EOF'
#!/bin/bash

# 激活虚拟环境
source venv/bin/activate

# 设置环境变量
export PYTHONPATH=$PWD/pdebench_extended:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0  # 如果有多GPU，可以指定

# 运行训练
echo "开始压力场训练..."
python pdebench_extended/train_pressure_field.py \
    --config pdebench_extended/configs/pressure_field_training.yaml \
    --data_path "your_data_path.pt" \
    --output_dir "./outputs" \
    --experiment_name "pressure_field_$(date +%Y%m%d_%H%M%S)"

echo "训练完成！"
EOF

    # 测试脚本
    cat > run_test.sh << 'EOF'
#!/bin/bash

# 激活虚拟环境
source venv/bin/activate

# 设置环境变量
export PYTHONPATH=$PWD/pdebench_extended:$PYTHONPATH

# 运行测试
echo "运行数据适配器测试..."
python pdebench_extended/examples/pressure_field_example.py

echo "测试完成！"
EOF

    # TensorBoard启动脚本
    cat > start_tensorboard.sh << 'EOF'
#!/bin/bash

# 激活虚拟环境
source venv/bin/activate

# 启动TensorBoard
echo "启动TensorBoard..."
echo "访问地址: http://服务器IP:6006"
tensorboard --logdir=./outputs --host=0.0.0.0 --port=6006
EOF

    # Jupyter启动脚本
    cat > start_jupyter.sh << 'EOF'
#!/bin/bash

# 激活虚拟环境
source venv/bin/activate

# 启动Jupyter
echo "启动Jupyter Notebook..."
echo "访问地址: http://服务器IP:8888"
jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root
EOF

    # 设置执行权限
    chmod +x run_training.sh run_test.sh start_tensorboard.sh start_jupyter.sh
    
    log_success "运行脚本创建完成"
}

# 创建系统服务 (可选)
create_systemd_service() {
    read -p "是否创建systemd服务? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        return
    fi
    
    log_info "创建systemd服务..."
    
    sudo tee /etc/systemd/system/pressure-field-training.service > /dev/null << EOF
[Unit]
Description=Pressure Field Training Service
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$PROJECT_DIR
ExecStart=$PROJECT_DIR/run_training.sh
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

    sudo systemctl daemon-reload
    sudo systemctl enable pressure-field-training.service
    
    log_success "systemd服务创建完成"
    log_info "使用命令: sudo systemctl start pressure-field-training"
}

# 配置防火墙
setup_firewall() {
    log_info "配置防火墙..."
    
    # 开放TensorBoard端口
    sudo firewall-cmd --permanent --add-port=6006/tcp
    # 开放Jupyter端口
    sudo firewall-cmd --permanent --add-port=8888/tcp
    # 重载防火墙
    sudo firewall-cmd --reload
    
    log_success "防火墙配置完成"
}

# 创建监控脚本
create_monitor_script() {
    log_info "创建系统监控脚本..."
    
    cat > monitor.sh << 'EOF'
#!/bin/bash

# 系统监控脚本
echo "=== 系统资源监控 ==="
echo "时间: $(date)"
echo ""

echo "=== CPU使用率 ==="
top -bn1 | grep "Cpu(s)" | awk '{print $2}' | awk -F'%' '{print "CPU使用率: " $1 "%"}'

echo ""
echo "=== 内存使用情况 ==="
free -h

echo ""
echo "=== 磁盘使用情况 ==="
df -h

echo ""
echo "=== GPU使用情况 ==="
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits
else
    echo "未检测到GPU"
fi

echo ""
echo "=== 训练进程状态 ==="
ps aux | grep python | grep -v grep

echo ""
echo "=== 网络连接 ==="
netstat -tlnp | grep -E ':(6006|8888)'
EOF

    chmod +x monitor.sh
    
    log_success "监控脚本创建完成"
}

# 显示使用说明
show_usage() {
    log_success "=== 部署完成！==="
    echo ""
    echo "项目目录: $PROJECT_DIR"
    echo ""
    echo "=== 使用说明 ==="
    echo "1. 训练模型:"
    echo "   ./run_training.sh"
    echo ""
    echo "2. 运行测试:"
    echo "   ./run_test.sh"
    echo ""
    echo "3. 启动TensorBoard:"
    echo "   ./start_tensorboard.sh"
    echo "   访问: http://服务器IP:6006"
    echo ""
    echo "4. 启动Jupyter:"
    echo "   ./start_jupyter.sh"
    echo "   访问: http://服务器IP:8888"
    echo ""
    echo "5. 系统监控:"
    echo "   ./monitor.sh"
    echo ""
    echo "6. 后台运行 (使用screen):"
    echo "   screen -S training"
    echo "   ./run_training.sh"
    echo "   Ctrl+A+D 分离会话"
    echo "   screen -r training 重新连接"
    echo ""
    echo "=== 重要提醒 ==="
    echo "1. 请确保上传数据文件到项目目录"
    echo "2. 修改run_training.sh中的数据路径"
    echo "3. 根据需要调整配置文件"
    echo "4. 定期备份训练结果"
}

# 主函数
main() {
    echo "=== CentOS 压力场训练系统一键部署脚本 ==="
    echo ""
    
    check_root
    detect_os
    install_system_deps
    install_python
    install_cuda
    setup_project
    setup_venv
    install_python_deps
    copy_project_files
    create_run_scripts
    create_systemd_service
    setup_firewall
    create_monitor_script
    show_usage
    
    log_success "部署脚本执行完成！"
}

# 执行主函数
main "$@"