#!/bin/bash

# VIVTransformer 可视化训练 Linux 一键部署脚本
# VIVTransformer Visualization Training Linux Deployment Script

set -e  # 遇到错误立即退出

echo "========================================"
echo "VIVTransformer 可视化训练 Linux 部署"
echo "VIVTransformer Visualization Training Linux Deployment"
echo "========================================"
echo ""

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 打印彩色信息
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查系统要求
check_system() {
    print_info "检查系统环境..."
    
    # 检查操作系统
    if [[ "$OSTYPE" != "linux-gnu"* ]]; then
        print_error "此脚本仅支持Linux系统"
        exit 1
    fi
    
    # 检查Python3
    if ! command -v python3 &> /dev/null; then
        print_error "未找到Python3，请先安装Python3"
        echo "Ubuntu/Debian: sudo apt install python3 python3-pip python3-venv"
        echo "CentOS/RHEL: sudo yum install python3 python3-pip"
        exit 1
    fi
    
    # 检查Python版本
    python_version=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    if [[ $(echo "$python_version < 3.7" | bc -l) -eq 1 ]]; then
        print_error "Python版本过低，需要3.7+，当前版本: $python_version"
        exit 1
    fi
    
    print_success "系统环境检查通过 (Python $python_version)"
}

# 创建虚拟环境
setup_venv() {
    print_info "设置Python虚拟环境..."
    
    if [ -d "viv_env" ]; then
        print_warning "虚拟环境已存在，跳过创建"
    else
        python3 -m venv viv_env
        print_success "虚拟环境创建完成"
    fi
    
    # 激活虚拟环境
    source viv_env/bin/activate
    print_success "虚拟环境已激活"
    
    # 升级pip
    pip install --upgrade pip > /dev/null 2>&1
    print_success "pip已升级到最新版本"
}

# 安装依赖包
install_dependencies() {
    print_info "安装Python依赖包..."
    
    # 基础依赖
    dependencies=(
        "torch"
        "torchvision"
        "matplotlib"
        "numpy"
        "seaborn"
        "pandas"
        "psutil"
        "tqdm"
    )
    
    for dep in "${dependencies[@]}"; do
        print_info "安装 $dep..."
        pip install "$dep" > /dev/null 2>&1
        print_success "$dep 安装完成"
    done
    
    print_success "所有依赖包安装完成"
}

# 配置环境变量
setup_environment() {
    print_info "配置环境变量..."
    
    # 设置matplotlib后端为非交互式
    export MPLBACKEND=Agg
    
    # 避免OpenMP冲突
    export KMP_DUPLICATE_LIB_OK=TRUE
    export OMP_NUM_THREADS=1
    export MKL_NUM_THREADS=1
    export TORCH_NUM_THREADS=1
    
    # 将环境变量写入激活脚本
    cat >> viv_env/bin/activate << EOF

# VIVTransformer 环境变量
export MPLBACKEND=Agg
export KMP_DUPLICATE_LIB_OK=TRUE
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TORCH_NUM_THREADS=1
EOF
    
    print_success "环境变量配置完成"
}

# 设置脚本权限
setup_permissions() {
    print_info "设置脚本执行权限..."
    
    chmod +x *.sh 2>/dev/null || true
    
    print_success "脚本权限设置完成"
}

# 验证安装
verify_installation() {
    print_info "验证安装..."
    
    # 测试Python包导入
    python3 -c "
import torch
import matplotlib
import numpy
import seaborn
import pandas
import psutil
print('所有依赖包导入成功')
" 2>/dev/null
    
    if [ $? -eq 0 ]; then
        print_success "依赖包验证通过"
    else
        print_error "依赖包验证失败"
        exit 1
    fi
    
    # 检查CUDA可用性（可选）
    cuda_available=$(python3 -c "import torch; print(torch.cuda.is_available())" 2>/dev/null)
    if [ "$cuda_available" = "True" ]; then
        print_success "CUDA可用，支持GPU加速"
    else
        print_warning "CUDA不可用，将使用CPU训练"
    fi
}

# 创建快速启动脚本
create_launcher() {
    print_info "创建快速启动脚本..."
    
    cat > run_training.sh << 'EOF'
#!/bin/bash

# 快速启动脚本
echo "激活虚拟环境..."
source viv_env/bin/activate

echo "选择运行模式:"
echo "1) 增强版可视化训练 (完整功能)"
echo "2) 独立预测可视化演示"
echo "3) 自定义参数训练"
read -p "请选择 (1-3): " choice

case $choice in
    1)
        echo "启动增强版可视化训练..."
        ./启动增强版可视化训练.sh
        ;;
    2)
        echo "启动预测可视化演示..."
        ./启动预测可视化演示.sh
        ;;
    3)
        read -p "输入训练轮数 (默认20): " epochs
        epochs=${epochs:-20}
        read -p "输入输出目录 (默认custom_output): " output_dir
        output_dir=${output_dir:-custom_output}
        echo "启动自定义训练..."
        python3 enhanced_visual_training.py --epochs $epochs --output-dir $output_dir
        ;;
    *)
        echo "无效选择"
        exit 1
        ;;
esac
EOF
    
    chmod +x run_training.sh
    print_success "快速启动脚本创建完成: run_training.sh"
}

# 显示使用说明
show_usage() {
    echo ""
    echo "========================================"
    echo "部署完成！使用说明:"
    echo "========================================"
    echo ""
    echo "🚀 快速启动:"
    echo "   ./run_training.sh"
    echo ""
    echo "📊 直接运行训练:"
    echo "   source viv_env/bin/activate"
    echo "   ./启动增强版可视化训练.sh"
    echo ""
    echo "🔍 独立预测可视化:"
    echo "   source viv_env/bin/activate"
    echo "   ./启动预测可视化演示.sh"
    echo ""
    echo "⚙️ 自定义参数:"
    echo "   source viv_env/bin/activate"
    echo "   python3 enhanced_visual_training.py --epochs 50 --output-dir my_results"
    echo ""
    echo "📁 输出文件位置:"
    echo "   - 增强版训练: enhanced_training_output/"
    echo "   - 预测可视化: demo_prediction_visualizations/"
    echo ""
    echo "📖 详细文档: LINUX_SERVER_GUIDE.md"
    echo ""
}

# 主函数
main() {
    echo "开始部署..."
    echo ""
    
    check_system
    setup_venv
    install_dependencies
    setup_environment
    setup_permissions
    verify_installation
    create_launcher
    
    print_success "部署完成！"
    show_usage
}

# 错误处理
trap 'print_error "部署过程中发生错误，请检查上述输出信息"' ERR

# 运行主函数
main