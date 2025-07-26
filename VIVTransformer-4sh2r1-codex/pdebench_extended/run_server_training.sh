#!/bin/bash

# 服务器端训练运行脚本
# 用于在Linux服务器上运行VIVTransformer训练

echo "=== VIVTransformer 服务器端训练脚本 ==="
echo "当前目录: $(pwd)"
echo "Python版本: $(python --version)"
echo ""

# 检查必要文件
echo "=== 检查必要文件 ==="
files_to_check=(
    "train_configurable_multiscale.py"
    "multiscale/data/multiscale_adapter.py"
    "mymodels/transformer.py"
    "fix_server_import.py"
)

for file in "${files_to_check[@]}"; do
    if [ -f "$file" ]; then
        echo "✓ $file"
    else
        echo "✗ $file (缺失)"
    fi
done
echo ""

# 运行导入修复脚本
echo "=== 运行导入修复脚本 ==="
python fix_server_import.py
echo ""

# 设置环境变量
export PYTHONPATH="$(pwd):$(pwd)/multiscale:$(pwd)/multiscale/data:$PYTHONPATH"
echo "设置PYTHONPATH: $PYTHONPATH"
echo ""

# 提供训练选项
echo "=== 训练选项 ==="
echo "请选择要运行的训练配置:"
echo "1. 快速测试 (3个epoch, 小批次)"
echo "2. 标准训练 (20个epoch, 中等批次)"
echo "3. 自定义分辨率训练 (9x9 -> 128x128)"
echo "4. 长时间训练 (100个epoch)"
echo "5. 自定义参数"
echo ""

read -p "请输入选择 (1-5): " choice

case $choice in
    1)
        echo "运行快速测试..."
        python train_configurable_multiscale.py --scale_factor 4 --num_epochs 3 --batch_size 2
        ;;
    2)
        echo "运行标准训练..."
        python train_configurable_multiscale.py --scale_factor 4 --num_epochs 20 --batch_size 4
        ;;
    3)
        echo "运行自定义分辨率训练..."
        python train_configurable_multiscale.py --scale_factor 4 --input_resolution 9 9 --output_resolution 128 128 --num_epochs 20 --batch_size 4
        ;;
    4)
        echo "运行长时间训练..."
        python train_configurable_multiscale.py --scale_factor 4 --num_epochs 100 --batch_size 4
        ;;
    5)
        echo "请输入自定义参数:"
        read -p "scale_factor (默认4): " scale_factor
        read -p "num_epochs (默认20): " num_epochs
        read -p "batch_size (默认4): " batch_size
        read -p "learning_rate (默认1e-4): " learning_rate
        
        # 设置默认值
        scale_factor=${scale_factor:-4}
        num_epochs=${num_epochs:-20}
        batch_size=${batch_size:-4}
        learning_rate=${learning_rate:-1e-4}
        
        echo "运行自定义训练..."
        python train_configurable_multiscale.py --scale_factor $scale_factor --num_epochs $num_epochs --batch_size $batch_size --learning_rate $learning_rate
        ;;
    *)
        echo "无效选择，运行默认配置..."
        python train_configurable_multiscale.py --scale_factor 4 --num_epochs 20 --batch_size 4
        ;;
esac

echo ""
echo "=== 训练完成 ==="
echo "检查logs/目录查看训练日志和TensorBoard文件"
echo "使用 tensorboard --logdir=logs 启动TensorBoard查看训练进度"