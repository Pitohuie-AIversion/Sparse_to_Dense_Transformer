@echo off
REM PDEBench多实验批处理脚本
REM 自动化数据集构建和模型训练过程

setlocal enabledelayedexpansion

echo ========================================
echo PDEBench多注意力机制实验批处理系统
echo ========================================
echo.

REM 检查Python环境
python --version >nul 2>&1
if errorlevel 1 (
    echo 错误: 未找到Python环境
    echo 请确保Python已安装并添加到PATH中
    pause
    exit /b 1
)

REM 检查必要的包
echo 检查Python依赖包...
python -c "import torch, numpy, h5py, yaml, matplotlib" >nul 2>&1
if errorlevel 1 (
    echo 警告: 某些依赖包可能未安装
    echo 正在尝试安装依赖包...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo 错误: 依赖包安装失败
        pause
        exit /b 1
    )
)

REM 创建必要的目录
echo 创建目录结构...
if not exist "data\pdebench" mkdir "data\pdebench"
if not exist "results" mkdir "results"
if not exist "logs" mkdir "logs"

REM 显示菜单
:menu
echo.
echo 请选择操作:
echo 1. 验证数据文件
echo 2. 构建单个数据集
echo 3. 构建所有数据集
echo 4. 运行单个实验
echo 5. 运行所有实验 (完整批处理)
echo 6. 仅显示实验矩阵
echo 7. 分析现有结果
echo 8. 退出
echo.
set /p choice="请输入选择 (1-8): "

if "%choice%"=="1" goto validate_data
if "%choice%"=="2" goto build_single
if "%choice%"=="3" goto build_all
if "%choice%"=="4" goto run_single
if "%choice%"=="5" goto run_all
if "%choice%"=="6" goto show_matrix
if "%choice%"=="7" goto analyze_results
if "%choice%"=="8" goto exit

echo 无效选择，请重新输入
goto menu

:validate_data
echo.
echo ========================================
echo 验证数据文件
echo ========================================
python build_training_dataset.py --validate-only
if errorlevel 1 (
    echo.
    echo 数据文件验证失败!
    echo 请检查以下事项:
    echo 1. 数据文件是否存在于正确路径
    echo 2. 配置文件中的路径是否正确
    echo 3. 文件权限是否正确
) else (
    echo.
    echo 数据文件验证成功!
)
pause
goto menu

:build_single
echo.
echo 可用的PDE类型:
echo 1. darcy_flow (2D Darcy Flow)
echo 2. ns_compressible (2D Compressible Navier-Stokes)
echo 3. shallow_water (2D Shallow Water)
echo.
set /p pde_choice="请选择PDE类型 (1-3): "

if "%pde_choice%"=="1" set pde_type=darcy_flow
if "%pde_choice%"=="2" set pde_type=ns_compressible
if "%pde_choice%"=="3" set pde_type=shallow_water

if not defined pde_type (
    echo 无效选择
    goto menu
)

echo.
echo ========================================
echo 构建 %pde_type% 数据集
echo ========================================
python build_training_dataset.py -p %pde_type%
if errorlevel 1 (
    echo 数据集构建失败!
) else (
    echo 数据集构建成功!
)
pause
goto menu

:build_all
echo.
echo ========================================
echo 构建所有数据集
echo ========================================
python build_training_dataset.py -p all
if errorlevel 1 (
    echo 某些数据集构建失败!
) else (
    echo 所有数据集构建成功!
)
pause
goto menu

:run_single
echo.
echo 可用的PDE类型:
echo 1. darcy_flow
echo 2. ns_compressible
echo 3. shallow_water
echo.
set /p pde_choice="请选择PDE类型 (1-3): "

if "%pde_choice%"=="1" set pde_type=darcy_flow
if "%pde_choice%"=="2" set pde_type=ns_compressible
if "%pde_choice%"=="3" set pde_type=shallow_water

if not defined pde_type (
    echo 无效选择
    goto menu
)

echo.
echo 可用的注意力机制:
echo 1. MultiHeadAttention
echo 2. ECAAttention
echo 3. SEAttention
echo 4. CBAM
echo 5. CoordinateAttention
echo.
set /p att_choice="请选择注意力机制 (1-5): "

if "%att_choice%"=="1" set attention_type=MultiHeadAttention
if "%att_choice%"=="2" set attention_type=ECAAttention
if "%att_choice%"=="3" set attention_type=SEAttention
if "%att_choice%"=="4" set attention_type=CBAM
if "%att_choice%"=="5" set attention_type=CoordinateAttention

if not defined attention_type (
    echo 无效选择
    goto menu
)

echo.
echo ========================================
echo 运行单个实验: %pde_type% + %attention_type%
echo ========================================

REM 设置环境变量
set CURRENT_PDE=%pde_type%
set ATTENTION_TYPE=%attention_type%

REM 运行训练
python main.py -c configs/multi_pde_training_config.yaml
if errorlevel 1 (
    echo 实验运行失败!
) else (
    echo 实验运行成功!
)
pause
goto menu

:run_all
echo.
echo ========================================
echo 运行所有实验 (完整批处理)
echo ========================================
echo.
echo 警告: 这将运行所有PDE类型和注意力机制的组合实验
echo 这可能需要很长时间 (数小时到数天)
echo.
set /p confirm="确认运行所有实验? (y/N): "
if /i not "%confirm%"=="y" goto menu

echo.
echo 开始运行所有实验...
python experiment_manager.py -c configs/multi_pde_training_config.yaml
if errorlevel 1 (
    echo 批处理实验失败!
) else (
    echo 批处理实验完成!
    echo 请查看results目录中的实验报告
)
pause
goto menu

:show_matrix
echo.
echo ========================================
echo 显示实验矩阵
echo ========================================
python experiment_manager.py -c configs/multi_pde_training_config.yaml --dry-run
pause
goto menu

:analyze_results
echo.
echo ========================================
echo 分析现有结果
echo ========================================
python experiment_manager.py -c configs/multi_pde_training_config.yaml --analyze-only
if errorlevel 1 (
    echo 结果分析失败! 可能没有现有结果文件
) else (
    echo 结果分析完成!
)
pause
goto menu

:exit
echo.
echo 感谢使用PDEBench多实验批处理系统!
echo 如有问题，请查看日志文件或联系开发者
pause
exit /b 0

:error
echo.
echo 发生错误，程序退出
pause
exit /b 1