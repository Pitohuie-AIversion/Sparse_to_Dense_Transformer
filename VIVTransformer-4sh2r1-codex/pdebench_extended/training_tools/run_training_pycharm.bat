@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

REM =============================================================================
REM PyCharm环境训练脚本 - 压力场重建训练
REM 适用于PyCharm IDE环境的训练启动脚本
REM =============================================================================

echo ╔══════════════════════════════════════════════════════════════════════════════╗
echo ║                    PyCharm环境 - 压力场重建训练                             ║
echo ║                     VIV Transformer PyCharm版本                            ║
echo ╚══════════════════════════════════════════════════════════════════════════════╝
echo.

REM 设置颜色
for /f %%A in ('echo prompt $E ^| cmd') do set "ESC=%%A"
set "RED=%ESC%[31m"
set "GREEN=%ESC%[32m"
set "YELLOW=%ESC%[33m"
set "BLUE=%ESC%[34m"
set "PURPLE=%ESC%[35m"
set "CYAN=%ESC%[36m"
set "NC=%ESC%[0m"

REM 配置参数
set "DATA_PATH=data\pressure_field_data.pt"
set "CONFIG_FILE=configs\pressure_field_training.yaml"
set "OUTPUT_DIR=outputs"
set "EXPERIMENT_NAME=pressure_field_training_%date:~0,4%%date:~5,2%%date:~8,2%_%time:~0,2%%time:~3,2%%time:~6,2%"
set "EXPERIMENT_NAME=%EXPERIMENT_NAME: =0%"

echo %CYAN%=== PyCharm环境检查 ===%NC%
echo.

REM 检查Python
python --version >nul 2>&1
if %errorlevel% == 0 (
    for /f "tokens=*" %%i in ('python --version 2^>^&1') do set "PYTHON_VERSION=%%i"
    echo %GREEN%✓%NC% Python: !PYTHON_VERSION!
) else (
    echo %RED%❌ Python未安装或不在PATH中%NC%
    echo 请在PyCharm中配置Python解释器
    pause
    exit /b 1
)

REM 检查PyTorch
python -c "import torch" >nul 2>&1
if %errorlevel% == 0 (
    for /f "tokens=*" %%i in ('python -c "import torch; print(torch.__version__)" 2^>^&1') do set "TORCH_VERSION=%%i"
    for /f "tokens=*" %%i in ('python -c "import torch; print(torch.cuda.is_available())" 2^>^&1') do set "CUDA_AVAILABLE=%%i"
    echo %GREEN%✓%NC% PyTorch: !TORCH_VERSION! (CUDA: !CUDA_AVAILABLE!)
) else (
    echo %RED%❌ PyTorch未安装%NC%
    echo 请在PyCharm终端中安装: pip install torch torchvision torchaudio
    pause
    exit /b 1
)

REM 检查其他依赖
echo %BLUE%检查其他依赖包...%NC%
python -c "import numpy, scipy, matplotlib, yaml" >nul 2>&1
if %errorlevel% == 0 (
    echo %GREEN%✓%NC% 基础依赖包已安装
) else (
    echo %YELLOW%⚠️  部分依赖包未安装%NC%
    echo 请在PyCharm终端中运行: pip install -r requirements.txt
)

REM 检查NVIDIA GPU
nvidia-smi >nul 2>&1
if %errorlevel% == 0 (
    echo %GREEN%✓%NC% NVIDIA GPU驱动已安装
    echo.
    echo %BLUE%=== GPU状态 ===%NC%
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader,nounits
) else (
    echo %YELLOW%⚠️  NVIDIA驱动未安装或GPU不可用%NC%
    echo 将使用CPU训练（速度较慢）
)

echo.
echo %PURPLE%=== 数据集检查 ===%NC%

REM 检查数据文件
set "DATA_FOUND=0"
if exist "%DATA_PATH%" (
    for %%A in ("%DATA_PATH%") do set "DATA_SIZE=%%~zA"
    set /a "DATA_SIZE_MB=!DATA_SIZE! / 1024 / 1024"
    echo %GREEN%✓%NC% 数据文件: !DATA_SIZE_MB!MB
    set "DATA_FOUND=1"
) else (
    REM 搜索其他可能的数据文件
    echo 搜索现有数据文件...
    for %%f in (*.pt *.hdf5 *.h5) do (
        if exist "%%f" (
            set "DATA_PATH=%%f"
            for %%A in ("%%f") do set "DATA_SIZE=%%~zA"
            set /a "DATA_SIZE_MB=!DATA_SIZE! / 1024 / 1024"
            echo %GREEN%✓%NC% 找到数据文件: %%f (!DATA_SIZE_MB!MB)
            set "DATA_FOUND=1"
            goto :data_found
        )
    )
    
    REM 在data目录中搜索
    if exist "data" (
        for /r "data" %%f in (*.pt *.hdf5 *.h5) do (
            if exist "%%f" (
                set "DATA_PATH=%%f"
                for %%A in ("%%f") do set "DATA_SIZE=%%~zA"
                set /a "DATA_SIZE_MB=!DATA_SIZE! / 1024 / 1024"
                echo %GREEN%✓%NC% 找到数据文件: %%f (!DATA_SIZE_MB!MB)
                set "DATA_FOUND=1"
                goto :data_found
            )
        )
    )
    
    :data_found
    if "!DATA_FOUND!"=="0" (
        echo %YELLOW%⚠️  未找到数据文件%NC%
        echo.
        echo 数据获取选项:
        echo 1. 运行 quick_start_windows.bat 自动下载数据
        echo 2. 手动下载数据到 data\ 目录
        echo 3. 生成示例数据用于测试
        echo.
        set /p "DATA_CHOICE=请选择 [1-3]: "
        
        if "!DATA_CHOICE!"=="1" (
            echo %BLUE%启动数据下载脚本...%NC%
            call quick_start_windows.bat
            goto :check_data_again
        )
        if "!DATA_CHOICE!"=="2" (
            echo %YELLOW%请手动下载数据文件到 data\ 目录%NC%
            echo 数据源:
            echo - GitHub: https://github.com/pdebench/PDEBench
            echo - 直接下载: https://darus.uni-stuttgart.de/dataset.xhtml?persistentId=doi:10.18419/darus-2986
            pause
            exit /b 1
        )
        if "!DATA_CHOICE!"=="3" (
            echo %BLUE%生成示例数据...%NC%
            python -c "
import torch
import numpy as np
import os

# 创建示例压力场数据
print('生成示例压力场数据...')
data = {
    'pressure_field': torch.randn(100, 64, 64),  # 100个样本，64x64网格
    'coordinates': torch.meshgrid(torch.linspace(0, 1, 64), torch.linspace(0, 1, 64)),
    'reynolds_number': torch.rand(100) * 1000 + 100,
    'time_steps': torch.arange(100)
}

os.makedirs('data', exist_ok=True)
torch.save(data, 'data/sample_pressure_data.pt')
print('示例数据已生成: data/sample_pressure_data.pt')
"
            set "DATA_PATH=data\sample_pressure_data.pt"
            set "DATA_FOUND=1"
        )
        
        :check_data_again
        REM 重新检查数据文件
        for %%f in (*.pt *.hdf5 *.h5) do (
            if exist "%%f" (
                set "DATA_PATH=%%f"
                set "DATA_FOUND=1"
                goto :data_check_done
            )
        )
        for /r "data" %%f in (*.pt *.hdf5 *.h5) do (
            if exist "%%f" (
                set "DATA_PATH=%%f"
                set "DATA_FOUND=1"
                goto :data_check_done
            )
        )
        :data_check_done
    )
)

if "!DATA_FOUND!"=="0" (
    echo %RED%❌ 未找到数据文件%NC%
    echo 请先获取数据集后再运行训练
    pause
    exit /b 1
)

REM 检查配置文件
if not exist "%CONFIG_FILE%" (
    echo %RED%❌ 配置文件不存在: %CONFIG_FILE%%NC%
    echo 请检查配置文件路径
    pause
    exit /b 1
)

REM 创建输出目录
if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"

echo.
echo %BLUE%=== 训练配置 ===%NC%
echo 数据文件: %DATA_PATH%
echo 配置文件: %CONFIG_FILE%
echo 输出目录: %OUTPUT_DIR%\%EXPERIMENT_NAME%
echo 实验名称: %EXPERIMENT_NAME%
echo.

REM 询问运行方式
echo %CYAN%=== 运行选项 ===%NC%
echo 1. 直接开始训练
echo 2. 启动TensorBoard监控
echo 3. 运行测试模式
echo 4. 查看配置信息
echo.
set /p "RUN_CHOICE=请选择运行方式 [1-4]: "

if "%RUN_CHOICE%"=="1" goto :start_training
if "%RUN_CHOICE%"=="2" goto :start_tensorboard
if "%RUN_CHOICE%"=="3" goto :test_mode
if "%RUN_CHOICE%"=="4" goto :show_config

:start_training
echo.
echo %GREEN%=== 开始训练 ===%NC%
echo 启动压力场重建训练...
echo.

REM 设置环境变量
set PYTHONPATH=%CD%;%PYTHONPATH%

REM 启动训练
python train_pressure_field.py --config "%CONFIG_FILE%" --data "%DATA_PATH%" --output "%OUTPUT_DIR%\%EXPERIMENT_NAME%" --name "%EXPERIMENT_NAME%"

if %errorlevel% == 0 (
    echo.
    echo %GREEN%✓ 训练完成%NC%
    echo 结果保存在: %OUTPUT_DIR%\%EXPERIMENT_NAME%
) else (
    echo.
    echo %RED%❌ 训练过程中出现错误%NC%
)
goto :end

:start_tensorboard
echo.
echo %BLUE%=== 启动TensorBoard ===%NC%
echo 启动TensorBoard监控界面...
echo.

start "TensorBoard" cmd /k "tensorboard --logdir=%OUTPUT_DIR% --port=6006 && echo TensorBoard已启动，访问 http://localhost:6006"
echo TensorBoard已在新窗口中启动
echo 访问地址: http://localhost:6006
echo.
goto :start_training

:test_mode
echo.
echo %YELLOW%=== 测试模式 ===%NC%
echo 运行快速测试...
echo.

set PYTHONPATH=%CD%;%PYTHONPATH%
python -c "
import torch
print(f'PyTorch版本: {torch.__version__}')
print(f'CUDA可用: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU数量: {torch.cuda.device_count()}')
    print(f'当前GPU: {torch.cuda.get_device_name()}')
print('环境测试完成')
"

echo.
echo 测试完成，按任意键继续训练...
pause >nul
goto :start_training

:show_config
echo.
echo %PURPLE%=== 配置信息 ===%NC%
echo.
echo 项目路径: %CD%
echo Python路径: 
where python
echo.
echo 配置文件内容:
type "%CONFIG_FILE%"
echo.
echo 按任意键返回主菜单...
pause >nul
goto :start_training

:end
echo.
echo 按任意键退出...
pause >nul