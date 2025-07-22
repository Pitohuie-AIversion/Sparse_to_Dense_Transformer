@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

REM =============================================================================
REM Windows快速启动脚本 - 压力场重建训练
REM 适用于Windows环境的一键训练启动
REM =============================================================================

echo ╔══════════════════════════════════════════════════════════════════════════════╗
echo ║                        压力场重建训练 - 快速启动                            ║
echo ║                     VIV Transformer Windows版本                            ║
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

echo %CYAN%=== 环境检查 ===%NC%
echo.

REM 检查Python
python --version >nul 2>&1
if %errorlevel% == 0 (
    for /f "tokens=*" %%i in ('python --version 2^>^&1') do set "PYTHON_VERSION=%%i"
    echo %GREEN%✓%NC% Python: !PYTHON_VERSION!
) else (
    echo %RED%❌ Python未安装或不在PATH中%NC%
    echo 请安装Python 3.8+
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
    echo 请安装PyTorch: pip install torch torchvision torchaudio
    pause
    exit /b 1
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
echo %PURPLE%=== 文件检查 ===%NC%

REM 检查数据文件
if exist "%DATA_PATH%" (
    for %%A in ("%DATA_PATH%") do set "DATA_SIZE=%%~zA"
    set /a "DATA_SIZE_MB=!DATA_SIZE! / 1024 / 1024"
    echo %GREEN%✓%NC% 数据文件: !DATA_SIZE_MB!MB
) else (
    echo %RED%❌ 数据文件未找到: %DATA_PATH%%NC%
    echo 请确保数据文件存在
    pause
    exit /b 1
)

REM 检查配置文件
if exist "%CONFIG_FILE%" (
    echo %GREEN%✓%NC% 配置文件: 存在
) else (
    echo %RED%❌ 配置文件未找到: %CONFIG_FILE%%NC%
    echo 请确保配置文件存在
    pause
    exit /b 1
)

REM 创建输出目录
if not exist "%OUTPUT_DIR%" (
    mkdir "%OUTPUT_DIR%"
    echo %GREEN%✓%NC% 创建输出目录: %OUTPUT_DIR%
) else (
    echo %GREEN%✓%NC% 输出目录: 存在
)

if not exist "%OUTPUT_DIR%\checkpoints" mkdir "%OUTPUT_DIR%\checkpoints"
if not exist "%OUTPUT_DIR%\logs" mkdir "%OUTPUT_DIR%\logs"
if not exist "%OUTPUT_DIR%\visualizations" mkdir "%OUTPUT_DIR%\visualizations"

echo.
echo %CYAN%=== GPU选择 ===%NC%

REM GPU选择逻辑
set "CUDA_DEVICE=auto"
nvidia-smi >nul 2>&1
if %errorlevel% == 0 (
    echo 检测到NVIDIA GPU，正在分析使用情况...
    
    REM 简单的GPU选择（选择第一个可用GPU）
    for /f "skip=1 tokens=1,2,3,4,5,6 delims=," %%a in ('nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,nounits 2^>nul') do (
        set "GPU_INDEX=%%a"
        set "GPU_NAME=%%b"
        set "MEM_USED=%%c"
        set "MEM_TOTAL=%%d"
        set "GPU_UTIL=%%e"
        set "GPU_TEMP=%%f"
        
        set /a "MEM_USAGE_PERCENT=!MEM_USED! * 100 / !MEM_TOTAL!"
        
        if !MEM_USAGE_PERCENT! LSS 10 (
            set "CUDA_DEVICE=!GPU_INDEX!"
            echo %GREEN%选择GPU !GPU_INDEX!: !GPU_NAME! (显存使用: !MEM_USAGE_PERCENT!%%)%NC%
            goto :gpu_selected
        )
    )
    
    :gpu_selected
    if "!CUDA_DEVICE!"=="auto" (
        set "CUDA_DEVICE=0"
        echo %YELLOW%使用默认GPU 0%NC%
    )
) else (
    set "CUDA_DEVICE=cpu"
    echo %YELLOW%使用CPU训练%NC%
)

echo.
echo %BLUE%=== 训练配置 ===%NC%
echo 实验名称: %EXPERIMENT_NAME%
echo 数据路径: %DATA_PATH%
echo 配置文件: %CONFIG_FILE%
echo 输出目录: %OUTPUT_DIR%
echo 使用设备: %CUDA_DEVICE%
echo.

REM 设置环境变量
if not "%CUDA_DEVICE%"=="cpu" (
    set "CUDA_VISIBLE_DEVICES=%CUDA_DEVICE%"
    echo 设置CUDA_VISIBLE_DEVICES=%CUDA_DEVICE%
)

set "OMP_NUM_THREADS=4"
set "MKL_NUM_THREADS=4"
echo 设置OMP_NUM_THREADS=4
echo 设置MKL_NUM_THREADS=4
echo.

REM 训练模式选择
echo %PURPLE%=== 训练模式选择 ===%NC%
echo.
echo 1. 前台训练 (可以看到实时输出)
echo 2. 后台训练 (在后台运行，生成日志文件)
echo 3. 启动TensorBoard监控
echo 4. 查看GPU状态
echo 0. 退出
echo.
set /p "MODE=请选择模式 [0-4]: "

if "%MODE%"=="1" goto :foreground_training
if "%MODE%"=="2" goto :background_training
if "%MODE%"=="3" goto :start_tensorboard
if "%MODE%"=="4" goto :show_gpu_status
if "%MODE%"=="0" goto :exit

echo %RED%无效选择%NC%
pause
goto :exit

:foreground_training
echo.
echo %GREEN%=== 启动前台训练 ===%NC%
echo 按Ctrl+C可以停止训练
echo.

REM 构建训练命令
set "TRAIN_CMD=python train_pressure_field.py"
set "TRAIN_CMD=%TRAIN_CMD% --config %CONFIG_FILE%"
set "TRAIN_CMD=%TRAIN_CMD% --data_path %DATA_PATH%"
set "TRAIN_CMD=%TRAIN_CMD% --output_dir %OUTPUT_DIR%"
set "TRAIN_CMD=%TRAIN_CMD% --experiment_name %EXPERIMENT_NAME%"

if not "%CUDA_DEVICE%"=="cpu" (
    set "TRAIN_CMD=%TRAIN_CMD% --device cuda"
) else (
    set "TRAIN_CMD=%TRAIN_CMD% --device cpu"
)

echo 执行命令: %TRAIN_CMD%
echo.

%TRAIN_CMD%

echo.
echo %GREEN%训练完成%NC%
pause
goto :exit

:background_training
echo.
echo %GREEN%=== 启动后台训练 ===%NC%

set "LOG_FILE=%OUTPUT_DIR%\logs\training_%EXPERIMENT_NAME%.log"
set "TRAIN_CMD=python train_pressure_field.py"
set "TRAIN_CMD=%TRAIN_CMD% --config %CONFIG_FILE%"
set "TRAIN_CMD=%TRAIN_CMD% --data_path %DATA_PATH%"
set "TRAIN_CMD=%TRAIN_CMD% --output_dir %OUTPUT_DIR%"
set "TRAIN_CMD=%TRAIN_CMD% --experiment_name %EXPERIMENT_NAME%"

if not "%CUDA_DEVICE%"=="cpu" (
    set "TRAIN_CMD=%TRAIN_CMD% --device cuda"
) else (
    set "TRAIN_CMD=%TRAIN_CMD% --device cpu"
)

echo 训练将在后台运行
echo 日志文件: %LOG_FILE%
echo.
echo 执行命令: %TRAIN_CMD%
echo.

start /b cmd /c "%TRAIN_CMD% > %LOG_FILE% 2>&1"

echo %GREEN%后台训练已启动%NC%
echo 可以使用以下命令查看日志:
echo type "%LOG_FILE%"
echo.
echo 或者使用任务管理器查看python进程
pause
goto :exit

:start_tensorboard
echo.
echo %GREEN%=== 启动TensorBoard ===%NC%

REM 检查TensorBoard
python -c "import tensorboard" >nul 2>&1
if %errorlevel% == 0 (
    echo 启动TensorBoard监控...
    echo 访问地址: http://localhost:6006
    echo 按Ctrl+C停止TensorBoard
    echo.
    
    start /b cmd /c "tensorboard --logdir=%OUTPUT_DIR% --port=6006 --host=0.0.0.0"
    
    timeout /t 3 >nul
    echo %GREEN%TensorBoard已启动%NC%
    echo 请在浏览器中访问: http://localhost:6006
) else (
    echo %RED%TensorBoard未安装%NC%
    echo 请安装: pip install tensorboard
)

pause
goto :exit

:show_gpu_status
echo.
echo %GREEN%=== GPU状态 ===%NC%

nvidia-smi >nul 2>&1
if %errorlevel% == 0 (
    nvidia-smi
) else (
    echo %RED%NVIDIA GPU不可用%NC%
)

echo.
pause
goto :exit

:exit
echo.
echo %GREEN%感谢使用VIV Transformer训练系统!%NC%
pause
exit /b 0