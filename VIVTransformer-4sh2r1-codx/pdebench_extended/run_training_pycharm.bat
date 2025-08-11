@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

REM =============================================================================
REM PyCharm Environment Training Script - Pressure Field Reconstruction Training
REM Optimized for PyCharm IDE environment, providing one-click training startup functionality
REM =============================================================================

echo ╔══════════════════════════════════════════════════════════════════════════════╗
echo ║                    PyCharm Environment - Pressure Field Reconstruction Training                             ║
echo ║                     VIV Transformer PyCharm Version                            ║
echo ╚══════════════════════════════════════════════════════════════════════════════╝
echo.

REM Set colors
for /f %%A in ('echo prompt $E ^| cmd') do set "ESC=%%A"
set "RED=%ESC%[31m"
set "GREEN=%ESC%[32m"
set "YELLOW=%ESC%[33m"
set "BLUE=%ESC%[34m"
set "PURPLE=%ESC%[35m"
set "CYAN=%ESC%[36m"
set "NC=%ESC%[0m"

REM Configuration parameters
set "DATA_PATH=data\pressure_field_data.pt"
set "CONFIG_FILE=configs\pressure_field_training.yaml"
set "OUTPUT_DIR=outputs"
set "EXPERIMENT_NAME=pressure_field_training_%date:~0,4%%date:~5,2%%date:~8,2%_%time:~0,2%%time:~3,2%%time:~6,2%"
set "EXPERIMENT_NAME=%EXPERIMENT_NAME: =0%"

echo %CYAN%=== PyCharm Environment Check ===%NC%
echo.

REM 检查Python
python --version >nul 2>&1
if %errorlevel% == 0 (
    for /f "tokens=*" %%i in ('python --version 2^>^&1') do set "PYTHON_VERSION=%%i"
    echo %GREEN%✓%NC% Python: !PYTHON_VERSION!
) else (
    echo %RED%❌ Python is not installed or not in PATH%NC%
    echo Please configure a Python interpreter in PyCharm
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
    echo %RED%❌ PyTorch is not installed%NC%
    echo Please install in PyCharm terminal: pip install torch torchvision torchaudio
    pause
    exit /b 1
)

REM 检查其他依赖
echo %BLUE%Checking other dependencies...%NC%
python -c "import numpy, scipy, matplotlib, yaml" >nul 2>&1
if %errorlevel% == 0 (
    echo %GREEN%✓%NC% Base dependencies are installed
) else (
    echo %YELLOW%⚠️  Some dependencies are missing%NC%
    echo Please run in PyCharm terminal: pip install -r requirements.txt
)

REM Check NVIDIA GPU
nvidia-smi >nul 2>&1
if %errorlevel% == 0 (
    echo %GREEN%✓%NC% NVIDIA GPU driver is installed
    echo.
    echo %BLUE%=== GPU Status ===%NC%
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader,nounits
) else (
    echo %YELLOW%⚠️  NVIDIA driver not installed or GPU unavailable%NC%
    echo Will use CPU for training (slower)
)

echo.
echo %PURPLE%=== Dataset Check ===%NC%

REM 检查数据文件
set "DATA_FOUND=0"
if exist "%DATA_PATH%" (
    for %%A in ("%DATA_PATH%") do set "DATA_SIZE=%%~zA"
    set /a "DATA_SIZE_MB=!DATA_SIZE! / 1024 / 1024"
    echo %GREEN%✓%NC% Data file: !DATA_SIZE_MB!MB
    set "DATA_FOUND=1"
) else (
    REM Search for other possible data files
    echo Searching for existing data files...
    for %%f in (*.pt *.hdf5 *.h5) do (
        if exist "%%f" (
            set "DATA_PATH=%%f"
            for %%A in ("%%f") do set "DATA_SIZE=%%~zA"
            set /a "DATA_SIZE_MB=!DATA_SIZE! / 1024 / 1024"
            echo %GREEN%✓%NC% Found data file: %%f (!DATA_SIZE_MB!MB)
            set "DATA_FOUND=1"
            goto :data_found
        )
    )
    
    REM Search in data directory
    if exist "data" (
        for /r "data" %%f in (*.pt *.hdf5 *.h5) do (
            if exist "%%f" (
                set "DATA_PATH=%%f"
                for %%A in ("%%f") do set "DATA_SIZE=%%~zA"
                set /a "DATA_SIZE_MB=!DATA_SIZE! / 1024 / 1024"
                echo %GREEN%✓%NC% Found data file: %%f (!DATA_SIZE_MB!MB)
                set "DATA_FOUND=1"
                goto :data_found
            )
        )
    )
    
    :data_found
    if "!DATA_FOUND!"=="0" (
        echo %YELLOW%⚠️  No data file found%NC%
        echo.
        echo Data acquisition options:
        echo 1. Run quick_start_windows.bat to automatically download data
        echo 2. Manually download data to the data\ directory
        echo 3. Generate sample data for testing
        echo.
        set /p "DATA_CHOICE=Please choose [1-3]: "
        
        if "!DATA_CHOICE!"=="1" (
            echo %BLUE%Starting data download script...%NC%
            call quick_start_windows.bat
            goto :check_data_again
        )
        if "!DATA_CHOICE!"=="2" (
            echo %YELLOW%Please manually download the data file to the data\ directory%NC%
            echo Data sources:
            echo - GitHub: https://github.com/pdebench/PDEBench
            echo - Direct download: https://darus.uni-stuttgart.de/dataset.xhtml?persistentId=doi:10.18419/darus-2986
            pause
            exit /b 1
        )
        if "!DATA_CHOICE!"=="3" (
            echo %BLUE%Generating sample data...%NC%
            python -c "
import torch
import numpy as np
import os

# Create sample pressure field data
print('Generating sample pressure field data...')
data = {
    'pressure_field': torch.randn(100, 64, 64),  # 100 samples, 64x64 grid
    'coordinates': torch.meshgrid(torch.linspace(0, 1, 64), torch.linspace(0, 1, 64)),
    'reynolds_number': torch.rand(100) * 1000 + 100,
    'time_steps': torch.arange(100)
}

os.makedirs('data', exist_ok=True)
torch.save(data, 'data/sample_pressure_data.pt')
print('Sample data generated: data/sample_pressure_data.pt')
"
            set "DATA_PATH=data\sample_pressure_data.pt"
            set "DATA_FOUND=1"
        )
        
        :check_data_again
        REM Re-check data file
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
    echo %RED%❌ Data file not found%NC%
    echo Please obtain the dataset before running training
    pause
    exit /b 1
)

REM 检查配置文件
if not exist "%CONFIG_FILE%" (
    echo %RED%❌ Configuration file does not exist: %CONFIG_FILE%%NC%
    echo Please check the configuration file path
    pause
    exit /b 1
)

REM 创建输出目录
if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"

echo.
echo %BLUE%=== Training Configuration ===%NC%
echo Data file: %DATA_PATH%
echo Config file: %CONFIG_FILE%
echo Output directory: %OUTPUT_DIR%\%EXPERIMENT_NAME%
echo Experiment name: %EXPERIMENT_NAME%
echo.

REM 询问运行方式
echo %CYAN%=== Run Options ===%NC%
echo 1. Start training directly
echo 2. Launch TensorBoard monitoring
echo 3. Run test mode
echo 4. View configuration info
echo.
set /p "RUN_CHOICE=Please choose a run mode [1-4]: "

if "%RUN_CHOICE%"=="1" goto :start_training
if "%RUN_CHOICE%"=="2" goto :start_tensorboard
if "%RUN_CHOICE%"=="3" goto :test_mode
if "%RUN_CHOICE%"=="4" goto :show_config

:start_training
echo.
echo %GREEN%=== Start Training ===%NC%
echo Starting pressure field reconstruction training...
echo.

REM 设置环境变量
set PYTHONPATH=%CD%;%PYTHONPATH%

REM 启动训练
python train_pressure_field.py --config "%CONFIG_FILE%" --data "%DATA_PATH%" --output "%OUTPUT_DIR%\%EXPERIMENT_NAME%" --name "%EXPERIMENT_NAME%"

if %errorlevel% == 0 (
    echo.
    echo %GREEN%✓ Training complete%NC%
    echo Results saved to: %OUTPUT_DIR%\%EXPERIMENT_NAME%
) else (
    echo.
    echo %RED%❌ An error occurred during training%NC%
)
goto :end

:start_tensorboard
echo.
echo %BLUE%=== Launch TensorBoard ===%NC%
echo Launching TensorBoard monitoring UI...

echo.
start "TensorBoard" cmd /k "tensorboard --logdir=%OUTPUT_DIR% --port=6006 && echo TensorBoard started, visit http://localhost:6006"
echo TensorBoard has been started in a new window
echo Access URL: http://localhost:6006
echo.
goto :start_training

:test_mode
echo.
echo %YELLOW%=== Test Mode ===%NC%
echo Running a quick test...

set PYTHONPATH=%CD%;%PYTHONPATH%
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU count: {torch.cuda.device_count()}')
    print(f'Current GPU: {torch.cuda.get_device_name()}')
print('Environment test complete')
"

echo.
echo Test complete, press any key to continue training...
pause >nul
goto :start_training

:show_config
echo.
echo %PURPLE%=== Configuration Info ===%NC%
echo.
echo Project path: %CD%
echo Python path: 
where python
echo.
echo Config file content:
type "%CONFIG_FILE%"
echo.
echo Press any key to return to the main menu...
pause >nul
goto :start_training

:end
echo.
echo Press any key to exit...
pause >nul