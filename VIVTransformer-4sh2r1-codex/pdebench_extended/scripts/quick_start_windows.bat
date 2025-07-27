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
echo %PURPLE%=== 数据集检查与下载 ===%NC%

REM 定义数据集相关变量
set "DATA_REPO_URL=https://github.com/pdebench/PDEBench.git"
set "DATA_DIR=data"
set "ALTERNATIVE_DATA_URLS[0]=https://darus.uni-stuttgart.de/api/access/datafile/132004"
set "ALTERNATIVE_DATA_URLS[1]=https://zenodo.org/record/6222489/files/2D_CFD_Rand_M0.1_Eta1e-08_Zeta1e-08_periodic_Train.hdf5"

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
    if exist "%DATA_DIR%" (
        for /r "%DATA_DIR%" %%f in (*.pt *.hdf5 *.h5) do (
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
        echo %YELLOW%⚠️  未找到现有数据文件，启动下载...%NC%
        echo.
        echo 数据下载选项:
        echo 1. 克隆PDEBench仓库 (推荐，包含完整数据集)
        echo 2. 直接下载数据文件 (快速，仅下载训练数据)
        echo 3. 生成示例数据 (用于测试)
        echo 4. 手动指定路径
        echo 0. 跳过下载 (稍后手动设置)
        echo.
        set /p "DOWNLOAD_CHOICE=请选择 [0-4]: "
        
        if "!DOWNLOAD_CHOICE!"=="1" goto :clone_repo
        if "!DOWNLOAD_CHOICE!"=="2" goto :direct_download
        if "!DOWNLOAD_CHOICE!"=="3" goto :generate_sample
        if "!DOWNLOAD_CHOICE!"=="4" goto :manual_path
        if "!DOWNLOAD_CHOICE!"=="0" goto :skip_download
        
        echo %RED%无效选择，跳过下载%NC%
        goto :skip_download
        
        :clone_repo
        echo.
        echo %CYAN%=== 克隆PDEBench仓库 ===%NC%
        
        REM 检查git
        git --version >nul 2>&1
        if %errorlevel% neq 0 (
            echo %RED%❌ Git未安装，请先安装Git%NC%
            echo 下载地址: https://git-scm.com/download/win
            pause
            goto :skip_download
        )
        
        if not exist "%DATA_DIR%" mkdir "%DATA_DIR%"
        
        if not exist "%DATA_DIR%\PDEBench" (
            echo 正在克隆仓库...
            cd /d "%DATA_DIR%"
            git clone --depth 1 "%DATA_REPO_URL%" PDEBench
            if %errorlevel% == 0 (
                echo %GREEN%✓ 仓库克隆成功%NC%
                
                REM 搜索数据文件
                for /r "PDEBench" %%f in (*.hdf5 *.pt *.h5) do (
                    if exist "%%f" (
                        set "DATA_PATH=%%f"
                        echo %GREEN%✓ 找到数据文件: %%f%NC%
                        set "DATA_FOUND=1"
                        goto :clone_done
                    )
                )
                :clone_done
            ) else (
                echo %RED%❌ 仓库克隆失败%NC%
            )
            cd /d "%~dp0"
        ) else (
            echo %GREEN%✓ PDEBench仓库已存在%NC%
            for /r "%DATA_DIR%\PDEBench" %%f in (*.hdf5 *.pt *.h5) do (
                if exist "%%f" (
                    set "DATA_PATH=%%f"
                    echo %GREEN%✓ 使用现有数据文件: %%f%NC%
                    set "DATA_FOUND=1"
                    goto :repo_exists_done
                )
            )
            :repo_exists_done
        )
        goto :download_complete
        
        :direct_download
        echo.
        echo %CYAN%=== 直接下载数据文件 ===%NC%
        
        REM 检查下载工具
        set "DOWNLOAD_TOOL="
        curl --version >nul 2>&1
        if %errorlevel% == 0 (
            set "DOWNLOAD_TOOL=curl"
        ) else (
            powershell -Command "Get-Command Invoke-WebRequest" >nul 2>&1
            if %errorlevel% == 0 (
                set "DOWNLOAD_TOOL=powershell"
            ) else (
                echo %RED%❌ 未找到下载工具 (curl或PowerShell)%NC%
                goto :skip_download
            )
        )
        
        if not exist "%DATA_DIR%" mkdir "%DATA_DIR%"
        
        echo 尝试下载数据文件...
        set "DOWNLOAD_SUCCESS=0"
        
        REM 尝试第一个URL
        set "FILENAME=%DATA_DIR%\pressure_data_1.hdf5"
        echo 正在下载: !ALTERNATIVE_DATA_URLS[0]!
        if "!DOWNLOAD_TOOL!"=="curl" (
            curl -L -o "!FILENAME!" "!ALTERNATIVE_DATA_URLS[0]!"
        ) else (
            powershell -Command "Invoke-WebRequest -Uri '!ALTERNATIVE_DATA_URLS[0]!' -OutFile '!FILENAME!'"
        )
        
        if exist "!FILENAME!" (
            for %%A in ("!FILENAME!") do set "FILE_SIZE=%%~zA"
            if !FILE_SIZE! gtr 1000000 (
                set "DATA_PATH=!FILENAME!"
                set "DATA_FOUND=1"
                set "DOWNLOAD_SUCCESS=1"
                echo %GREEN%✓ 下载成功: !FILENAME!%NC%
            ) else (
                del "!FILENAME!"
                echo %YELLOW%⚠️  下载的文件太小，尝试下一个源%NC%
            )
        )
        
        REM 如果第一个失败，尝试第二个URL
        if "!DOWNLOAD_SUCCESS!"=="0" (
            set "FILENAME=%DATA_DIR%\pressure_data_2.hdf5"
            echo 正在下载: !ALTERNATIVE_DATA_URLS[1]!
            if "!DOWNLOAD_TOOL!"=="curl" (
                curl -L -o "!FILENAME!" "!ALTERNATIVE_DATA_URLS[1]!"
            ) else (
                powershell -Command "Invoke-WebRequest -Uri '!ALTERNATIVE_DATA_URLS[1]!' -OutFile '!FILENAME!'"
            )
            
            if exist "!FILENAME!" (
                for %%A in ("!FILENAME!") do set "FILE_SIZE=%%~zA"
                if !FILE_SIZE! gtr 1000000 (
                    set "DATA_PATH=!FILENAME!"
                    set "DATA_FOUND=1"
                    echo %GREEN%✓ 下载成功: !FILENAME!%NC%
                ) else (
                    del "!FILENAME!"
                    echo %RED%❌ 所有下载源都失败%NC%
                )
            )
        )
        goto :download_complete
        
        :generate_sample
        echo.
        echo %CYAN%=== 生成示例数据 ===%NC%
        
        if not exist "%DATA_DIR%" mkdir "%DATA_DIR%"
        
        REM 创建Python脚本生成示例数据
        (
            echo import torch
            echo import numpy as np
            echo from pathlib import Path
            echo.
            echo def generate_sample_data^(^):
            echo     print^("生成示例压力场数据..."^)
            echo     
            echo     # 创建小规模示例数据
            echo     batch_size = 100
            echo     height, width = 64, 64
            echo     time_steps = 10
            echo     
            echo     # 生成压力场数据
            echo     pressure_data = torch.randn^(batch_size, time_steps, height, width, dtype=torch.float32^)
            echo     
            echo     # 添加边界条件
            echo     pressure_data[:, :, 0, :] = 0  # 上边界
            echo     pressure_data[:, :, -1, :] = 0  # 下边界
            echo     pressure_data[:, :, :, 0] = 0  # 左边界
            echo     pressure_data[:, :, :, -1] = 0  # 右边界
            echo     
            echo     # 保存为PyTorch格式
            echo     output_file = Path^("sample_pressure_data.pt"^)
            echo     torch.save^({
            echo         'pressure': pressure_data,
            echo         'time': torch.linspace^(0, 1, time_steps^),
            echo         'metadata': {
            echo             'description': 'Sample pressure field data for testing',
            echo             'batch_size': batch_size,
            echo             'spatial_dims': ^(height, width^),
            echo             'time_steps': time_steps
            echo         }
            echo     }, output_file^)
            echo     
            echo     print^(f"示例数据已生成: {output_file}"^)
            echo     print^(f"数据形状: {pressure_data.shape}"^)
            echo     return str^(output_file^)
            echo.
            echo if __name__ == "__main__":
            echo     generate_sample_data^(^)
        ) > "%DATA_DIR%\generate_sample.py"
        
        cd /d "%DATA_DIR%"
        python generate_sample.py
        if %errorlevel% == 0 (
            if exist "sample_pressure_data.pt" (
                set "DATA_PATH=%DATA_DIR%\sample_pressure_data.pt"
                set "DATA_FOUND=1"
                echo %GREEN%✓ 示例数据生成成功%NC%
                echo %YELLOW%⚠️  注意: 这是示例数据，仅用于测试%NC%
            )
        ) else (
            echo %RED%❌ 示例数据生成失败%NC%
        )
        cd /d "%~dp0"
        goto :download_complete
        
        :manual_path
        echo.
        set /p "MANUAL_PATH=请输入数据文件路径: "
        if exist "!MANUAL_PATH!" (
            set "DATA_PATH=!MANUAL_PATH!"
            set "DATA_FOUND=1"
            echo %GREEN%✓ 使用手动指定的数据文件%NC%
        ) else (
            echo %RED%❌ 指定的文件不存在: !MANUAL_PATH!%NC%
        )
        goto :download_complete
        
        :skip_download
        echo %YELLOW%⚠️  跳过数据下载%NC%
        echo 请手动下载数据集或设置正确的数据路径
        goto :download_complete
        
        :download_complete
    )
)

echo.
echo %PURPLE%=== 文件检查 ===%NC%

REM 最终数据文件检查
if "!DATA_FOUND!"=="1" (
    if exist "%DATA_PATH%" (
        for %%A in ("%DATA_PATH%") do set "DATA_SIZE=%%~zA"
        set /a "DATA_SIZE_MB=!DATA_SIZE! / 1024 / 1024"
        echo %GREEN%✓%NC% 数据文件: !DATA_SIZE_MB!MB
    ) else (
        echo %RED%❌ 数据文件路径无效: %DATA_PATH%%NC%
        pause
        exit /b 1
    )
) else (
    echo %RED%❌ 未找到数据文件%NC%
    echo 请确保数据文件存在或重新运行脚本选择下载
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