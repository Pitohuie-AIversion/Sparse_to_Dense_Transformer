@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

echo.
echo ========================================
echo Simple Visual Training Launcher
echo ========================================
echo.

:: Check Python environment
echo Checking Python environment...
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python not found
    echo Please ensure Python is installed and added to PATH
    pause
    exit /b 1
)

for /f "tokens=*" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo Python environment: !PYTHON_VERSION!

:: Check config file
if not exist "demo_config.yaml" (
    echo Error: Config file demo_config.yaml not found
    echo Please ensure config file exists
    pause
    exit /b 1
)
echo Config file: demo_config.yaml

:: Check training script
if not exist "simple_visual_training.py" (
    echo Error: Training script simple_visual_training.py not found
    pause
    exit /b 1
)
echo Training script: simple_visual_training.py

echo.
echo Starting training...
echo ========================================
echo.

:: Set environment variable to avoid OpenMP warning
set KMP_DUPLICATE_LIB_OK=TRUE

:: Run training
python simple_visual_training.py --config demo_config.yaml

set EXIT_CODE=%errorlevel%

echo.
echo ========================================
if %EXIT_CODE% equ 0 (
    echo Training completed successfully!
    echo.
    echo View results:
    echo - Training plots: demo_training_output\plots\
    echo - Training log: demo_training_output\training_log.txt
    echo - Training summary: demo_training_output\training_summary.txt
    echo.
    echo Press any key to open output directory...
    pause >nul
    if exist "demo_training_output" (
        explorer "demo_training_output"
    )
) else (
    echo Training failed with exit code: %EXIT_CODE%
    echo Please check error messages
    pause
)

exit /b %EXIT_CODE%