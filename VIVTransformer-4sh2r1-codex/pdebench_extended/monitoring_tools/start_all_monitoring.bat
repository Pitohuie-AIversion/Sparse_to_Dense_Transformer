@echo off
setlocal enabledelayedexpansion

echo ========================================
echo    VIV Transformer Monitoring Launcher
echo ========================================
echo.

set "PROJECT_DIR=%~dp0"
cd /d "%PROJECT_DIR%"

echo [%time%] Starting all monitoring tools...
echo.

REM Check Python environment
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python not found, please ensure Python is installed and in PATH
    pause
    exit /b 1
)

REM Check TensorBoard
python -c "import tensorboard" >nul 2>&1
if %errorlevel% neq 0 (
    echo [WARNING] TensorBoard not installed, skipping TensorBoard startup
    set "TENSORBOARD_AVAILABLE=0"
) else (
    set "TENSORBOARD_AVAILABLE=1"
)

REM Check NVIDIA GPU
nvidia-smi >nul 2>&1
if %errorlevel% neq 0 (
    echo [WARNING] NVIDIA GPU not available, skipping GPU monitoring
    set "GPU_AVAILABLE=0"
) else (
    set "GPU_AVAILABLE=1"
)

echo ========================================
echo 1. Starting TensorBoard Monitoring
if %TENSORBOARD_AVAILABLE%==1 (
    echo [%time%] Starting TensorBoard...
    start "TensorBoard" cmd /k "echo TensorBoard Monitor - Visit http://localhost:6006 && python start_tensorboard.py"
    timeout /t 2 >nul
    echo [SUCCESS] TensorBoard started
) else (
    echo [SKIP] TensorBoard not available
)
echo.

echo ========================================
echo 2. Starting GPU Monitoring
if %GPU_AVAILABLE%==1 (
    echo [%time%] Starting GPU monitoring...
    start "GPU Monitor" cmd /k "echo GPU Status Monitor && :loop && nvidia-smi && timeout /t 5 >nul && goto loop"
    echo [SUCCESS] GPU monitoring started
) else (
    echo [SKIP] GPU not available
)
echo.

echo ========================================
echo 3. Starting System Monitoring
echo [%time%] Starting system monitoring...
start "System Monitor" cmd /k "echo System Resource Monitor && :loop && echo [%time%] System Status: && wmic cpu get loadpercentage /value && wmic OS get TotalVisibleMemorySize,FreePhysicalMemory /value && timeout /t 10 >nul && goto loop"
echo [SUCCESS] System monitoring started
echo.

echo ========================================
echo 4. Starting Training Log Monitoring
if exist "outputs" (
    echo [%time%] Starting training log monitoring...
    start "Training Logs" cmd /k "echo Training Log Monitor && cd outputs && :loop && dir *.log 2>nul && timeout /t 15 >nul && goto loop"
    echo [SUCCESS] Training log monitoring started
) else (
    echo [SKIP] Output directory does not exist
)
echo.

echo ========================================
echo 5. Starting Process Monitoring
echo [%time%] Starting process monitoring...
start "Process Monitor" cmd /k "echo Python Process Monitor && :loop && echo [%time%] Python Training Processes: && tasklist /fi imagename eq python.exe /fo table && timeout /t 8 >nul && goto loop"
echo [SUCCESS] Process monitoring started
echo.

echo ========================================
echo        Monitoring Tools Started!
echo ========================================
echo.
echo Started monitoring windows:
if %TENSORBOARD_AVAILABLE%==1 echo   * TensorBoard: http://localhost:6006
if %GPU_AVAILABLE%==1 echo   * GPU Monitor: Real-time GPU status
echo   * System Monitor: CPU/Memory/Disk status
echo   * Training Logs: Real-time training logs
echo   * Process Monitor: Python process status
echo.
echo Tips:
echo   * Closing any monitoring window will not affect others
echo   * Press Ctrl+C to stop corresponding monitor
echo   * Re-run this script to restart all monitoring
echo.
echo Press any key to exit launcher...
pause >nul
exit /b 0