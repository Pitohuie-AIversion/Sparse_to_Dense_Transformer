@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

REM =============================================================================
REM Windows Startup Script - Training Management System
REM Suitable for Windows environment training management launcher
REM =============================================================================

echo ╔══════════════════════════════════════════════════════════════════════════════╗
echo ║                      Training Management System v1.0                        ║
echo ║                 VIV Transformer Pressure Field Training                     ║
echo ╚══════════════════════════════════════════════════════════════════════════════╝
echo.
echo Starting training management system...
echo.

REM Check if Git Bash is available
where bash >nul 2>&1
if %errorlevel% == 0 (
    echo [INFO] Using Git Bash to launch training management system
    bash training_manager.sh
) else (
    echo [WARNING] bash command not found, trying to use WSL
    where wsl >nul 2>&1
    if !errorlevel! == 0 (
        echo [INFO] Using WSL to launch training management system
        wsl bash training_manager.sh
    ) else (
        echo [ERROR] Neither bash nor WSL environment found
        echo.
        echo Please install one of the following environments:
        echo 1. Git for Windows (recommended)
        echo 2. Windows Subsystem for Linux (WSL)
        echo.
        echo Or use the following Python command to start training directly:
        echo python train_pressure_field.py
        echo.
        pause
        exit /b 1
    )
)

pause