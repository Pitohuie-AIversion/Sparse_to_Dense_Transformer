@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

REM =============================================================================
REM Windows启动脚本 - 训练管理系统
REM 适用于Windows环境的训练管理启动器
REM =============================================================================

echo ╔══════════════════════════════════════════════════════════════════════════════╗
echo ║                          训练管理系统 v1.0                                  ║
echo ║                     VIV Transformer 压力场重建训练                          ║
echo ╚══════════════════════════════════════════════════════════════════════════════╝
echo.
echo 正在启动训练管理系统...
echo.

REM 检查Git Bash是否可用
where bash >nul 2>&1
if %errorlevel% == 0 (
    echo [INFO] 使用Git Bash启动训练管理系统
    bash training_manager.sh
) else (
    echo [WARNING] 未找到bash命令，尝试使用WSL
    where wsl >nul 2>&1
    if !errorlevel! == 0 (
        echo [INFO] 使用WSL启动训练管理系统
        wsl bash training_manager.sh
    ) else (
        echo [ERROR] 未找到bash或WSL环境
        echo.
        echo 请安装以下任一环境：
        echo 1. Git for Windows (推荐)
        echo 2. Windows Subsystem for Linux (WSL)
        echo.
        echo 或者使用以下Python命令直接启动训练：
        echo python train_pressure_field.py
        echo.
        pause
        exit /b 1
    )
)

pause