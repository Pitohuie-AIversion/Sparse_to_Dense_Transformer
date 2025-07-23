@echo off
chcp 65001 >nul
echo ========================================
echo 🎯 压力场重建训练 - 带可视化功能
echo ========================================
echo.
echo 正在启动训练程序...
echo.

:: 检查Python是否可用
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ 错误: 未找到Python，请确保Python已正确安装并添加到PATH
    pause
    exit /b 1
)

:: 运行训练脚本
python run_training_with_vis.py

echo.
echo 训练完成！按任意键退出...
pause >nul