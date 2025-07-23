@echo off
chcp 65001 >nul
echo ========================================
echo Enhanced Visual Training with Fixed Epoch Prediction Visualization
echo ========================================
echo.

REM Set environment variable to avoid OpenMP conflicts
set KMP_DUPLICATE_LIB_OK=TRUE

echo Starting enhanced visual training demo...
echo This will demonstrate:
echo - Real-time training metrics visualization
echo - Fixed epoch prediction comparisons
echo - Error analysis and training progress tracking
echo - Hardware monitoring and performance analysis
echo.

REM Run the enhanced visual training demo
python enhanced_visual_training.py --demo --epochs 15

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ========================================
    echo Enhanced training demo completed successfully!
    echo ========================================
    echo.
    echo Generated visualizations can be found in:
    echo - enhanced_training_output/plots/              (Training curves)
    echo - enhanced_training_output/predictions/        (Prediction visualizations)
    echo   - single_samples/                            (Individual predictions)
    echo   - batch_comparisons/                         (Batch predictions)
    echo   - error_analysis/                            (Error analysis)
    echo   - time_series/                               (Training progress)
    echo - enhanced_training_output/logs/               (Training logs)
    echo - enhanced_training_output/hardware_logs/      (Hardware monitoring)
    echo.
    echo Check the comprehensive report:
    echo - enhanced_training_output/comprehensive_training_report.md
    echo.
    echo Open the HTML report in your browser:
    echo - enhanced_training_output/hardware_logs/training_report.html
    echo.
) else (
    echo.
    echo ========================================
    echo Error occurred during enhanced training demo!
    echo ========================================
    echo.
    echo Please check:
    echo 1. Python environment is properly set up
    echo 2. Required packages are installed (torch, matplotlib, numpy, seaborn)
    echo 3. All required modules are available
    echo 4. Current directory contains all necessary scripts
    echo.
)

echo Press any key to exit...
pause >nul