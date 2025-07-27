@echo off
chcp 65001 >nul
echo ========================================
echo Fixed Epoch Prediction Visualization Demo
echo ========================================
echo.

REM Set environment variable to avoid OpenMP conflicts
set KMP_DUPLICATE_LIB_OK=TRUE

echo Starting prediction visualization demo...
echo.

REM Run the prediction visualization demo
python create_prediction_visualization.py --demo

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ========================================
    echo Demo completed successfully!
    echo ========================================
    echo.
    echo Generated visualizations can be found in:
    echo - demo_prediction_visualizations/single_samples/
    echo - demo_prediction_visualizations/batch_comparisons/
    echo - demo_prediction_visualizations/error_analysis/
    echo - demo_prediction_visualizations/time_series/
    echo.
    echo Check the visualization report for details:
    echo - demo_prediction_visualizations/epoch_10_visualization_report.md
    echo.
) else (
    echo.
    echo ========================================
    echo Error occurred during demo execution!
    echo ========================================
    echo.
    echo Please check:
    echo 1. Python environment is properly set up
    echo 2. Required packages are installed (matplotlib, numpy, seaborn)
    echo 3. Current directory contains the script
    echo.
)

echo Press any key to exit...
pause >nul