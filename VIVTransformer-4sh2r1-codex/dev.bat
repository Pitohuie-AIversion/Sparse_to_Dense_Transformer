@echo off
REM VIVTransformer 开发工具批处理脚本
REM 为Windows用户提供便捷的开发命令

setlocal enabledelayedexpansion

REM 设置颜色代码
set "GREEN=[92m"
set "YELLOW=[93m"
set "RED=[91m"
set "BLUE=[94m"
set "RESET=[0m"

REM 检查参数
if "%1"=="" goto :help
if "%1"=="help" goto :help
if "%1"=="--help" goto :help
if "%1"=="-h" goto :help

REM 主要命令分发
if "%1"=="setup" goto :setup
if "%1"=="install" goto :install
if "%1"=="clean" goto :clean
if "%1"=="format" goto :format
if "%1"=="format-check" goto :format_check
if "%1"=="lint" goto :lint
if "%1"=="quality" goto :quality
if "%1"=="fix" goto :fix
if "%1"=="test" goto :test
if "%1"=="test-fast" goto :test_fast
if "%1"=="quick-test" goto :quick_test
if "%1"=="dashboard" goto :dashboard
if "%1"=="quality-gate" goto :quality_gate
if "%1"=="pre-commit" goto :pre_commit
if "%1"=="ci-check" goto :ci_check
if "%1"=="all-checks" goto :all_checks
if "%1"=="daily" goto :daily
if "%1"=="commit-ready" goto :commit_ready
if "%1"=="newbie" goto :newbie
if "%1"=="status" goto :status

echo %RED%错误: 未知命令 "%1"%RESET%
echo 使用 "dev help" 查看可用命令
goto :end

:help
echo %BLUE%VIVTransformer 开发工具%RESET%
echo ============================
echo.
echo %YELLOW%环境设置:%RESET%
echo   setup          - 一键设置完整开发环境
echo   install        - 安装项目依赖
echo   clean          - 清理临时文件和缓存
echo.
echo %YELLOW%代码质量:%RESET%
echo   format         - 代码格式化
echo   format-check   - 检查代码格式（不修改）
echo   lint           - 代码风格检查
echo   quality        - 完整代码质量检查
echo   fix            - 自动修复代码问题
echo.
echo %YELLOW%测试:%RESET%
echo   test           - 运行完整测试套件
echo   test-fast      - 运行快速测试
echo   quick-test     - 快速功能验证
echo.
echo %YELLOW%报告和检查:%RESET%
echo   dashboard      - 生成质量仪表板
echo   quality-gate   - 质量门禁检查
echo   pre-commit     - 运行pre-commit钩子
echo   ci-check       - 模拟CI检查流程
echo   all-checks     - 运行所有检查（推荐提交前使用）
echo.
echo %YELLOW%快捷工作流:%RESET%
echo   daily          - 日常开发工作流
echo   commit-ready   - 提交前完整检查
echo   newbie         - 新开发者环境设置
echo   status         - 显示项目状态
echo.
echo %YELLOW%示例用法:%RESET%
echo   dev setup      # 首次设置环境
echo   dev daily      # 日常开发检查
echo   dev commit-ready # 提交前检查
echo   dev dashboard  # 查看质量报告
goto :end

:setup
echo %GREEN%🚀 设置VIVTransformer开发环境...%RESET%
python setup_dev_env.py
if errorlevel 1 (
    echo %RED%❌ 环境设置失败%RESET%
    exit /b 1
)
echo %GREEN%✅ 环境设置完成%RESET%
goto :end

:install
echo %GREEN%📦 安装项目依赖...%RESET%
python -m pip install --upgrade pip
if exist requirements.txt (
    pip install -r requirements.txt
) else (
    echo %YELLOW%⚠️ requirements.txt 不存在%RESET%
)
if exist pyproject.toml (
    pip install -e ".[dev]"
) else (
    echo %YELLOW%⚠️ pyproject.toml 不存在，跳过开发依赖安装%RESET%
)
echo %GREEN%✅ 依赖安装完成%RESET%
goto :end

:clean
echo %GREEN%🧹 清理临时文件...%RESET%
if exist __pycache__ rmdir /s /q __pycache__ 2>nul
if exist .pytest_cache rmdir /s /q .pytest_cache 2>nul
if exist .mypy_cache rmdir /s /q .mypy_cache 2>nul
if exist .coverage del .coverage 2>nul
if exist htmlcov rmdir /s /q htmlcov 2>nul
if exist dist rmdir /s /q dist 2>nul
if exist build rmdir /s /q build 2>nul
for /r . %%i in (*.pyc) do del "%%i" 2>nul
for /r . %%i in (*.pyo) do del "%%i" 2>nul
for /d /r . %%i in (*.egg-info) do rmdir /s /q "%%i" 2>nul
echo %GREEN%✅ 清理完成%RESET%
goto :end

:format
echo %GREEN%🎨 格式化代码...%RESET%
if exist scripts\format_code.py (
    python scripts\format_code.py --target-dirs modify_multi_attention tests scripts
) else (
    echo %RED%❌ scripts\format_code.py 不存在%RESET%
    exit /b 1
)
echo %GREEN%✅ 代码格式化完成%RESET%
goto :end

:format_check
echo %GREEN%🔍 检查代码格式...%RESET%
if exist scripts\format_code.py (
    python scripts\format_code.py --check --target-dirs modify_multi_attention tests scripts
) else (
    echo %RED%❌ scripts\format_code.py 不存在%RESET%
    exit /b 1
)
goto :end

:lint
echo %GREEN%📋 运行代码风格检查...%RESET%
if exist scripts\code_quality.py (
    python scripts\code_quality.py --tools flake8,pylint --target-dirs modify_multi_attention tests
) else (
    echo %RED%❌ scripts\code_quality.py 不存在%RESET%
    exit /b 1
)
goto :end

:quality
echo %GREEN%⭐ 运行完整质量检查...%RESET%
if exist scripts\code_quality.py (
    python scripts\code_quality.py --tools all --show-details
) else (
    echo %RED%❌ scripts\code_quality.py 不存在%RESET%
    exit /b 1
)
goto :end

:fix
echo %GREEN%🔧 自动修复代码问题...%RESET%
if exist scripts\auto_fix.py (
    python scripts\auto_fix.py --target-dirs modify_multi_attention tests scripts
) else (
    echo %RED%❌ scripts\auto_fix.py 不存在%RESET%
    exit /b 1
)
echo %GREEN%✅ 自动修复完成%RESET%
goto :end

:test
echo %GREEN%🧪 运行完整测试套件...%RESET%
python -m pytest tests\ -v
if errorlevel 1 (
    echo %RED%❌ 测试失败%RESET%
    exit /b 1
)
echo %GREEN%✅ 测试通过%RESET%
goto :end

:test_fast
echo %GREEN%⚡ 运行快速测试...%RESET%
python -m pytest tests\ -v -x --tb=short
if errorlevel 1 (
    echo %RED%❌ 快速测试失败%RESET%
    exit /b 1
)
echo %GREEN%✅ 快速测试通过%RESET%
goto :end

:quick_test
echo %GREEN%⚡ 快速功能验证...%RESET%
if exist scripts\quick_test.py (
    python scripts\quick_test.py
) else (
    echo %RED%❌ scripts\quick_test.py 不存在%RESET%
    exit /b 1
)
goto :end

:dashboard
echo %GREEN%📊 生成质量仪表板...%RESET%
if exist scripts\quality_dashboard.py (
    python scripts\quality_dashboard.py --target-dirs modify_multi_attention tests scripts
    echo %GREEN%📋 质量报告已生成，请查看 reports\quality_dashboard.html%RESET%
) else (
    echo %RED%❌ scripts\quality_dashboard.py 不存在%RESET%
    exit /b 1
)
goto :end

:quality_gate
echo %GREEN%🚪 运行质量门禁检查...%RESET%
if exist scripts\quality_gate.py (
    if exist quality_gate.json (
        python scripts\quality_gate.py --config quality_gate.json
    ) else (
        python scripts\quality_gate.py
    )
) else (
    echo %RED%❌ scripts\quality_gate.py 不存在%RESET%
    exit /b 1
)
goto :end

:pre_commit
echo %GREEN%🪝 运行pre-commit钩子...%RESET%
pre-commit run --all-files
if errorlevel 1 (
    echo %YELLOW%⚠️ Pre-commit检查发现问题%RESET%
)
goto :end

:ci_check
echo %GREEN%🔄 模拟CI检查流程...%RESET%
call :fix
if errorlevel 1 goto :end
call :format_check
if errorlevel 1 goto :end
call :quality
if errorlevel 1 goto :end
call :test
if errorlevel 1 goto :end
call :quality_gate
if errorlevel 1 goto :end
echo %GREEN%✅ CI检查完成%RESET%
goto :end

:all_checks
echo %GREEN%🎯 运行所有检查（推荐提交前使用）...%RESET%
call :fix
if errorlevel 1 goto :end
call :format
if errorlevel 1 goto :end
call :quality
if errorlevel 1 goto :end
call :test
if errorlevel 1 goto :end
call :quality_gate
if errorlevel 1 goto :end
echo %GREEN%🎉 所有检查完成！代码已准备好提交。%RESET%
goto :end

:daily
echo %GREEN%📅 日常开发工作流...%RESET%
call :fix
if errorlevel 1 goto :end
call :format
if errorlevel 1 goto :end
call :quick_test
if errorlevel 1 goto :end
echo %GREEN%✅ 日常检查完成%RESET%
goto :end

:commit_ready
echo %GREEN%📝 提交前完整检查...%RESET%
call :all_checks
if errorlevel 1 goto :end
call :dashboard
echo %GREEN%🎉 代码已准备好提交！%RESET%
goto :end

:newbie
echo %GREEN%👋 新开发者环境设置...%RESET%
call :setup
if errorlevel 1 goto :end
echo %GREEN%⚙️ 安装pre-commit钩子...%RESET%
pre-commit install
pre-commit install --hook-type pre-push
call :quick_test
echo %GREEN%🎉 欢迎加入VIVTransformer开发！%RESET%
goto :end

:status
echo %GREEN%📊 项目状态:%RESET%
echo.
echo %YELLOW%Python版本:%RESET%
python --version
echo.
echo %YELLOW%Pip版本:%RESET%
pip --version
echo.
echo %YELLOW%当前目录:%RESET%
cd
echo.
echo %YELLOW%Git状态:%RESET%
git status --porcelain 2>nul || echo Git未初始化
echo.
echo %YELLOW%依赖状态:%RESET%
pip check 2>nul || echo 依赖检查失败
echo.
echo %YELLOW%快速测试:%RESET%
call :quick_test
goto :end

:end
endlocal