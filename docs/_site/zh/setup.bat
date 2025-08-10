@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

echo.
echo ========================================
echo   VIVTransformer GitHub Pages 配置工具
echo ========================================
echo.

:: 检查Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python未安装，请先安装Python 3.7+
    echo 下载地址: https://www.python.org/downloads/
    pause
    exit /b 1
)

:: 显示菜单
echo 请选择配置模式:
echo.
echo [1] 🚀 快速配置 (自动检测Git信息)
echo [2] 🔧 交互式配置 (手动输入信息)
echo [3] 📋 仅查看当前配置
echo [4] 🛠️  安装依赖环境
echo [5] 🧪 测试本地预览
echo [0] ❌ 退出
echo.
set /p choice="请输入选项 [1-5, 0]: "

if "%choice%"=="1" goto quick_setup
if "%choice%"=="2" goto interactive_setup
if "%choice%"=="3" goto show_config
if "%choice%"=="4" goto install_deps
if "%choice%"=="5" goto test_preview
if "%choice%"=="0" goto exit
goto menu

:quick_setup
echo.
echo 🚀 开始快速配置...
python setup.py --auto
goto end

:interactive_setup
echo.
echo 🔧 开始交互式配置...
python setup.py
goto end

:show_config
echo.
echo 📋 当前配置信息:
echo.
if exist "_config.yml" (
    echo === Jekyll 配置 ===
    findstr /C:"title:" /C:"description:" /C:"baseurl:" /C:"url:" "_config.yml"
    echo.
) else (
    echo ❌ 配置文件不存在
)

if exist "GITHUB_PAGES_SETUP.md" (
    echo === 部署信息 ===
    type "GITHUB_PAGES_SETUP.md" | findstr /C:"GitHub用户名" /C:"仓库名" /C:"网站地址"
    echo.
)

echo 按任意键返回菜单...
pause >nul
goto menu

:install_deps
echo.
echo 🛠️  检查和安装依赖环境...
echo.

:: 检查Ruby
ruby --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Ruby未安装
    echo 请访问 https://rubyinstaller.org/ 下载安装Ruby
    echo 建议安装Ruby 3.0+版本
    echo.
) else (
    echo ✅ Ruby已安装
    ruby --version
)

:: 检查Bundler
bundle --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Bundler未安装，正在安装...
    gem install bundler
) else (
    echo ✅ Bundler已安装
    bundle --version
)

:: 安装Jekyll依赖
if exist "Gemfile" (
    echo.
    echo 📦 安装Jekyll依赖...
    bundle install
    if errorlevel 1 (
        echo ❌ 依赖安装失败
    ) else (
        echo ✅ 依赖安装完成
    )
) else (
    echo ❌ Gemfile不存在
)

echo.
echo 按任意键返回菜单...
pause >nul
goto menu

:test_preview
echo.
echo 🧪 启动本地预览服务器...
echo.

if not exist "Gemfile" (
    echo ❌ Gemfile不存在，请先运行配置
    goto menu
)

bundle --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Bundler未安装，请先安装依赖环境
    goto menu
)

echo 正在启动Jekyll服务器...
echo 服务器启动后，请访问: http://localhost:4000
echo 按 Ctrl+C 停止服务器
echo.

bundle exec jekyll serve --host 0.0.0.0 --port 4000 --livereload
goto end

:menu
echo.
goto start

:end
echo.
echo 🎉 操作完成！
echo.
if exist "GITHUB_PAGES_SETUP.md" (
    echo 📚 查看详细部署说明: GITHUB_PAGES_SETUP.md
    echo.
)
echo 按任意键退出...
pause >nul

:exit
echo.
echo 👋 再见！
exit /b 0

:start
echo.
echo ========================================
echo   VIVTransformer GitHub Pages 配置工具
echo ========================================
echo.

:: 检查Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python未安装，请先安装Python 3.7+
    echo 下载地址: https://www.python.org/downloads/
    pause
    exit /b 1
)

:: 显示菜单
echo 请选择配置模式:
echo.
echo [1] 🚀 快速配置 (自动检测Git信息)
echo [2] 🔧 交互式配置 (手动输入信息)
echo [3] 📋 仅查看当前配置
echo [4] 🛠️  安装依赖环境
echo [5] 🧪 测试本地预览
echo [0] ❌ 退出
echo.
set /p choice="请输入选项 [1-5, 0]: "

if "%choice%"=="1" goto quick_setup
if "%choice%"=="2" goto interactive_setup
if "%choice%"=="3" goto show_config
if "%choice%"=="4" goto install_deps
if "%choice%"=="5" goto test_preview
if "%choice%"=="0" goto exit
goto menu