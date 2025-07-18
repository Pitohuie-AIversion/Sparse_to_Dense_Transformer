@echo off
echo 🚀 启动VIVTransformer文档网站本地预览...
echo.

REM 检查是否在docs目录
if not exist "_config.yml" (
    echo ❌ 错误：请在docs目录下运行此脚本
    echo 当前目录：%CD%
    pause
    exit /b 1
)

REM 检查Ruby和Bundle是否安装
where ruby >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo ❌ 错误：未找到Ruby，请先安装Ruby
    echo 下载地址：https://rubyinstaller.org/
    pause
    exit /b 1
)

where bundle >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo ❌ 错误：未找到Bundler，正在安装...
    gem install bundler
    if %ERRORLEVEL% neq 0 (
        echo ❌ Bundler安装失败
        pause
        exit /b 1
    )
)

REM 检查Gemfile.lock是否存在，如果不存在则安装依赖
if not exist "Gemfile.lock" (
    echo 📦 首次运行，正在安装依赖...
    bundle install
    if %ERRORLEVEL% neq 0 (
        echo ❌ 依赖安装失败
        pause
        exit /b 1
    )
    echo ✅ 依赖安装完成
    echo.
)

REM 启动Jekyll服务器
echo 🌐 启动Jekyll服务器...
echo 📍 本地访问地址：http://localhost:4000
echo 🔄 按 Ctrl+C 停止服务器
echo.

bundle exec jekyll serve --livereload --open-url

if %ERRORLEVEL% neq 0 (
    echo.
    echo ❌ Jekyll服务器启动失败
    echo 💡 尝试以下解决方案：
    echo    1. 运行 'bundle install' 重新安装依赖
    echo    2. 检查端口4000是否被占用
    echo    3. 查看上方错误信息
    pause
)

echo.
echo 👋 Jekyll服务器已停止
pause