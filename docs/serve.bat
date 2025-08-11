@echo off
echo 🚀 Starting VIVTransformer docs local preview...
echo.

REM Check current directory is docs
if not exist "_config.yml" (
    echo ❌ Error: Please run this script inside the docs directory
    echo Current directory: %CD%
    pause
    exit /b 1
)

REM Check Ruby and Bundler
where ruby >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo ❌ Error: Ruby not found. Please install Ruby first
    echo Download: https://rubyinstaller.org/
    pause
    exit /b 1
)

where bundle >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo ❌ Error: Bundler not found. Installing...
    gem install bundler
    if %ERRORLEVEL% neq 0 (
        echo ❌ Bundler installation failed
        pause
        exit /b 1
    )
)

REM Install dependencies on first run (no Gemfile.lock)
if not exist "Gemfile.lock" (
    echo 📦 First run detected, installing dependencies...
    bundle install
    if %ERRORLEVEL% neq 0 (
        echo ❌ Dependency installation failed
        pause
        exit /b 1
    )
    echo ✅ Dependencies installed
    echo.
)

REM Start Jekyll server
echo 🌐 Starting Jekyll server...
echo 📍 Local URL: http://localhost:4000
echo 🔄 Press Ctrl+C to stop the server
echo.

bundle exec jekyll serve --open-url

if %ERRORLEVEL% neq 0 (
    echo.
    echo ❌ Jekyll server failed to start
    echo 💡 Try the following:
    echo    1. Run 'bundle install' to reinstall dependencies
    echo    2. Check if port 4000 is in use
    echo    3. Review the error details above
    pause
)

echo.
echo 👋 Jekyll server stopped
pause