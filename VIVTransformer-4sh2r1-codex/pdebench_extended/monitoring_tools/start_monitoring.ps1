# VIV Transformer Monitoring Launcher (PowerShell Version)
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "   VIV Transformer Monitoring Launcher" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

$PROJECT_DIR = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $PROJECT_DIR

Write-Host "[$(Get-Date -Format 'HH:mm:ss')] Starting all monitoring tools..." -ForegroundColor Green
Write-Host ""

# Check Python environment
try {
    python --version | Out-Null
    $PYTHON_AVAILABLE = $true
} catch {
    Write-Host "[ERROR] Python not found, please ensure Python is installed and in PATH" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

# Check TensorBoard
try {
    python -c "import tensorboard" 2>$null
    $TENSORBOARD_AVAILABLE = $true
} catch {
    Write-Host "[WARNING] TensorBoard not installed, skipping TensorBoard startup" -ForegroundColor Yellow
    $TENSORBOARD_AVAILABLE = $false
}

# Check NVIDIA GPU
try {
    nvidia-smi | Out-Null
    $GPU_AVAILABLE = $true
} catch {
    Write-Host "[WARNING] NVIDIA GPU not available, skipping GPU monitoring" -ForegroundColor Yellow
    $GPU_AVAILABLE = $false
}

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "1. Starting TensorBoard Monitoring" -ForegroundColor Cyan
if ($TENSORBOARD_AVAILABLE) {
    Write-Host "[$(Get-Date -Format 'HH:mm:ss')] Starting TensorBoard..." -ForegroundColor Green
    Start-Process powershell -ArgumentList "-NoExit", "-Command", "Write-Host 'TensorBoard Monitor - Visit http://localhost:6006' -ForegroundColor Green; python start_tensorboard.py" -WindowStyle Normal
    Start-Sleep -Seconds 2
    Write-Host "[SUCCESS] TensorBoard started" -ForegroundColor Green
} else {
    Write-Host "[SKIP] TensorBoard not available" -ForegroundColor Yellow
}
Write-Host ""

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "2. Starting GPU Monitoring" -ForegroundColor Cyan
if ($GPU_AVAILABLE) {
    Write-Host "[$(Get-Date -Format 'HH:mm:ss')] Starting GPU monitoring..." -ForegroundColor Green
    Start-Process powershell -ArgumentList "-NoExit", "-Command", "Write-Host 'GPU Status Monitor' -ForegroundColor Green; while(`$true) { Clear-Host; Write-Host 'GPU Status Monitor' -ForegroundColor Green; nvidia-smi; Start-Sleep -Seconds 5 }" -WindowStyle Normal
    Write-Host "[SUCCESS] GPU monitoring started" -ForegroundColor Green
} else {
    Write-Host "[SKIP] GPU not available" -ForegroundColor Yellow
}
Write-Host ""

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "3. Starting System Monitoring" -ForegroundColor Cyan
Write-Host "[$(Get-Date -Format 'HH:mm:ss')] Starting system monitoring..." -ForegroundColor Green
Start-Process powershell -ArgumentList "-NoExit", "-Command", "Write-Host 'System Resource Monitor' -ForegroundColor Green; while(`$true) { Clear-Host; Write-Host 'System Resource Monitor' -ForegroundColor Green; Write-Host '[$(Get-Date -Format 'HH:mm:ss')] System Status:' -ForegroundColor Yellow; Get-WmiObject -Class Win32_Processor | Select-Object -ExpandProperty LoadPercentage; Get-WmiObject -Class Win32_OperatingSystem | Select-Object TotalVisibleMemorySize, FreePhysicalMemory; Start-Sleep -Seconds 10 }" -WindowStyle Normal
Write-Host "[SUCCESS] System monitoring started" -ForegroundColor Green
Write-Host ""

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "4. Starting Training Log Monitoring" -ForegroundColor Cyan
if (Test-Path "outputs") {
    Write-Host "[$(Get-Date -Format 'HH:mm:ss')] Starting training log monitoring..." -ForegroundColor Green
    Start-Process powershell -ArgumentList "-NoExit", "-Command", "Write-Host 'Training Log Monitor' -ForegroundColor Green; Set-Location outputs; while(`$true) { Clear-Host; Write-Host 'Training Log Monitor' -ForegroundColor Green; Get-ChildItem *.log -ErrorAction SilentlyContinue | ForEach-Object { Write-Host `$_.Name -ForegroundColor Yellow }; Start-Sleep -Seconds 15 }" -WindowStyle Normal
    Write-Host "[SUCCESS] Training log monitoring started" -ForegroundColor Green
} else {
    Write-Host "[SKIP] Output directory does not exist" -ForegroundColor Yellow
}
Write-Host ""

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "5. Starting Process Monitoring" -ForegroundColor Cyan
Write-Host "[$(Get-Date -Format 'HH:mm:ss')] Starting process monitoring..." -ForegroundColor Green
Start-Process powershell -ArgumentList "-NoExit", "-Command", "Write-Host 'Python Process Monitor' -ForegroundColor Green; while(`$true) { Clear-Host; Write-Host 'Python Process Monitor' -ForegroundColor Green; Write-Host '[$(Get-Date -Format 'HH:mm:ss')] Python Training Processes:' -ForegroundColor Yellow; Get-Process python -ErrorAction SilentlyContinue | Format-Table -AutoSize; Start-Sleep -Seconds 8 }" -WindowStyle Normal
Write-Host "[SUCCESS] Process monitoring started" -ForegroundColor Green
Write-Host ""

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "        Monitoring Tools Started!" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Started monitoring windows:" -ForegroundColor Green
if ($TENSORBOARD_AVAILABLE) { Write-Host "  * TensorBoard: http://localhost:6006" -ForegroundColor White }
if ($GPU_AVAILABLE) { Write-Host "  * GPU Monitor: Real-time GPU status" -ForegroundColor White }
Write-Host "  * System Monitor: CPU/Memory/Disk status" -ForegroundColor White
Write-Host "  * Training Logs: Real-time training logs" -ForegroundColor White
Write-Host "  * Process Monitor: Python process status" -ForegroundColor White
Write-Host ""
Write-Host "Tips:" -ForegroundColor Yellow
Write-Host "  * Closing any monitoring window will not affect others" -ForegroundColor White
Write-Host "  * Press Ctrl+C to stop corresponding monitor" -ForegroundColor White
Write-Host "  * Re-run this script to restart all monitoring" -ForegroundColor White
Write-Host ""
Read-Host "Press Enter to exit launcher"