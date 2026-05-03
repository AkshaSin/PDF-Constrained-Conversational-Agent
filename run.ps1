# run.ps1 — Helper script to run the app using the virtual environment
# Usage: .\run.ps1

Write-Host "🚀 Starting PDF-Constrained Conversational Agent..." -ForegroundColor Cyan

# 1. Check if Redis is running
$redisCli = "C:\Redis\redis-cli.exe"
$redisRunning = $false

if (Test-Path $redisCli) {
    $check = & $redisCli ping 2>$null
    if ($check -eq "PONG") {
        Write-Host "✅ Redis is running" -ForegroundColor Green
        $redisRunning = $true
    }
}

if (-not $redisRunning) {
    Write-Host "⚠️ Warning: Could not detect local Redis. If you are using Upstash, ignore this. If using local Redis, make sure you ran .\start_redis.ps1 first." -ForegroundColor Yellow
}

# 2. Check if venv exists
if (-not (Test-Path "venv\Scripts\python.exe")) {
    Write-Host "❌ Virtual environment not found! Please run 'python -m venv venv' and install requirements." -ForegroundColor Red
    exit 1
}

# 3. Run the app using the venv Python
Write-Host "✅ Starting Gradio server..." -ForegroundColor Green
.\venv\Scripts\python.exe app.py
