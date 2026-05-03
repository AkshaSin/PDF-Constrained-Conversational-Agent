# start_redis.ps1 — Run this once before starting the app
# Usage: Right-click → "Run with PowerShell"
#        OR from terminal: .\start_redis.ps1

$redisExe = "C:\Redis\redis-server.exe"
$redisCli = "C:\Redis\redis-cli.exe"

# Check if already running
$running = & $redisCli ping 2>$null
if ($running -eq "PONG") {
    Write-Host "✅ Redis is already running on localhost:6379" -ForegroundColor Green
    exit 0
}

Write-Host "🚀 Starting Redis server..." -ForegroundColor Cyan
Start-Process -FilePath $redisExe -WindowStyle Minimized
Start-Sleep -Seconds 2

$check = & $redisCli ping 2>$null
if ($check -eq "PONG") {
    Write-Host "✅ Redis started successfully on localhost:6379" -ForegroundColor Green
} else {
    Write-Host "❌ Redis failed to start. Check C:\Redis\redis-server.exe exists." -ForegroundColor Red
}
