# ============================================
# Credit Scoring V2 - Start All Services
# ============================================
# Usage: Right-click -> Run with PowerShell
#    or: powershell -ExecutionPolicy Bypass -File start-all.ps1
# ============================================

$ROOT = (Resolve-Path -LiteralPath $PSScriptRoot).Path

function Start-CmdWindow {
    param(
        [Parameter(Mandatory = $true)][string]$Title,
        [Parameter(Mandatory = $true)][string]$WorkDir,
        [Parameter(Mandatory = $true)][string]$RunCommand
    )

    # Use cmd + pushd to support UNC paths. pushd auto-maps UNC to a temp drive.
    $cmd = "title $Title && pushd `"$WorkDir`" && $RunCommand"
    Start-Process -FilePath "cmd.exe" -ArgumentList "/K", $cmd -WindowStyle Normal
}

Write-Host "============================================" -ForegroundColor Cyan
Write-Host " Credit Scoring V2 - Starting All Services" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

# 1. Backend (Spring Boot) - Port 8080
Write-Host "[1/3] Starting Backend (Spring Boot) on port 8080..." -ForegroundColor Yellow
Start-CmdWindow -Title "CreditScoring Backend" -WorkDir "$ROOT\backend" -RunCommand "echo Backend starting... && call mvnw.cmd spring-boot:run"

Start-Sleep -Seconds 3

# 2. ML Service (FastAPI) - Port 8000
Write-Host "[2/3] Starting ML Service (FastAPI) on port 8000..." -ForegroundColor Yellow
Start-CmdWindow -Title "CreditScoring ML Service" -WorkDir "$ROOT\ml-service" -RunCommand "echo ML Service starting... && `"`"E:\Anaconda\envs\credit-scoring\python.exe`"`" main.py"

Start-Sleep -Seconds 2

# 3. Frontend (Vite + React) - Port 5173
Write-Host "[3/3] Starting Frontend (Vite + React) on port 5173..." -ForegroundColor Yellow
Start-CmdWindow -Title "CreditScoring Frontend" -WorkDir "$ROOT\frontend" -RunCommand "echo Frontend starting... && call npm run dev"

Write-Host ""
Write-Host "============================================" -ForegroundColor Green
Write-Host " All services launched!" -ForegroundColor Green
Write-Host "============================================" -ForegroundColor Green
Write-Host ""
Write-Host " Frontend:   http://localhost:5173" -ForegroundColor White
Write-Host " Backend:    http://localhost:8080" -ForegroundColor White
Write-Host " ML Service: http://localhost:8000" -ForegroundColor White
Write-Host ""
Write-Host " Login: admin / admin123" -ForegroundColor DarkGray
Write-Host ""
Write-Host "Each service runs in its own window." -ForegroundColor DarkGray
Write-Host "Close the windows to stop the services." -ForegroundColor DarkGray
