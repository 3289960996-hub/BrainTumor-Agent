param(
    [string]$BindAddress = "127.0.0.1",
    [int]$BackendPort = 8000,
    [int]$FrontendPort = 5173,
    [string]$RedisServerPath = $env:BTA_REDIS_SERVER,
    [switch]$NoBrowser
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$runtimeRoot = Join-Path $projectRoot "runtime"
$runRoot = Join-Path ([System.IO.Path]::GetTempPath()) "BrainTumor-Agent"
$logRoot = Join-Path $runRoot "logs"
$redisRoot = Join-Path $runRoot "redis"
$statePath = Join-Path $runRoot "services.json"

New-Item -ItemType Directory -Path $logRoot -Force | Out-Null
New-Item -ItemType Directory -Path $redisRoot -Force | Out-Null

function Test-TcpPort([string]$Address, [int]$Port) {
    $client = [System.Net.Sockets.TcpClient]::new()
    try {
        $task = $client.ConnectAsync($Address, $Port)
        return $task.Wait(800) -and $client.Connected
    } catch {
        return $false
    } finally {
        $client.Dispose()
    }
}

function Wait-Http([string]$Url, [string]$Name) {
    for ($attempt = 0; $attempt -lt 60; $attempt++) {
        try {
            $response = Invoke-WebRequest -Uri $Url -UseBasicParsing -TimeoutSec 2
            if ($response.StatusCode -ge 200 -and $response.StatusCode -lt 500) {
                return
            }
        } catch {}
        Start-Sleep -Milliseconds 500
    }
    throw "$Name did not become ready. See runtime/logs for details."
}

function Find-DockerCli {
    $command = Get-Command docker.exe -ErrorAction SilentlyContinue
    if ($command) {
        return $command.Source
    }
    $candidates = @(
        (Join-Path $env:LOCALAPPDATA "Programs\DockerDesktop\resources\bin\docker.exe"),
        "C:\Program Files\Docker\Docker\resources\bin\docker.exe",
        "E:\Docker\Docker\resources\bin\docker.exe"
    )
    return $candidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
}

function Wait-DockerEngine([string]$DockerCli) {
    $ready = $false
    try {
        & $DockerCli info *> $null
        $ready = $LASTEXITCODE -eq 0
    } catch {}
    if ($ready) {
        return
    }
    $desktopCandidates = @(
        (Join-Path $env:LOCALAPPDATA "Programs\DockerDesktop\Docker Desktop.exe"),
        "C:\Program Files\Docker\Docker\Docker Desktop.exe",
        "E:\Docker\Docker\Docker Desktop.exe"
    )
    $desktop = $desktopCandidates |
        Where-Object { Test-Path -LiteralPath $_ } |
        Select-Object -First 1
    if (-not $desktop) {
        throw "Docker Desktop is installed but its engine is not running."
    }
    Start-Process -FilePath $desktop
    for ($attempt = 0; $attempt -lt 90; $attempt++) {
        Start-Sleep -Seconds 2
        $ready = $false
        try {
            & $DockerCli info *> $null
            $ready = $LASTEXITCODE -eq 0
        } catch {}
        if ($ready) {
            return
        }
    }
    throw "Docker Desktop did not become ready within 3 minutes."
}

function Test-WorkerReady {
    $pythonPath = Join-Path $projectRoot ".venv\Scripts\python.exe"
    if (-not (Test-Path -LiteralPath $pythonPath -PathType Leaf)) {
        return $false
    }
    try {
        & $pythonPath -c @"
from backend.app.tasks.celery_app import celery_app
raise SystemExit(0 if celery_app.control.inspect(timeout=2).ping() else 1)
"@ 2>$null
        return $LASTEXITCODE -eq 0
    } catch {
        return $false
    }
}

function Wait-Worker {
    for ($attempt = 0; $attempt -lt 20; $attempt++) {
        if (Test-WorkerReady) {
            return
        }
        Start-Sleep -Seconds 1
    }
    throw "Analysis Worker did not become ready. See $logRoot\worker.err.log for details."
}

function Find-RedisServer([string]$ConfiguredPath) {
    if ($ConfiguredPath) {
        if (-not (Test-Path -LiteralPath $ConfiguredPath -PathType Leaf)) {
            throw "Configured Redis server was not found: $ConfiguredPath"
        }
        return (Resolve-Path -LiteralPath $ConfiguredPath).Path
    }

    $command = Get-Command redis-server.exe -ErrorAction SilentlyContinue
    if ($command) {
        return $command.Source
    }

    $candidates = @(
        (Join-Path $projectRoot "runtime\tools\redis\redis-server.exe"),
        (Join-Path $projectRoot "runtime\tools\memurai-package\Memurai\memurai.exe"),
        (Join-Path $projectRoot "runtime\tools\memurai\memurai.exe"),
        "E:\Redis\redis-server.exe",
        "E:\Program Files\Redis\redis-server.exe",
        "E:\Program Files\Memurai\memurai.exe",
        (Join-Path $env:ProgramFiles "Redis\redis-server.exe"),
        (Join-Path $env:ProgramFiles "Memurai\memurai.exe"),
        (Join-Path $env:LOCALAPPDATA "Programs\Redis\redis-server.exe")
    )
    return $candidates |
        Where-Object { Test-Path -LiteralPath $_ -PathType Leaf } |
        Select-Object -First 1
}

function Start-ManagedProcess(
    [string]$Name,
    [string]$FilePath,
    [string[]]$Arguments,
    [string]$WorkingDirectory = $projectRoot
) {
    $outputPath = Join-Path $logRoot "$Name.out.log"
    $errorPath = Join-Path $logRoot "$Name.err.log"
    return Start-Process `
        -FilePath $FilePath `
        -ArgumentList $Arguments `
        -WorkingDirectory $WorkingDirectory `
        -WindowStyle Hidden `
        -RedirectStandardOutput $outputPath `
        -RedirectStandardError $errorPath `
        -PassThru
}

if (Test-Path -LiteralPath $statePath -PathType Leaf) {
    $existing = Get-Content -LiteralPath $statePath -Raw | ConvertFrom-Json
    if (
        (Test-TcpPort $BindAddress $BackendPort) -and
        (Test-TcpPort $BindAddress $FrontendPort) -and
        (Test-WorkerReady)
    ) {
        $existingUrl = "http://${BindAddress}:$FrontendPort/"
        Write-Host "BrainTumor-Agent is already running: $existingUrl" -ForegroundColor Green
        exit 0
    }
    Remove-Item -LiteralPath $statePath -Force
}

$state = [ordered]@{
    started_at = [DateTimeOffset]::UtcNow.ToString("o")
    redis_mode = "external"
    redis_pid = $null
    backend_pid = $null
    frontend_pid = $null
    worker_pid = $null
}

if (-not (Test-TcpPort "127.0.0.1" 6379)) {
    $redisServer = Find-RedisServer $RedisServerPath
    if ($redisServer) {
        $redisProcess = Start-Process `
            -FilePath $redisServer `
            -ArgumentList @(
                "--bind", "127.0.0.1",
                "--protected-mode", "yes",
                "--port", "6379",
                "--appendonly", "yes",
                "--dir", $redisRoot
            ) `
            -WorkingDirectory $redisRoot `
            -WindowStyle Hidden `
            -RedirectStandardOutput (Join-Path $logRoot "redis.out.log") `
            -RedirectStandardError (Join-Path $logRoot "redis.err.log") `
            -PassThru
        $state.redis_mode = "process"
        $state.redis_pid = $redisProcess.Id
    } else {
        $docker = Find-DockerCli
        if (-not $docker) {
            throw "Redis is not running and neither redis-server nor Docker is installed. Install Docker Desktop once, then rerun this script."
        }
        Wait-DockerEngine $docker
        & $docker compose -f (Join-Path $projectRoot "compose.yaml") up -d redis
        if ($LASTEXITCODE -ne 0) {
            throw "Docker could not start Redis. Make sure Docker Desktop is running."
        }
        $state.redis_mode = "docker"
    }
    for ($attempt = 0; $attempt -lt 30 -and -not (Test-TcpPort "127.0.0.1" 6379); $attempt++) {
        Start-Sleep -Milliseconds 500
    }
    if (-not (Test-TcpPort "127.0.0.1" 6379)) {
        throw "Redis did not become ready on port 6379."
    }
}

if (-not (Test-TcpPort $BindAddress $BackendPort)) {
    $pythonPath = Join-Path $projectRoot ".venv\Scripts\python.exe"
    $backend = Start-ManagedProcess "backend" $pythonPath @(
        "-m", "uvicorn", "backend.app.main:app",
        "--host", $BindAddress,
        "--port", $BackendPort.ToString()
    )
    $state.backend_pid = $backend.Id
}

if (-not (Test-WorkerReady)) {
    $pythonPath = Join-Path $projectRoot ".venv\Scripts\python.exe"
    $worker = Start-ManagedProcess "worker" $pythonPath @(
        "-m", "celery",
        "-A", "backend.app.tasks.celery_app:celery_app",
        "worker",
        "--loglevel=INFO",
        "--pool=solo",
        "--concurrency=1",
        "--without-mingle",
        "--without-gossip"
    )
    $state.worker_pid = $worker.Id
}

if (-not (Test-TcpPort $BindAddress $FrontendPort)) {
    $nodePath = (Get-Command node.exe -ErrorAction Stop).Source
    $vitePath = Join-Path $projectRoot "frontend\node_modules\vite\bin\vite.js"
    $frontend = Start-ManagedProcess "frontend" $nodePath @(
        $vitePath,
        "--configLoader", "runner",
        "--host", $BindAddress,
        "--port", $FrontendPort.ToString()
    ) (Join-Path $projectRoot "frontend")
    $state.frontend_pid = $frontend.Id
}

$state | ConvertTo-Json | Set-Content -LiteralPath $statePath -Encoding utf8
Wait-Http "http://${BindAddress}:$BackendPort/api/v1/health" "Backend"
Wait-Http "http://${BindAddress}:$FrontendPort/" "Frontend"
Wait-Worker

$appUrl = "http://${BindAddress}:$FrontendPort/"
Write-Host "BrainTumor-Agent is ready: $appUrl" -ForegroundColor Green
Write-Host "Logs: $logRoot"
Write-Host "Stop: .\scripts\stop_all.ps1"
if (-not $NoBrowser) {
    Start-Process $appUrl
}
