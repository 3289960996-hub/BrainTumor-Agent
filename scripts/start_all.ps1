param(
    [string]$BindAddress = "127.0.0.1",
    [int]$BackendPort = 8000,
    [int]$FrontendPort = 5173,
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

if (Test-Path -LiteralPath $statePath -PathType Leaf) {
    $existing = Get-Content -LiteralPath $statePath -Raw | ConvertFrom-Json
    if ($existing.worker_pid -and (Get-Process -Id $existing.worker_pid -ErrorAction SilentlyContinue)) {
        $existingUrl = "http://${BindAddress}:$FrontendPort/"
        Write-Host "BrainTumor-Agent is already running: $existingUrl" -ForegroundColor Green
        exit 0
    }
    Remove-Item -LiteralPath $statePath -Force
}

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

function Start-ManagedProcess(
    [string]$Name,
    [string]$ScriptPath,
    [string[]]$ScriptArguments
) {
    $outputPath = Join-Path $logRoot "$Name.out.log"
    $errorPath = Join-Path $logRoot "$Name.err.log"
    $arguments = @(
        "-NoProfile",
        "-ExecutionPolicy", "Bypass",
        "-File", $ScriptPath
    ) + $ScriptArguments
    return Start-Process `
        -FilePath "powershell.exe" `
        -ArgumentList $arguments `
        -WorkingDirectory $projectRoot `
        -WindowStyle Hidden `
        -RedirectStandardOutput $outputPath `
        -RedirectStandardError $errorPath `
        -PassThru
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
    $redisServer = Get-Command redis-server.exe -ErrorAction SilentlyContinue
    if ($redisServer) {
        $redisProcess = Start-Process `
            -FilePath $redisServer.Source `
            -ArgumentList @("--port", "6379", "--appendonly", "yes") `
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
    $backend = Start-ManagedProcess "backend" (Join-Path $PSScriptRoot "start_backend.ps1") @(
        "-BindAddress", $BindAddress,
        "-Port", $BackendPort.ToString(),
        "-NoReload"
    )
    $state.backend_pid = $backend.Id
}

$worker = Start-ManagedProcess "worker" (Join-Path $PSScriptRoot "start_worker.ps1") @()
$state.worker_pid = $worker.Id

if (-not (Test-TcpPort $BindAddress $FrontendPort)) {
    $frontend = Start-ManagedProcess "frontend" (Join-Path $PSScriptRoot "start_frontend.ps1") @(
        "-BindAddress", $BindAddress,
        "-Port", $FrontendPort.ToString()
    )
    $state.frontend_pid = $frontend.Id
}

$state | ConvertTo-Json | Set-Content -LiteralPath $statePath -Encoding utf8
Wait-Http "http://${BindAddress}:$BackendPort/api/v1/health" "Backend"
Wait-Http "http://${BindAddress}:$FrontendPort/" "Frontend"

$appUrl = "http://${BindAddress}:$FrontendPort/"
Write-Host "BrainTumor-Agent is ready: $appUrl" -ForegroundColor Green
Write-Host "Logs: $logRoot"
Write-Host "Stop: .\scripts\stop_all.ps1"
if (-not $NoBrowser) {
    Start-Process $appUrl
}
