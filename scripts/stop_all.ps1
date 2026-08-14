$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$statePath = Join-Path ([System.IO.Path]::GetTempPath()) "BrainTumor-Agent\services.json"

function Stop-ProcessTree([int]$ProcessId) {
    $children = Get-CimInstance Win32_Process -Filter "ParentProcessId=$ProcessId" -ErrorAction SilentlyContinue
    foreach ($child in $children) {
        Stop-ProcessTree -ProcessId $child.ProcessId
    }
    Stop-Process -Id $ProcessId -Force -ErrorAction SilentlyContinue
}

if (-not (Test-Path -LiteralPath $statePath -PathType Leaf)) {
    Write-Host "No managed BrainTumor-Agent services are recorded."
    exit 0
}

$state = Get-Content -LiteralPath $statePath -Raw | ConvertFrom-Json
foreach ($property in @("frontend_pid", "worker_pid", "backend_pid", "redis_pid")) {
    $processId = $state.$property
    if ($processId) {
        Stop-ProcessTree -ProcessId ([int]$processId)
    }
}

if ($state.redis_mode -eq "docker") {
    $dockerCommand = Get-Command docker.exe -ErrorAction SilentlyContinue
    $dockerPath = if ($dockerCommand) {
        $dockerCommand.Source
    } else {
        Join-Path $env:LOCALAPPDATA "Programs\DockerDesktop\resources\bin\docker.exe"
    }
    if (Test-Path -LiteralPath $dockerPath) {
        & $dockerPath compose -f (Join-Path $projectRoot "compose.yaml") stop redis
    }
}

Remove-Item -LiteralPath $statePath -Force
Write-Host "Managed BrainTumor-Agent services have stopped." -ForegroundColor Green
