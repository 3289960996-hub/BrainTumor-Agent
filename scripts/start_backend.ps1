param(
    [string]$BindAddress = "127.0.0.1",
    [int]$Port = 8000,
    [switch]$NoReload
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$pythonPath = Join-Path $projectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $pythonPath -PathType Leaf)) {
    throw "Project Python was not found: $pythonPath"
}

Set-Location -LiteralPath $projectRoot
$uvicornArgs = @(
    "-m", "uvicorn", "backend.app.main:app",
    "--host", $BindAddress,
    "--port", $Port.ToString()
)
if (-not $NoReload) {
    $uvicornArgs += "--reload"
}

Write-Host "Backend: http://${BindAddress}:$Port"
& $pythonPath @uvicornArgs
exit $LASTEXITCODE
