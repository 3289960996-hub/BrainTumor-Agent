param(
    [int]$Concurrency = 1
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$pythonPath = Join-Path $projectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $pythonPath -PathType Leaf)) {
    throw "Project Python was not found: $pythonPath"
}
if ($Concurrency -ne 1) {
    Write-Warning "MRI inference is resource intensive; concurrency 1 is recommended."
}

Set-Location -LiteralPath $projectRoot
$env:OMP_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
$env:OPENBLAS_NUM_THREADS = "1"
Write-Host "Starting MRI analysis worker (concurrency=$Concurrency)"
& $pythonPath -m celery -A backend.app.tasks.celery_app:celery_app worker `
    --loglevel=INFO `
    --pool=solo `
    --concurrency=$Concurrency `
    --without-mingle `
    --without-gossip
exit $LASTEXITCODE
