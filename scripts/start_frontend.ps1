param(
    [string]$BindAddress = "127.0.0.1",
    [int]$Port = 5173
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$frontendRoot = Join-Path $projectRoot "frontend"
$npmCommand = Get-Command npm.cmd -ErrorAction Stop

if (-not (Test-Path -LiteralPath (Join-Path $frontendRoot "package.json"))) {
    throw "Frontend package.json was not found: $frontendRoot"
}

Set-Location -LiteralPath $frontendRoot
Write-Host "Frontend: http://${BindAddress}:$Port"
& $npmCommand.Source run dev -- --host $BindAddress --port $Port
exit $LASTEXITCODE
